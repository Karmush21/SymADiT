"""Copyright (c) Meta Platforms, Inc. and affiliates."""
import os
from contextlib import redirect_stdout, redirect_stderr
import copy
from typing import Any, Dict, Literal
import math
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from src.crystal.crystal_reconstruction import CrystalReconstructionEvaluator
from src.utils import pylogger
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_scatter import scatter
from torchmetrics import MeanMetric
from pyxtal import pyxtal

from src.ae_regularization.latent_space import saturate_latent, DiagonalGaussianDistribution
from src.symmetry.atom import z_to_symbol
from src.symmetry.cell import cell_mask_from_sg
from src.symmetry.wyckoff_data import load_or_create_wyckoff_data




log = pylogger.RankedLogger(__name__)



class SymAutoencoder(LightningModule):
    """LightningModule for autoencoding 3D atomic systems. Implements AE loss.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        pos_loss: str,
        cell_loss: str,
        latent_dim: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        use_regularized_ae: bool,
        use_regularized_kl: bool,
        loss_weights: Dict,
        augmentations: DictConfig,
        compile: bool,
        debug: bool = False,
    ) -> None:
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # Only save parameters needed for latent flow matching
        self.save_hyperparameters(logger=False)

        # encoder and decoder models | These are nn.modules
        self.encoder = encoder
        self.decoder = decoder
        self.pos_loss = pos_loss
        self.cell_loss = cell_loss
        self.loss_weights = loss_weights
        self.augmentations = augmentations
        self.compile_model = compile
        self.debug = debug
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.scheduler_frequency = scheduler_frequency
        

        
        if self.hparams.use_regularized_ae and self.hparams.use_regularized_kl:
            raise ValueError(
                "use_regularized_ae and use_regularized_kl cannot both be True."
            )


        

        
        if self.hparams.use_regularized_kl:
            self.quant_conv = torch.nn.Linear(self.encoder.d_model, 2 * latent_dim, bias=False)
        
        # If we use the saturate function instead
        else:
            self.quant_conv = torch.nn.Linear(self.encoder.d_model, latent_dim, bias=False)

        
        # 
        self.post_quant_conv = torch.nn.Linear(latent_dim, self.decoder.d_model, bias=False)


        log.info(f"Using {self.pos_loss} as loss function for posistion")


        # evaluator objects for computing metrics
        self.val_reconstruction_evaluator = CrystalReconstructionEvaluator()
        self.test_reconstruction_evaluator = CrystalReconstructionEvaluator()






        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                
                "loss_atom_types": MeanMetric(),
                "loss_wyckoff": MeanMetric(),
                "loss_pos": MeanMetric(),
                "loss_lengths": MeanMetric(),
                "loss_angles": MeanMetric(),
                

                "reg_loss_kl": MeanMetric(),

            }
        )
        self.val_metrics = ModuleDict(
            {
                "loss": MeanMetric(),

                "loss_atom_types": MeanMetric(),
                "loss_wyckoff": MeanMetric(),
                "loss_pos": MeanMetric(),
                
                #"loss_cell": MeanMetric(),
                "loss_lengths": MeanMetric(),
                "loss_angles": MeanMetric(),
                
                "reg_loss_kl": MeanMetric(),

                "match_rate": MeanMetric(),
                "rms_dist": MeanMetric(),
                "valid": MeanMetric(),
                "comp_valid": MeanMetric(),
                "struct_valid": MeanMetric(),
                "validity": MeanMetric(),

                "pyxtal_build_rate": MeanMetric(), 



            }
        )
        
        self.test_metrics = copy.deepcopy(self.val_metrics)
    
    def _reset_stage_metrics(self, stage: Literal["train", "val", "test"]):
        metrics = getattr(self, f"{stage}_metrics")

        for metric in metrics.values():
            metric.reset()

        if stage in ["val", "test"]:
            reconstruction_evaluator = getattr(
                self,
                f"{stage}_reconstruction_evaluator",
            )
            reconstruction_evaluator.clear()

        

    def _is_last_train_epoch(self) -> bool:
        t = self.trainer
        return (
            t is not None
            and not t.sanity_checking
            and t.max_epochs is not None
            and (t.current_epoch + 1) >= int(t.max_epochs)  # epochs are 0-indexed
        )


    def encode(self, batch):
        encoded_batch = self.encoder(batch)

        encoded_batch["moments"] = self.quant_conv(encoded_batch["x"])

        if self.hparams.use_regularized_kl:
            posterior = DiagonalGaussianDistribution(encoded_batch["moments"], deterministic=False)

            encoded_batch["posterior"] = posterior
            encoded_batch["x"] = posterior.sample()

        elif self.hparams.use_regularized_ae:
            encoded_batch["x"] = saturate_latent(encoded_batch["moments"])

        # Do nothing
        else:
            encoded_batch["x"] = encoded_batch["moments"]

        return encoded_batch


    def decode(self, encoded_batch):
        
        decoder_batch = encoded_batch.copy()

        decoder_batch["x"] = self.post_quant_conv(encoded_batch["x"])
        
        out = self.decoder(decoder_batch)

        return out


    def forward(self, batch: Data):
        encoded_batch = self.encode(batch)

        out = self.decode(encoded_batch)

        return out, encoded_batch

    @staticmethod
    def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str, periodic: bool = False, beta: float = 0.02) -> torch.Tensor:

        delta = pred - target

        if periodic:
            delta = (delta + 0.5) % 1.0 - 0.5

        if loss_type == "l1":
            loss = delta.abs()

        elif loss_type == "l2":
            loss = delta.pow(2)

        elif loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none", beta=beta)

        elif loss_type == "periodic" and periodic:
            loss = 1.0 - torch.cos(2.0 * math.pi * delta)

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        mask = mask.float()
        dof = mask.sum(dim=1).clamp_min(1.0)

        return (loss * mask).sum(dim=1) / dof

    
    #####################################################################################################

    def reconstruction_criterion(self, batch: Data, out: Dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
            Reconstruction losses, returned as per-graph losses (shape: [num_graphs]).
        """

        loss_atom_types_per_atom = F.cross_entropy(out["atom_types"], batch.atom_types, reduction="none")
        loss_atom_types = scatter(loss_atom_types_per_atom, batch.batch, dim=0, reduce="mean")



        logits_w = out["wyckoff"] 
        targets_w = batch.wyckoff_ids    

        # Map each atom to its graph's space group (ground truth)
        sg_per_atom = batch.sg_number[batch.batch] 

        # Build a boolean mask of valid Wyckoff tokens per atom
        mask = torch.zeros_like(logits_w, dtype=torch.bool)  # [N_atoms, num_wp_tokens]

        unique_sgs = sg_per_atom.unique()
        for sg in unique_sgs:
            sg_int = int(sg.item())
            atom_idx = (sg_per_atom == sg)
            wy_start, wy_end = self.index_dict[sg_int]
            mask[atom_idx, wy_start:wy_end] = True

        # Set invalid logits to a very large negative number so softmax ~ 0 there
        masked_logits_w = logits_w.masked_fill(~mask, -1e9)

        loss_wyckoff_per_atom = F.cross_entropy(masked_logits_w, targets_w, reduction="none")

        loss_wyckoff = scatter(loss_wyckoff_per_atom, batch.batch, dim=0, reduce="mean")


        
        # Lattice reconstruction
        loss_lengths = self.masked_reconstruction_loss(pred=out["lengths_pred"], target=batch.full_cell_rad[:, :3], mask=batch.full_cell_mask[:, :3], loss_type=self.cell_loss)

        loss_angles = self.masked_reconstruction_loss(pred=out["angles_pred"], target=batch.full_cell_rad[:, 3:], mask=batch.full_cell_mask[:, 3:], loss_type=self.cell_loss)


        # Position reconstruction
        loss_pos_per_node = self.masked_reconstruction_loss(pred=out["pos_pred"], target=batch.full_rep, mask=batch.full_rep_mask, loss_type=self.pos_loss, periodic=True)

        loss_pos = scatter(loss_pos_per_node, batch.batch, dim=0, reduce="mean")

        return {
            "loss_atom_types":  loss_atom_types,
            "loss_wyckoff":     loss_wyckoff,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
            "loss_pos":         loss_pos,
            
        }

    # Here is the loss calculation
    def criterion(self, batch: Data, encoded_batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Reconstruction loss
        loss_reconst = self.reconstruction_criterion(batch, out)
        
        # NOTE ADiT kept kl here
        if self.hparams.use_regularized_kl:
            loss_kl_per_node = encoded_batch["posterior"].kl()
            loss_kl = scatter(loss_kl_per_node, batch.batch, dim=0, dim_size=batch.num_graphs, reduce="mean")
        else:
            loss_kl = out["pos_pred"].new_zeros(batch.num_graphs)

        # per-atom losses, # NOTE We index per node here 
        w_atom   = self.loss_weights["loss_atom_types"]
        w_wy     = self.loss_weights["loss_wyckoff"]
        w_pos    = self.loss_weights["loss_pos"]
        w_c      = self.loss_weights["loss_cell"]
        
        w_kl = self.loss_weights["loss_kl"] if self.hparams.use_regularized_kl else 0.0
        
        

        loss = (
        w_atom * loss_reconst["loss_atom_types"].mean()
        + w_wy   * loss_reconst["loss_wyckoff"].mean()
        + w_pos  * loss_reconst["loss_pos"].mean()
        + w_c  * loss_reconst["loss_lengths"].mean() 
        + w_c  * loss_reconst["loss_angles"].mean() 
        
        + w_kl   * loss_kl.mean()
        )

        return {
        "loss": loss,
        "loss_atom_types": w_atom * loss_reconst["loss_atom_types"],
        "loss_wyckoff":    w_wy   * loss_reconst["loss_wyckoff"],
        "loss_pos":        w_pos  * loss_reconst["loss_pos"],
        "loss_lengths":        w_c  * loss_reconst["loss_lengths"],
        "loss_angles":        w_c  * loss_reconst["loss_angles"],
        
        }
    

    #####################################################################################################


    

    def on_train_epoch_start(self) -> None:
        self._reset_stage_metrics("train")

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:

        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        
        if self.augmentations.do_augmentations:
            with torch.no_grad():
                clean_atom_types = batch.atom_types.clone()
                clean_free_rep_pad = batch.free_rep_pad.clone()
                clean_free_cell_pad = batch.free_cell_pad.clone()

                # 1) Jitter free representative DOFs
                pos_sigma = getattr(self.augmentations, "pos_sigma", 0.0)
                if pos_sigma > 0.0:
                    pos_noise = torch.randn_like(batch.free_rep_pad, device=self.device) * pos_sigma
                    pos_noise = pos_noise * batch.free_rep_mask_pad
                    batch.free_rep_pad = torch.remainder(batch.free_rep_pad + pos_noise, 1.0)

                # 2) Jitter free cell DOFs
                cell_len_sigma = getattr(self.augmentations, "cell_len_sigma", 0.0)
                cell_ang_sigma = getattr(self.augmentations, "cell_ang_sigma", 0.0)

                if cell_len_sigma > 0.0 or cell_ang_sigma > 0.0:
                    cell_noise = torch.zeros_like(batch.free_cell_pad, device=self.device)

                    for g in range(batch.free_cell_pad.shape[0]):
                        n_len = int(batch.num_dof_lengths[g].item())   # graph-level scalar
                        n_ang = int(batch.num_dof_angles[g].item())    # graph-level scalar

                        if n_len > 0:
                            cell_noise[g, :n_len] = torch.randn(n_len, device=self.device) * cell_len_sigma

                        if n_ang > 0:
                            start = n_len
                            end = n_len + n_ang
                            cell_noise[g, start:end] = torch.randn(n_ang, device=self.device) * cell_ang_sigma

                    cell_noise = cell_noise * batch.free_cell_mask_pad
                    batch.free_cell_pad = batch.free_cell_pad + cell_noise

                # NOTE Remove for now, something ADiT used
                # 3) optional atom masking
                # type_mask_frac = getattr(self.augmentations, "type_mask_frac", 0.0)
                # if type_mask_frac > 0.0:
                #     total_atoms = batch.atom_types.size(0)
                #     n_mask = int(total_atoms * type_mask_frac)
                #     if n_mask > 0:
                #         mask_idx = torch.randperm(total_atoms, device=self.device)[:n_mask]
                #         batch.atom_types[mask_idx] = 0


        out, encoded_batch = self.forward(batch)

        # Restore clean targets for loss
        if self.augmentations.do_augmentations:
            batch.atom_types = clean_atom_types
            batch.free_rep_pad   = clean_free_rep_pad
            batch.free_cell_pad  = clean_free_cell_pad

        
        loss_dict = self.criterion(batch, encoded_batch, out)


        # update and log train metrics
        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False if k != "loss" else True,
            )

        return loss_dict["loss"]
    

    #####################################################################################################



    def on_validation_epoch_start(self) -> None:
        self._reset_stage_metrics("val")
    
    
    def validation_step(self, batch: Data, batch_idx: int) -> None:
        self.evaluation_step(batch, batch_idx, stage="val")
    

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        self._reset_stage_metrics("test")

    def test_step(self, batch: Data, batch_idx: int) -> None:
        self.evaluation_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################



    def evaluation_step(self, batch: Data, batch_idx: int, stage: Literal["val", "test"]) -> None:
        """Perform a single evaluation step on a batch of data from the validation/test set."""
        
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        
        
        metrics = getattr(self, f"{stage}_metrics")
        reconstruction_evaluator = getattr(self, f"{stage}_reconstruction_evaluator")
        reconstruction_evaluator.device = metrics["loss"].device

        # forward pass
        out, encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(batch, encoded_batch, out)

        # update and log per-step val metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

   
        num_orbits = batch["num_orbits"].tolist()
        start_idx = 0       
        for idx_in_batch, num_sites in enumerate(num_orbits):

            id = batch.id[idx_in_batch]
                        

            sg_gt = int(batch.sg_number[idx_in_batch].item()) 

            # --------------------------
            # 1) Wyckoff prediction
            # --------------------------
            wy_logits = out["wyckoff"].narrow(0, start_idx, num_sites) 
            wy_start, wy_end = self.index_dict[sg_gt]
            local_logits = wy_logits[:, wy_start:wy_end]            
            local_idx = local_logits.argmax(dim=1)                     
            global_idx = local_idx + wy_start                          
            wy_pred_tokens = [self.flat_pairs[i.item()] for i in global_idx]
            wy_pred_clean  = [s.split("_", 1)[1] for s in wy_pred_tokens]  # 'sg_4a' -> '4a'

            # --------------------------
            # 2) Get model predictions for this graph
            # --------------------------
            atom_logits = out["atom_types"].narrow(0, start_idx, num_sites)     # [num_sites, ...]
            cell_pred   = out["cell_pred"][idx_in_batch]                        # [6] (radians)
            pos_pred    = out["pos_pred"].narrow(0, start_idx, num_sites)       # [num_sites, 3]


            atom_type_idx = atom_logits.argmax(dim=1)
            atom_type_idx[atom_type_idx == 0] = 1
            atom_types_list = z_to_symbol(atom_type_idx)  # list[str] length = num_sites

            # --------------------------
            # 3) Build FREE-DOF vector x using masks derived from GT symmetry
            # --------------------------

            pos_mask = batch.full_rep_mask.narrow(0, start_idx, num_sites)
            pos_pred_wrapped = torch.remainder(pos_pred, 1.0)
            pos_free_pyxtal = pos_pred_wrapped.reshape(-1)[pos_mask.reshape(-1).bool()]

            cell_deg = cell_pred.clone()
            cell_deg[3:] = torch.rad2deg(cell_deg[3:])
            cell_mask_np = cell_mask_from_sg(sg_gt)
            cell_mask = torch.as_tensor(cell_mask_np, device=cell_deg.device, dtype=torch.bool)
            cell_free = cell_deg[cell_mask] 

            x = torch.cat([cell_free, pos_free_pyxtal], dim=0).detach().cpu().numpy()
            
            # --------------------------
            # 4) Build pyxtal crystal using GT symmetry + predicted free DOFs
            # --------------------------
            sample_idx = batch_idx * batch.batch_size + idx_in_batch

            reconstruction_evaluator.append_pyxtal_pred_array({"sample_idx": sample_idx, 
                                                         "sg":sg_gt,
                                                         "wyckoff":wy_pred_clean, 
                                                         "pos_free":pos_free_pyxtal, 
                                                         "cell_free":cell_free, 
                                                         "atom_types":atom_types_list,
                                                         "id": id})

            
            
            c_pred = pyxtal()
            try:
                with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                    c_pred.from_spg_wps_rep(sg_gt, wy_pred_clean, x, atom_types_list)
            except Exception as e:
                c_pred = None

            
            success = 1.0 if c_pred is not None else 0.0
            success_tensor = torch.tensor(success, device=self.device)
            metrics["pyxtal_build_rate"](success_tensor)


            track_spacegroup = c_pred.group.number if c_pred is not None else 0
                        
            reconstruction_evaluator.append_pred_array(
                {
                    "pyxtal_crys": c_pred,
                    "sample_idx": sample_idx,
                    "spacegroup": track_spacegroup,
                    "id": id,
                }
            )
            
            start_idx = start_idx + num_sites
        
        self.log(
            f"{stage}/pyxtal_build_rate",
            metrics["pyxtal_build_rate"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        
        # Save groundtruths for metrics and visualisation (see validation_epoch_end)
        for idx_in_batch, _data in enumerate(batch.to_data_list()):
            
            id = _data.id
            spacegroup = _data.sg_number.item() 
            wyckoff_indices = _data.wyckoff_ids.detach().cpu().tolist()
            wyckoff_sites = [self.flat_pairs[i] for i in wyckoff_indices]
            wyckoff_list = [s.split('_', 1)[1] for s in wyckoff_sites] # remove "sg_" from the string
            

            x_pyxtal = np.concatenate([_data.x_cell_free_dof, _data.x_pos_free_dof], axis=0)

            atom_list =  z_to_symbol(_data.atom_types)

            sample_idx =  (batch_idx) * batch.batch_size + idx_in_batch

            reconstruction_evaluator.append_pyxtal_gt_array({"sample_idx":sample_idx, 
                                   "sg":spacegroup,
                                   "wyckoff":wyckoff_list, 
                                   "pos_free":torch.from_numpy(_data.x_pos_free_dof),
                                    "cell_free":torch.from_numpy(_data.x_cell_free_dof), 
                                    "atom_types":atom_list})
            
            c_gt = pyxtal()
            c_gt.from_spg_wps_rep(spacegroup, wyckoff_list, x_pyxtal, atom_list)
            
            
            reconstruction_evaluator.append_gt_array(
                {
                    "pyxtal_crys": c_gt,
                    "sample_idx": sample_idx,
                    "spacegroup": spacegroup,
                    "id": id,
                }
            )



    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        '''
            Lightning hook that is called when a validation/test epoch ends.
        '''

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        
        
        
        metrics = getattr(self, f"{stage}_metrics")
        reconstruction_evaluator = getattr(self, f"{stage}_reconstruction_evaluator")
        reconstruction_evaluator.device = metrics["loss"].device
        
        
        do_save = (stage == "val" and self._is_last_train_epoch()) or (stage == "test")

        # Run the metrics on the reconstructed crystals and compare to ground truth
        rec_metrics_dict = reconstruction_evaluator.get_metrics(debug=self.debug)


        for k, v in rec_metrics_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=(k == "match_rate"),
                sync_dist=True,
            )
        
        
        # Currently only save things during the last epoch. 
        if do_save and isinstance(self.logger, WandbLogger) and self.trainer.is_global_zero:
            print("This print because sometimes wandb is not nice")

            pred_table = reconstruction_evaluator.get_wandb_table(current_epoch=self.current_epoch)

            pyxtal_pred_table = reconstruction_evaluator.get_pyxtal_wandb_table(current_epoch=self.current_epoch)

            self.logger.experiment.log({f"{stage}_pred_table_device{self.global_rank}": pred_table})
            
            self.logger.experiment.log({f"{stage}_pyxtal_table_epoch_{self.current_epoch}": pyxtal_pred_table})

                        



    #####################################################################################################

    def setup(self, stage: str) -> None:

        # Load static Wyckoff metadata once per process
        if not hasattr(self, "flat_pairs"):
            wyckoff_data = load_or_create_wyckoff_data()

            self.flat_pairs = wyckoff_data["flat_pairs"]
            self.index_dict = wyckoff_data["index_dict"]

        if self.compile_model and stage == "fit":
            self.encode = torch.compile(self.encode)
            self.decode = torch.compile(self.decode)
            self.quant_conv = torch.compile(self.quant_conv)
            self.post_quant_conv = torch.compile(self.post_quant_conv)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer_factory(params=self.trainer.model.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/rms_dist",
                    "interval": "epoch",
                    "frequency": self.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}




