"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy
import os
import random
import time
from typing import Any, Dict, Literal
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric, SumMetric
from tqdm import tqdm
from pyxtal import pyxtal
from contextlib import redirect_stdout, redirect_stderr


from src.crystal.crystal_generation import CrystalGenerationEvaluator
from src.models.ae_module import SymAutoencoder
from src.utils import pylogger
from src.utils.generation_utils import (
    sample_spacegroups_and_orbits, 
    sample_wyckoffs_and_atoms
)
from src.utils.create_random_ae import create_random_autoencoder


from src.symmetry.wyckoff_data import load_or_create_wyckoff_data
from src.symmetry.wyckoff_masks import extract_masks
from src.symmetry.cell import cell_mask_from_sg




log = pylogger.RankedLogger(__name__)






def saturate_latent(z: torch.Tensor, B: float = 5.0) -> torch.Tensor:
    return z / torch.sqrt(1.0 + (z * z) / (B * B))


class LatentDiffusionLitModule(LightningModule):
    """LightningModule for latent diffusion generative modellling of 3D atomic systems.

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
        autoencoder_ckpt: str,
        x_loss: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
        debug: bool,
        novelty: DictConfig,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        
        self.autoencoder_ckpt = autoencoder_ckpt 


        if autoencoder_ckpt:
            log.info(f"Loading Autoencoder ckpt: {self.autoencoder_ckpt}")
            self.autoencoder = SymAutoencoder.load_from_checkpoint(
                autoencoder_ckpt, map_location="cpu"
            )
        else:
            log.warning("\nNO AUTOENCODER CHECKPOINT PROVIDED. INITIALZIING NEW AUTOENCODER INSTEAD. PURELY FOR DEBUGGING\n")
            self.autoencoder = create_random_autoencoder()

        # freeze autoencoder
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()


        self.denoiser = denoiser

        self.interpolant = interpolant

    

        
        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "org_loss_mse": MeanMetric(),
                "geom_pos": MeanMetric(),
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
                {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "org_loss_mse": MeanMetric(),
                "geom_pos": MeanMetric(),
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
                "valid_rate": MeanMetric(),
                "struct_valid_rate": MeanMetric(),
                "comp_valid_rate": MeanMetric(),
                "unique_rate": MeanMetric(),
                "novel_rate": MeanMetric(),
                "sampling_time": MeanMetric(),

                #-----------
                "pyxtal_build_rate": MeanMetric(), 
                "pyxtal_build_count": SumMetric(), 


                "sg_pred_vs_cond_acc": MeanMetric(),

                "p1_rate": MeanMetric(), 
                }
            )
             

        self.test_metrics = copy.deepcopy(self.val_metrics)



    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _is_last_train_epoch(self) -> bool:
        t = self.trainer
        return (
            t is not None
            and not t.sanity_checking
            and t.max_epochs is not None
            and (t.current_epoch + 1) >= int(t.max_epochs)  # epochs are 0-indexed
        )


    def _reset_eval_metrics(self) -> None:
        for d in self.val_metrics.values():
            d.reset()
        
        for d in self.test_metrics.values():
            d.reset()



    def forward(self, batch: Data):

        # Encode batch to latent space
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            
            if self.autoencoder.hparams.use_regularized_kl:
                posterior = encoded_batch["posterior"]

                encoded_batch["x"] = posterior.sample()

            elif self.autoencoder.hparams.use_regularized_ae:
                encoded_batch["x"] = saturate_latent(encoded_batch["moments"])
            
            else:
                # Plain deterministic autoencoder:
                # pass the projected latent directly to the decoder.
                encoded_batch["x"] = encoded_batch["moments"]
            
                
            x_1 = encoded_batch["x"]

            # Convert from PyG batch to dense batch with padding
            x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])
            dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}



        # Corrupt batch using the interpolant
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)



        spacegroup = batch.sg_number


        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):

            # self-conditioning
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    spacegroup=spacegroup,
                    mask=mask,
                    x_sc=None,
                )
        else:
            x_sc = None

        # Run denoiser model
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            spacegroup=spacegroup,
            mask=mask,
            x_sc=x_sc,
        )

        return pred_x, noisy_dense_encoded_batch

    

    def criterion(
        self,
        noisy_dense_encoded_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
        pred_out,
        batch,
    ) -> dict[str, torch.Tensor]:

        gt_x_1 = noisy_dense_encoded_batch["x_1"]

        norm_scale = 1 - torch.min(
            noisy_dense_encoded_batch["t"].unsqueeze(-1),
            torch.tensor(0.9, device=pred_x.device, dtype=pred_x.dtype),
        )
        x_error = (gt_x_1 - pred_x) / norm_scale 

        # Masking and averaging
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] *
            noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)

        if self.hparams.x_loss == "l2":
            per_entry = x_error.pow(2)

        elif self.hparams.x_loss == "l1":
            per_entry = x_error.abs()

        elif self.hparams.x_loss == "smooth_l1":
            beta = getattr(self.hparams, "smooth_l1_beta", 0.02)
            per_entry = F.smooth_l1_loss(
                x_error,
                torch.zeros_like(x_error),
                reduction="none",
                beta=beta,
            )

        else:
            raise ValueError(f"Wrong loss: {self.hparams.x_loss}")

        x_loss = torch.sum(per_entry * loss_mask[..., None], dim=(-1, -2)) / loss_denom

        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        # add diffusion loss stratified across t
        num_bins = 4
        flat_losses = x_loss.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)

        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"x_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss

        loss_dict["t_avg"] = np.mean(flat_t)

        return loss_dict

    
    def on_fit_start(self) -> None:
        """Called before both training and the validation sanity check."""
        self._reset_eval_metrics()


    #####################################################################################################

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        noisy_batch = {
            "x": pred_x[noisy_dense_encoded_batch["token_mask"]],
            "batch": batch.batch,
            "token_idx": batch.token_idx
        }
        
        pred_out = self.autoencoder.decode(noisy_batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x, pred_out, batch)

        
        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=False if k != "loss" else True,
            )

        return loss_dict["loss"]

    #####################################################################################################

    
    
    def on_validation_epoch_start(self) -> None:
        """Called when the val epoch begins."""
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        self.evaluation_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        """Called when the val epoch ends."""
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################


    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch: Data, batch_idx: int) -> None:
        self.evaluation_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch starts."""
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")

        # Reset metrics
        metrics = getattr(self, f"{stage}_metrics")
        for metric in metrics.values():
            metric.reset()

        # Clear generation evaluator
        generation_evaluator = getattr(self, f"{stage}_generation_evaluator")
        generation_evaluator.clear()

    def evaluation_step(self, batch: Data, batch_idx: int, stage: Literal["val", "test"]) -> None:
        """Perform a single evaluation step on a batch of data from the validation/test set."""
        
    
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")

        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluator = getattr(self, f"{stage}_generation_evaluator")
        generation_evaluator.device = metrics["loss"].device

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        noisy_batch = {
            "x": pred_x[noisy_dense_encoded_batch["token_mask"]],
            "batch": batch.batch,
            "token_idx": batch.token_idx
        }
        
        pred_out = self.autoencoder.decode(noisy_batch)


        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x, pred_out, batch)

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
    

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluator = getattr(self, f"{stage}_generation_evaluator")


        save_now = (stage == "val" and self._is_last_train_epoch()) or (stage == "test")


        
        idx = 0
        
        generation_evaluator.device = metrics["loss"].device
        t_start = time.time()
        for samples_so_far in tqdm(
            range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
            desc=f"    Sampling"):
            
            # Perform sampling and decoding to crystal structures
            out, spacegroup, num_orbits = self.sample_and_decode(
                spacegroups_bincount=self.spacegroups_bincount,
                sg_num_orbits_conditionals = self.sg_num_orbits_conditionals,
                batch_size=self.hparams.sampling.batch_size,
                cfg_scale=self.hparams.sampling.cfg_scale,
                allowed_spacegroups=self.hparams.conditioning.allowed_spacegroups,
                forbidden_spacegroups=self.hparams.conditioning.forbidden_spacegroups
            )

            start_idx = 0
            self.pyxtal_pred_list = []
            for idx_in_batch, num_sites in enumerate(num_orbits.tolist()):

                _spacegroup = spacegroup[idx_in_batch].item()

                wy_logits = out["wyckoff"].narrow(0, start_idx, num_sites)
                atom_logits = out["atom_types"].narrow(0, start_idx, num_sites)

                sampled_wyckoffs_and_atoms = sample_wyckoffs_and_atoms(
                    wy_logits=wy_logits,
                    atom_logits=atom_logits,
                    INDEX_DICT = self.sg_to_wyckoff_range,
                    FLAT_PAIRS = self.wyckoff_tokens,
                    WYCKOFF_POS_MASK = self.wyckoff_pos_mask,
                    spacegroup=_spacegroup,
                    wyckoff_temperature=self.hparams.sampling.temperature.wyckoff,
                    atom_temperature=self.hparams.sampling.temperature.atom_type,
                    device=self.device,
                )

                
                wyckoff_types = sampled_wyckoffs_and_atoms["wyckoff_types"]
                wyckoff_types_clean = sampled_wyckoffs_and_atoms["wyckoff_types_clean"]
                
                atom_types_list = sampled_wyckoffs_and_atoms["atom_types_list"]
                
                
                pos_pred    = out["pos_pred"].narrow(0, start_idx, num_sites)
                pos_mask = extract_masks(wyckoff_types, self.wyckoff_pos_mask, self.device)
                pos_free = pos_pred.reshape(-1)[pos_mask.reshape(-1).bool()] 
                pos_free_pyxtal = torch.remainder(pos_free, 1.0)

                cell_pred   = out["cell_pred"][idx_in_batch]
                cell_deg = cell_pred.clone()
                cell_deg[3:] = torch.rad2deg(cell_deg[3:])

                cell_mask_np = cell_mask_from_sg(_spacegroup)
                cell_mask = torch.as_tensor(cell_mask_np, device=cell_deg.device, dtype=torch.bool)
                cell_free_pyxtal = cell_deg[cell_mask] 

                x = torch.cat([cell_free_pyxtal, pos_free_pyxtal], dim=0).detach().cpu().numpy()

               
                sample_idx = idx

                
                generation_evaluator.append_pyxtal_list({"sample_idx": sample_idx, 
                                                                "sg":_spacegroup, 
                                                                "wyckoff":wyckoff_types_clean, 
                                                                "pos_free":pos_free_pyxtal, 
                                                                "cell_free":cell_free_pyxtal, 
                                                                "atom_types":atom_types_list})


                c_pred = pyxtal()
                try:
                    with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                        c_pred.from_spg_wps_rep(
                            _spacegroup,
                            wyckoff_types_clean,
                            x,
                            atom_types_list,
                        )
                except Exception as e:
                    c_pred = None

                
                success = 1.0 if c_pred is not None else 0.0
                success_tensor = torch.tensor(success, device=self.device)

                metrics["pyxtal_build_rate"](success_tensor)
                metrics["pyxtal_build_count"](success_tensor)

                generation_evaluator.append_pred_array(
                    {
                        "pyxtal_crys": c_pred,
                        "sample_idx": sample_idx,
                        "spacegroup": int(_spacegroup)
                    }
                )
                
                start_idx = start_idx + num_sites
        
                idx += 1
            
            
            
            
        self.log(
            f"{stage}/pyxtal_build_rate",
            metrics["pyxtal_build_rate"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            f"{stage}/pyxtal_build_count",
            metrics["pyxtal_build_count"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )


        
        t_end = time.time()


        # Compute generation metrics
        gen_metrics_dict = generation_evaluator.get_metrics(save=False, debug = self.autoencoder.hparams.debug)
        
        gen_metrics_dict["sampling_time"] = t_end - t_start

        
       
        for k, v in gen_metrics_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False if k != "valid_rate" else True,
                sync_dist=True,
            )

        # Save only from the first rank
        if save_now and type(self.logger) == WandbLogger and self.trainer.is_global_zero:
            print("This print because sometimes wandb is not nice")
            
            pred_table = generation_evaluator.get_wandb_table(
                current_epoch=self.current_epoch
            )

            pyxtal_pred_table = generation_evaluator.get_pyxtal_wandb_table(
                current_epoch = self.current_epoch
            )
            
            
            self.logger.experiment.log(
                {f"{stage}_samples_table_device{self.global_rank}": pred_table}
            )

            self.logger.experiment.log(
                    {f"{stage}_pyxtal_table_epoch_{self.current_epoch}": pyxtal_pred_table}
                )

    #####################################################################################################

    def sample_and_decode(
        self,
        spacegroups_bincount,
        sg_num_orbits_conditionals,
        batch_size,
        cfg_scale=2.0,
        num_timesteps = None,
        allowed_spacegroups = None,
        forbidden_spacegroups = None,
        presampled_spacegroup=None,
        presampled_num_orbits=None,
    ):      
        
        # ------------------------------------------------------------------
        # Obtain generation metadata
        # ------------------------------------------------------------------
        if presampled_spacegroup is None and presampled_num_orbits is None:
            spacegroup, num_orbits = sample_spacegroups_and_orbits(
                    spacegroups_bincount=spacegroups_bincount,
                    sg_num_orbits_conditionals=sg_num_orbits_conditionals,
                    num_samples=batch_size,
                    device=self.device,
                    allowed_spacegroups=allowed_spacegroups,
                    forbidden_spacegroups=forbidden_spacegroups
                )
            

        elif presampled_spacegroup is not None and presampled_num_orbits is not None:
            spacegroup = torch.as_tensor(presampled_spacegroup, device=self.device, dtype=torch.long).reshape(-1)

            num_orbits = torch.as_tensor(presampled_num_orbits, device=self.device, dtype=torch.long).reshape(-1)

            if spacegroup.numel() != num_orbits.numel():
                raise ValueError(
                    "presampled_spacegroup and presampled_num_orbits "
                    "must have the same length."
                )

            if spacegroup.numel() != batch_size:
                raise ValueError(
                    f"batch_size={batch_size}, but received "
                    f"{spacegroup.numel()} pre-sampled items."
                )

            if torch.any(num_orbits <= 0):
                raise ValueError(
                    "All pre-sampled orbit counts must be positive."
                )

        else:
            raise ValueError(
                "presampled_spacegroup and presampled_num_orbits "
                "must either both be provided or both be None."
            )


        num_tokens = int(num_orbits.max().item())

        token_mask = torch.arange(num_tokens, device=self.device)[None, :] < num_orbits[:, None]
        

        # create new samples from interpolant
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=num_tokens,
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            spacegroup=spacegroup,
            device = self.device,
            cfg_scale=cfg_scale,
            num_timesteps=num_timesteps,
            token_mask=token_mask,
        )

        # get final samples and remove padding (to PyG format)
        x = samples["clean_traj"][-1][token_mask]

        batch = {
            "x": x,
            "batch": torch.repeat_interleave(
                torch.arange(len(num_orbits), device=self.device), num_orbits
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }

        # decode samples to crystal structures using frozen decoder
        out = self.autoencoder.decode(batch)

        
        return out, spacegroup, num_orbits, 
    
    #####################################################################################################

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        wyckoff_data = load_or_create_wyckoff_data()
        self.wyckoff_tokens = wyckoff_data["flat_pairs"]
        self.sg_to_wyckoff_range = wyckoff_data["index_dict"]
        self.wyckoff_pos_mask = wyckoff_data["wyckoff_pos_mask"]


        self.val_generation_evaluator = CrystalGenerationEvaluator(novelty = self.hparams.novelty)
        self.test_generation_evaluator = CrystalGenerationEvaluator(novelty = {"compute_novelty": False})

       
        self.spacegroups_bincount = torch.load(os.path.join(self.hparams.sampling.data_dir, "spacegroups_bincount.pt"), map_location="cpu").float()
        self.sg_num_orbits_conditionals = torch.load(os.path.join(self.hparams.sampling.data_dir, "sg_num_orbits_conditionals.pt"), map_location="cpu")

        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)






    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mp20/valid_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
