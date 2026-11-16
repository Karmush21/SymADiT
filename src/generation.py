"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import rootutils
#rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Any, Dict, Optional, Tuple
import os
import time
import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import wandb
from omegaconf import DictConfig
from tqdm import tqdm 
from contextlib import redirect_stdout, redirect_stderr
from pyxtal import pyxtal




rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

# NOTE If you're not using git
#rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)




# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


from src.utils.generation_utils import (
    sample_spacegroups_and_orbits, 
    sample_wyckoffs_and_atoms,
    build_exact_orbit_buckets
)

from src.models.ldm_module import LatentDiffusionLitModule
from src.crystal.crystal_generation import CrystalGenerationEvaluator



from src.symmetry.wyckoff_data import load_or_create_wyckoff_data
from src.symmetry.wyckoff_masks import extract_masks
from src.symmetry.cell import cell_mask_from_sg
from src.utils import RankedLogger, instantiate_loggers


log = RankedLogger(__name__, rank_zero_only=True)

def resolve_device(cfg) -> torch.device:
    dev = str(cfg.generate.device).lower()

    # Explicit CPU request
    if dev == "cpu":
        return torch.device("cpu")

    # Explicit CUDA request (either "cuda" or "cuda:N")
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("generate.device requests CUDA but torch.cuda.is_available() is False")
        if dev == "cuda":
            # torchrun-safe default: cuda:LOCAL_RANK
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device(f"cuda:{local_rank}")
        return torch.device(dev)

    # Optional: allow "auto"
    if dev == "auto":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    raise ValueError(f"Unknown generate.device='{cfg.generate.device}' (use cpu|cuda|cuda:N|auto)")




def generate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        Generation
    """

    assert cfg.ldm_ckpt, "ldm_ckpt must be set"

    log.info(f"Running Generation")
    
    
    # ---------------------------------------------------------
    # Static Wyckoff metadata
    # ---------------------------------------------------------

    wyckoff_data = load_or_create_wyckoff_data()

    flat_pairs = wyckoff_data["flat_pairs"]
    index_dict = wyckoff_data["index_dict"]
    wyckoff_pos_mask = wyckoff_data["wyckoff_pos_mask"]



    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    
    
    loggers = instantiate_loggers(cfg.get("logger"))

    # Pick wandb logger if present
    wandb_logger = None
    for lg in (loggers or []):
        if isinstance(lg, WandbLogger):
            wandb_logger = lg
            break
    
    
    ldm_ckpt_path = cfg.ldm_ckpt
    vae_ckpt_path = cfg.vae_ckpt
    
    # NOTE Novelty will be calculcated in evaluation script
    crystal_generator = CrystalGenerationEvaluator(novelty = {"compute_novelty": False})

    
    if vae_ckpt_path:
        log.info(f"Loading Diffusion model ckpt: {ldm_ckpt_path}")
        log.info(f"Loading VAE model ckpt: {vae_ckpt_path}")
        model = LatentDiffusionLitModule.load_from_checkpoint(
            ldm_ckpt_path, 
            autoencoder_ckpt = vae_ckpt_path,
            strict=False,
            map_location="cpu"
            )
    else:
        log.info(f"Loading Diffusion model ckpt: {ldm_ckpt_path}")
        model = LatentDiffusionLitModule.load_from_checkpoint(ldm_ckpt_path, strict = False, map_location="cpu")
    


    device = resolve_device(cfg)

    model.eval()
    if device.type == "cuda":
        model = model.to(device)          # only move if GPU
    else:
        # ensure CPU (since you loaded with map_location="cpu" this is already true)
        model = model.to("cpu")

    # Robust check
    param_dev = next(model.parameters()).device
    log.info(f"Resolved device={device} | model params on {param_dev}")
    assert param_dev.type == device.type, f"Model on {param_dev}, expected {device}"


    
    
    log.info(f"Loading priors: {cfg.priors.spacegroups_bincount}")
    sg_bc          = torch.load(cfg.priors.spacegroups_bincount, map_location="cpu")
    sg_bc = sg_bc.to(device)
    sg_num_orbits_conditionals = torch.load(cfg.priors.sg_num_orbits_conditionals, map_location="cpu")
    



    num_samples = int(cfg.generate.num_samples)
    batch_size = int(cfg.generate.batch_size)
    allowed_sgs = cfg.conditioning.allowed_spacegroups  # or None
    forbidden_sgs = cfg.conditioning.forbidden_spacegroups  # or None
    cfg_scale = cfg.generate.cfg_scale
    num_timesteps = cfg.generate.num_timesteps


    saving_dir = f'{cfg.generate.saving_dir}/cif/'



    with torch.inference_mode():
        all_spacegroups, all_num_orbits = sample_spacegroups_and_orbits(
                spacegroups_bincount=sg_bc,
                sg_num_orbits_conditionals=sg_num_orbits_conditionals,
                num_samples=num_samples,
                device = device,
                allowed_spacegroups=allowed_sgs,
                forbidden_spacegroups=forbidden_sgs,
            )
        

    # ------------------------------------------------------------------
    # Build exact orbit-count buckets
    # ------------------------------------------------------------------
    orbit_buckets = build_exact_orbit_buckets(all_num_orbits)

    sample_idx = 0

    progress = tqdm(total=num_samples, desc="Bucketed model sampling",)

    # Process one exact orbit-count bucket at a time.
    for orbit_count in sorted(orbit_buckets.keys()):

        bucket_indices = orbit_buckets[orbit_count]
        bucket_size = int(bucket_indices.numel())

        # A bucket may contain more samples than the configured batch size,
        # so divide it into chunks.
        for bucket_start in range(0, bucket_size, batch_size):

            bucket_end = min(bucket_start + batch_size, bucket_size)

            selected_indices = bucket_indices[bucket_start:bucket_end]

            current_batch_size = int(selected_indices.numel())

            batch_spacegroups = all_spacegroups[selected_indices]

            batch_num_orbits = all_num_orbits[selected_indices]

            # Every sample in this exact bucket must have the same
            # number of orbit tokens.
            assert torch.all(batch_num_orbits == orbit_count) 
            (
                f"Bucket {orbit_count} contains inconsistent "
                f"orbit counts: {batch_num_orbits.tolist()}"
            )
        

            with torch.inference_mode():
                
                out, spacegroup, num_orbits = model.sample_and_decode(
                    spacegroups_bincount=sg_bc,
                    sg_num_orbits_conditionals=sg_num_orbits_conditionals,
                    batch_size=current_batch_size,
                    cfg_scale=cfg_scale,
                    num_timesteps=num_timesteps,
                    presampled_spacegroup=batch_spacegroups,
                    presampled_num_orbits=batch_num_orbits
                )
            

            # ----------------------------------------------------------
            # Process the decoded crystals
            # ----------------------------------------------------------
            start_idx = 0
            for idx_in_batch, num_sites in enumerate(num_orbits.tolist()):
                num_sites = int(num_sites)

                current_spacegroup = int(spacegroup[idx_in_batch].item())

                wy_logits = out["wyckoff"].narrow(0, start_idx, num_sites,)

                atom_logits = out["atom_types"].narrow(0, start_idx, num_sites,)

                sampled_wyckoffs_and_atoms = sample_wyckoffs_and_atoms(
                    wy_logits=wy_logits,
                    atom_logits=atom_logits,
                    INDEX_DICT=index_dict,
                    FLAT_PAIRS=flat_pairs,
                    WYCKOFF_POS_MASK=wyckoff_pos_mask,
                    spacegroup=current_spacegroup,
                    wyckoff_temperature=cfg.temperature.wyckoff,
                    atom_temperature=cfg.temperature.atom_type,
                    device=device,
                )
                

                wyckoff_types = sampled_wyckoffs_and_atoms["wyckoff_types"]
                

                wyckoff_types_clean = sampled_wyckoffs_and_atoms["wyckoff_types_clean"]
                

                atom_types_list = sampled_wyckoffs_and_atoms["atom_types_list"]
                

                # ------------------------------------------------------
                # Position degrees of freedom
                # ------------------------------------------------------
                pos_pred = out["pos_pred"].narrow(0, start_idx, num_sites)

                pos_mask = extract_masks(wyckoff_types, wyckoff_pos_mask, device)

                pos_free = pos_pred.reshape(-1)[pos_mask.reshape(-1).bool()]

                pos_free_pyxtal = torch.remainder(pos_free,1.0)

                # ------------------------------------------------------
                # Cell degrees of freedom
                # ------------------------------------------------------
                cell_pred = out["cell_pred"][idx_in_batch]

                cell_deg = cell_pred.clone()
                cell_deg[3:] = torch.rad2deg(cell_deg[3:])

                cell_mask_np = cell_mask_from_sg(current_spacegroup)

                cell_mask = torch.as_tensor(cell_mask_np, device=cell_deg.device, dtype=torch.bool)

                cell_free_pyxtal = cell_deg[cell_mask]

                pyxtal_parameters = torch.cat([cell_free_pyxtal, pos_free_pyxtal], dim=0).detach().cpu().numpy()

                # ------------------------------------------------------
                # Store compact prediction
                # ------------------------------------------------------
                crystal_generator.append_pyxtal_list(
                    {
                        "sample_idx": sample_idx,
                        "sg": current_spacegroup,
                        "wyckoff": wyckoff_types_clean,
                        "pos_free": pos_free_pyxtal,
                        "cell_free": cell_free_pyxtal,
                        "atom_types": atom_types_list,
                    }
                )

                # ------------------------------------------------------
                # Construct PyXtal object
                # ------------------------------------------------------
                crystal = pyxtal()
                try:
                    with (open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull)):
                        crystal.from_spg_wps_rep(
                            current_spacegroup,
                            wyckoff_types_clean,
                            pyxtal_parameters,
                            atom_types_list,
                        )

                except Exception:
                    crystal = None

                crystal_generator.append_pred_array(
                    {
                        "pyxtal_crys": crystal,
                        "sample_idx": sample_idx,
                        "spacegroup": current_spacegroup,
                    }
                )

                start_idx += num_sites
                sample_idx += 1

            progress.update(current_batch_size)

    
    progress.close()

    assert sample_idx == num_samples, (
        f"Expected {num_samples} generated samples, "
        f"but processed {sample_idx}."
    )

    total_count = sample_idx
        
    
    gen_metrics_dict = crystal_generator.get_metrics(
            save=True,
            save_dir=saving_dir,
        )

    # --- reduce boolean tensors to rates ---
    reduced_metrics = {}

    for k, v in gen_metrics_dict.items():
        if torch.is_tensor(v):
            if v.ndim == 0:
                # scalar tensor
                reduced_metrics[k] = float(v.item())
            else:
                # per-sample boolean tensor → mean rate
                reduced_metrics[k] = float(v.float().mean().item())
        else:
            reduced_metrics[k] = float(v)

    print("\n[Generation metrics — reduced]")
    for k, v in reduced_metrics.items():
        print(f"  {k:20s}: {v:.4f}")


            
    pred_table = crystal_generator.get_wandb_table(current_epoch=0)

    pyxtal_pred_table = crystal_generator.get_pyxtal_wandb_table(current_epoch = 0)
    
    

    if wandb_logger is not None:
        # Log ONLY the tables
        wandb_logger.experiment.log(
            {
                "pred_table": pred_table,
                "pyxtal_pred_table": pyxtal_pred_table,
            }
        )

        
        wandb_logger.experiment.summary["num_generated"] = int(total_count)

        for k, v in reduced_metrics.items():
            wandb_logger.experiment.summary[k] = v
    
        wandb.finish()
    else:
        print("No WandbLogger configured; skipping W&B logging.")


    log.info(f"Done, Generated {num_samples} Samples")

    

@hydra.main(version_base="1.3", config_path="../configs", config_name="generation.yaml")
def main(cfg: DictConfig) -> Optional[float]:    
    
    generate(cfg)


if __name__ == "__main__":
    main()
