"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import csv
import warnings
from functools import partial
from typing import Any, Dict
import numpy as np
import torch
import wandb
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from src.crystal.crystal import Crystal
from tqdm import tqdm
import random
import pandas as pd

from pathlib import Path

from src.crystal.sanity_check import sanity_check_crystal
from src.utils import joblib_map, pylogger
from src.evaluation_utils.build_training_materials import build_training_data_from_csv
from src.evaluation_utils.novel_and_unique import evaluate_novelty_and_save_parallel

warnings.filterwarnings("ignore", category=UserWarning)

log = pylogger.RankedLogger(__name__)




def cif_to_struct_silent(cif_str: str) -> Structure:
    with warnings.catch_warnings():
        # Suppress CIF rounding warning
        warnings.filterwarnings(
            "ignore",
            message=r".*fractional coordinates rounded to ideal values.*",
            category=UserWarning,
        )
        # Suppress electronegativity warning for any element (He, Ar, etc.)
        warnings.filterwarnings(
            "ignore",
            message=r"No Pauling electronegativity for .*",
            category=UserWarning,
        )
        return Structure.from_str(cif_str, fmt="cif")



def load_training_cifs_for_novelty(data_dir):
    dataset_name = os.path.basename(data_dir)

    df = pd.read_csv(os.path.join(data_dir, "raw", "raw.csv"))

    if dataset_name == "mp20":
        train_idx = torch.arange(0, 1000)

    elif dataset_name == "mpts52":
        train_idx = torch.arange(0, 27380)

    elif dataset_name == "mp20_mpts52_merged":
        split_path = os.path.join(data_dir, "train_val_split.pt")
        split_dict = torch.load(split_path, weights_only=False)
        train_idx = split_dict["train_idx"]

    else:
        raise ValueError(f"Unknown dataset for novelty: {dataset_name}")

    training_data_cif = df.iloc[train_idx.tolist()]["cif"].tolist()
    return training_data_cif

class CrystalGenerationEvaluator:
    """Evaluator for crystal generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(
        self,
        novelty,
        stol=0.5,
        angle_tol=10,
        ltol=0.3,
        device="cpu",
    ):

        self.dataset_struct_list = None  # loader first time it is required
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []
        self.pred_crys_list = []
        self.device = device
        
        self.pyxtal_pred_table = None
        self.pyxtal_list = []

        self.novelty = novelty["compute_novelty"] 

        if self.novelty:
            training_data_cif = load_training_cifs_for_novelty(novelty["data_dir"])
            
            self.training_structures = build_training_data_from_csv(training_data_cif, cell_representation="conventional") # NOTE Because of pyxtal
        

    def append_pred_array(self, pred: Dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)
    
    def append_pyxtal_list(self, pred: Dict):
        """Append pyxtal crystal"""
        self.pyxtal_list.append(pred)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_crys_list = []

        self.pyxtal_list = []

    def _arrays_to_crystals(self, debug): 
        """Convert stored predictions and ground truths to Crystal objects for evaluation."""
        
       
        if not debug:
            self.pred_crys_list = joblib_map(partial(array_dict_to_crystal),
                self.pred_arrays_list,
                n_jobs=-1,
                inner_max_num_threads=1,
                desc=f"    Pred to Crystal (JOBLIB)",
                total=len(self.pred_arrays_list),
            )
        
        else:
            for x in tqdm(self.pred_arrays_list, desc="    Pred to Crystal (Debug)"):
                crys = array_dict_to_crystal(x)
                self.pred_crys_list.append(crys)

 
    def _get_novelty(self, struct):
        # matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        for other_struct in self.dataset_struct_list:
            if self.matcher.fit(struct, other_struct, skip_structure_reduction=True):
                return False
        return True

    def get_metrics(self, save: bool = False, save_dir: str = "", debug=False):
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."

        self.save = save

        if self.save:
            os.makedirs(save_dir, exist_ok=True)
            self.save_dir = Path(save_dir)
 
        # Convert predictions and ground truths to Crystal objects
        self._arrays_to_crystals(debug)
        

        # Compute validity metrics
        metrics_dict = {
            "valid_rate": torch.tensor([c.valid for c in self.pred_crys_list], device=self.device),
            "comp_valid_rate": torch.tensor(
                [c.comp_valid for c in self.pred_crys_list], device=self.device
            ),
            "struct_valid_rate": torch.tensor(
                [c.struct_valid for c in self.pred_crys_list], device=self.device
            ),
        }

        
        valid_structs = [c.structure for c in self.pred_crys_list if c.invalid_reason == "ok"]
        unique_struct_groups = self.matcher.group_structures(valid_structs)
        
        if len(valid_structs) > 0:
            metrics_dict["unique_rate"] = torch.tensor(
                len(unique_struct_groups) / len(valid_structs), device=self.device
            )
        else:
            metrics_dict["unique_rate"] = torch.tensor(0.0, device=self.device)


        
        if self.novelty:
            unique_struct_groups_single = [group[0] for group in unique_struct_groups]
            _, novel_rate, _, _ = evaluate_novelty_and_save_parallel(unique_struct_groups_single,
                                                            self.training_structures,
                                                            self.matcher,
                                                            save_dir = None)
            metrics_dict["novel_rate"] = torch.tensor(novel_rate, device=self.device)
        
        else:
            metrics_dict["novel_rate"] = torch.tensor(0.0, device=self.device)


        if self.save:
            for pred in tqdm(self.pred_crys_list, desc="Saving Predictions"):
                sample_id = pred.sample_idx

                suffix = ""
                if pred.invalid_reason != "ok":
                    suffix = f"_absurd_{pred.invalid_reason}"

                sg = pred.spacegroup  # int
                
                filename = f"crystal_{sample_id}{suffix}_sg{sg}.cif"
                cif_path = self.save_dir / filename

                
                pred.pyxtal_crys.to_file(str(cif_path)) 
                



        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0):
        # Log crystal structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Valid?",
                "Comp valid?",
                "Struct valid?",
                "Pred atom types",
                "Pred lengths",
                "Pred angles",
            ]
        )

        for idx in range(len(self.pred_crys_list)):
            sample_idx = self.pred_crys_list[idx].sample_idx

            num_atoms = len(self.pred_crys_list[idx].atom_types)

            pred_atom_types = " ".join([str(int(t)) for t in self.pred_crys_list[idx].atom_types])

            pred_lengths = " ".join([f"{l:.2f}" for l in self.pred_crys_list[idx].lengths])

            pred_angles = " ".join([f"{a:.2f}" for a in self.pred_crys_list[idx].angles])

            
            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                self.pred_crys_list[idx].valid,
                self.pred_crys_list[idx].comp_valid,
                self.pred_crys_list[idx].struct_valid,
                pred_atom_types,
                pred_lengths,
                pred_angles,
            )

        return pred_table


    def get_pyxtal_wandb_table(
        self,
        current_epoch: int,
        max_samples: int = 1000,
    ) -> wandb.Table:
        """
        Build a W&B table for PyXtal-style predictions only.

        Each entry in self.pyxtal_list is expected to be a dict with keys:
            'sample_idx', 'sg', 'wyckoff', 'lengths', 'angles',
            'atom_types', 'pos_free', 'cell_free', 'num_orbits' (optional)
        """

        assert len(self.pyxtal_list) > 0, "No PyXtal predictions to log."

        N = len(self.pyxtal_list)

        # ---- SAMPLE INDICES ----
        if N <= max_samples:
            chosen_idx = list(range(N))
        else:
            chosen_idx = random.sample(range(N), max_samples)

        # ---- INIT TABLE ----
        if self.pyxtal_pred_table is None:
            self.pyxtal_pred_table = wandb.Table(
                columns=[
                    "Epoch",
                    "Sample idx",
                    "Num Orbits",
                    "Space Group",
                    "Pred Wyckoff",
                    "Pred Atom Types",
                    "Pred Free Pos",
                    "Pred Free Cell",
                ]
            )

        pred_table_pyxtal = self.pyxtal_pred_table

        # ---- ADD SAMPLED ROWS ----
        for idx in chosen_idx:
            p = self.pyxtal_list[idx]

            sample_idx = p["sample_idx"]
            num_orbits = len(p["atom_types"])

            pred_atom_types = " ".join(str(t) for t in p["atom_types"])
            pred_wyckoff = " ".join(str(w) for w in p["wyckoff"])
            pred_free_pos = " ".join(f"{l:.2f}" for l in p["pos_free"])
            pred_free_cell = " ".join(f"{a:.2f}" for a in p["cell_free"])
            
            pred_sg = p["sg"]
            
            pred_table_pyxtal.add_data(
                current_epoch,
                sample_idx,
                num_orbits,
                pred_sg,
                pred_wyckoff,
                pred_atom_types,
                pred_free_pos,
                pred_free_cell,
            )

        return pred_table_pyxtal



def array_dict_to_crystal(x: dict) -> Crystal:
    
    # Avoid modifying the dictionary stored in pred_arrays_list.
    data = dict(x)

    c_pred = data.get("pyxtal_crys")

    ok, reason, pyg_pred, c_pred = sanity_check_crystal(c_pred)


    data["pyxtal_crys"] = c_pred
    data["pymatgen_crys"] = pyg_pred
    data["reason"] = reason

    data["atom_types"] = np.asarray(pyg_pred.atomic_numbers, dtype=int)
    data["pos"] = np.asarray(pyg_pred.cart_coords, dtype=float)
    data["frac_coords"] = np.asarray(pyg_pred.frac_coords, dtype=float)
    data["lengths"] = np.asarray(pyg_pred.lattice.abc, dtype=float)
    data["angles"] = np.asarray(pyg_pred.lattice.angles, dtype=float)

    return Crystal(data)


