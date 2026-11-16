"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from functools import partial
from typing import Any, Dict
import numpy as np
import torch
import wandb
from pymatgen.analysis.structure_matcher import StructureMatcher
from src.crystal.crystal import Crystal
from src.utils import joblib_map
from tqdm import tqdm
import random
from src.crystal.sanity_check import sanity_check_crystal




class CrystalReconstructionEvaluator:
    """Evaluator for crystal reconstruction tasks. Can be used within a Lightning module, appending
    predictions and ground truths during training and computing metrics at the end of an epoch, or
    can be used as a standalone object to evaluate predictions on a dataset.

    Args:
        stol (float): StructureMatcher tolerance for matching sites.
        angle_tol (float): StructureMatcher tolerance for matching angles.
        ltol (float): StructureMatcher tolerance for matching lengths.
    """

    def __init__(self, stol=0.5, angle_tol=10, ltol=0.3, device="cpu"):
        
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []  # list of Dict[str, np.array] predictions
        self.gt_arrays_list = []  # list of Dict[str, np.array] ground truths
        self.pred_crys_list = []  # list of Crystal predictions
        self.gt_crys_list = []  # list of Crystal ground truths
        self.device = device

        self.pyxtal_pred_list = []
        self.pyxtal_gt_list = []

        self.pred_table = None
        self.pyxtal_pred_table = None

        self.comp_valid = []
        self.struct_valid = []
        self.validity = []

    def append_pred_array(self, pred: Dict[str, np.array]):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: Dict[str, np.array]):
        """Append a ground truth to the evaluator."""
        self.gt_arrays_list.append(gt)
    
    def append_pyxtal_pred_array(self, pred: Dict):
        self.pyxtal_pred_list.append(pred)
        
    
    def append_pyxtal_gt_array(self, pred: Dict):
        self.pyxtal_gt_list.append(pred)

    
    def clear(self):
        """Clear the stored predictions and ground truths, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_crys_list = []
        self.gt_crys_list = []

        self.pyxtal_pred_list = []
        self.pyxtal_gt_list = []

    def _arrays_to_crystals(self, debug):
        """Convert stored predictions and ground truths to Crystal objects for evaluation.
        
        partial: freeze some arguments of a function ahead of time and create a new function
        that only needs the remaining arguments 

        joblib_map is something with doing things in parallel, we have created this
        
        
        """
        if not debug:
            self.pred_crys_list = joblib_map(
                partial(array_dict_to_crystal),
                self.pred_arrays_list,
                n_jobs=-1, 
                inner_max_num_threads=1,
                desc=f"    Pred to Crystal",
                total=len(self.pred_arrays_list),
            )
            self.gt_crys_list = joblib_map(
                partial(array_dict_to_crystal),
                self.gt_arrays_list,
                n_jobs=-1,
                inner_max_num_threads=1,
                desc=f"    G.T. to Crystal",
                total=len(self.gt_arrays_list),
            )

        else:
            for x in tqdm(self.pred_arrays_list, desc="    Pred to Crystal"):
                crys = array_dict_to_crystal(x)
                self.pred_crys_list.append(crys)

            for x in tqdm(self.gt_arrays_list, desc="    G.T. to Crystal"):
                crys = array_dict_to_crystal(x)
                self.gt_crys_list.append(crys)



    def _get_metrics(self, pred, gt):

        if getattr(pred, "invalid_reason", None) == "could_not_build_pyxtal":
            return float("inf")
        try:
            rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
            rms_dist = float("inf") if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return float("inf")
    

    def get_metrics(self, debug, save_dir = None, save_struct_fails = None) -> Dict[str, Any]:
        """Compute the match rate and avg. RMS distance between predictions and ground truths.

        Note: self.rms_dists can be used to access RMSD per sample but is not returned.

        Returns:
            Dict: Dictionary of metrics, including match rate and avg. RMSD.
        """
        self.save_dir = save_dir
        self.save_struct_fails = save_struct_fails
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

         
        self._arrays_to_crystals(debug)


        pred_comp_valid = []
        pred_struct_valid = []
        pred_valid = []

        for pred in self.pred_crys_list:
            pred_comp_valid.append(pred.comp_valid)
            pred_struct_valid.append(pred.struct_valid)
            pred_valid.append(pred.valid)
        

        self.comp_valid  = torch.tensor(pred_comp_valid, device=self.device)
        self.struct_valid = torch.tensor(pred_struct_valid, device=self.device)
        self.validity = torch.tensor(pred_valid, device=self.device)


        self.rms_dists = []
        for i in tqdm(range(len(self.pred_crys_list)), desc="   Reconstruction eval"):
            self.rms_dists.append(
                self._get_metrics(self.pred_crys_list[i], self.gt_crys_list[i]) #, validity[i])
            )
        
        self.rms_dists = torch.tensor(self.rms_dists, device=self.device)
        self.match_rate = (~torch.isinf(self.rms_dists)).long()


        if self.save_dir:
            for pred, gt in zip(self.pred_crys_list, self.gt_crys_list):
                sample_id = pred.sample_idx
                sg = pred.spacegroup
                base_name = f"crystal_{sample_id}"
                
                # Sanity check functions are prio number 1
                if pred.invalid_reason != "ok":
                    pred_path = os.path.join(self.save_dir, f"{base_name}_pred_{pred.invalid_reason}_sg{sg}.cif")
                    gt_path   = os.path.join(self.save_dir, f"{base_name}_gt_sg{sg}.cif")
                    
                    pred.structure.to(pred_path)
                    gt.structure.to(gt_path)
                
                # To investigate those that we don't get valid on
                elif not pred.valid:

                    if not pred.comp_valid and not pred.struct_valid:
                        reason = "val_fail"
                    elif not pred.struct_valid:
                        reason = "struct_fail"
                    elif not pred.comp_valid:
                        reason = "comp_fail"
                    else:
                        reason = "unknown_fail"

                    pred_path = os.path.join(
                        self.save_dir,
                        f"{base_name}_pred_{reason}_sg{sg}.cif"
                    )
                    gt_path = os.path.join(self.save_dir, f"{base_name}_gt_sg{sg}.cif")

                    pred.structure.to(pred_path)
                    gt.structure.to(gt_path)

        
                
        if self.match_rate.sum() == 0:
      
            self.rms_dists = torch.tensor([10.0] * len(self.match_rate), device=self.device)
            return {
                "match_rate": torch.tensor([0.0] * len(self.match_rate), device=self.device),
                "rms_dist": self.rms_dists,
                "comp_valid": self.comp_valid.float(),
                "struct_valid": self.struct_valid.float(),
                "validity": self.validity.float(),
            }
        
        else:
            return {
                "match_rate": self.match_rate,
                "rms_dist": self.rms_dists[~torch.isinf(self.rms_dists)], # This will return ANOTHER TENSOR, not change self.rms_dists tensor! important for table!
                "comp_valid": self.comp_valid.float(),
                "struct_valid": self.struct_valid.float(),
                "validity": self.validity.float(),
            }

    def get_wandb_table(self, current_epoch: int = 0) -> wandb.Table:
        """Create a wandb.Table object with the results of the evaluation."""
        if self.pred_table is None:
            self.pred_table = wandb.Table(
                columns=[
                    "Epoch",
                    "id",
                    "Sample idx",
                    "Num atoms",
                    "RMSD",
                    "Match?",
                    "Valid?",
                    "Comp valid?",
                    "Struct valid?",
                    "True atom types",
                    "Pred atom types",
                    "True lengths",
                    "Pred lengths",
                    "True angles",
                    "Pred angles",
                ]
            )


        pred_table = self.pred_table
        
        
        for idx in range(len(self.pred_crys_list)):
            sample_idx = self.gt_crys_list[idx].sample_idx
            assert sample_idx == self.pred_crys_list[idx].sample_idx

            num_atoms = len(self.gt_crys_list[idx].atom_types)

            id = self.gt_crys_list[idx].id
            
            
            rmsd = self.rms_dists[idx]
            match = bool(self.match_rate[idx].item())

            true_atom_types = " ".join([str(int(t)) for t in self.gt_crys_list[idx].atom_types])

            pred_atom_types = " ".join([str(int(t)) for t in self.pred_crys_list[idx].atom_types])

            true_lengths = " ".join([f"{l:.2f}" for l in self.gt_crys_list[idx].lengths])

            true_angles = " ".join([f"{a:.2f}" for a in self.gt_crys_list[idx].angles])

            pred_lengths = " ".join([f"{l:.2f}" for l in self.pred_crys_list[idx].lengths])

            pred_angles = " ".join([f"{a:.2f}" for a in self.pred_crys_list[idx].angles])



            # # Update table
            pred_table.add_data(
                current_epoch,
                id,
                sample_idx,
                num_atoms,
                rmsd,
                match,
                self.pred_crys_list[idx].valid,
                self.pred_crys_list[idx].comp_valid,
                self.pred_crys_list[idx].struct_valid,
                true_atom_types,
                pred_atom_types,
                true_lengths,
                pred_lengths,
                true_angles,
                pred_angles,
            )

        return pred_table
    

    def get_pyxtal_wandb_table(
        self,
        current_epoch: int,
        max_samples: int = 1000,
    ) -> wandb.Table:
        """
        Build a W&B table comparing PyXtal-style GT vs predictions using:
            - self.pyxtal_gt_list
            - self.pyxtal_pred_list

        Each entry is expected to be a dict with keys:
            'sample_idx', 'sg', 'wyckoff', 'lengths', 'angles', 'atom_types', 'num_orbits' (optional)
        """

        assert len(self.pyxtal_pred_list) == len(self.pyxtal_gt_list), \
            "Prediction and GT pyxtal lists must match in length"

        N = len(self.pyxtal_pred_list)
        assert N > 0, "No PyXtal predictions/GT to log."

        # ---- SAMPLE INDICES ----
        if N <= max_samples:
            chosen_idx = list(range(N))
        else:
            chosen_idx = random.sample(range(N), max_samples)

        
        if self.pyxtal_pred_table is None:
            self.pyxtal_pred_table = wandb.Table(
                columns=[
                    "Epoch",
                    "id",
                    "Sample idx",
                    "(True) SG",
                    
                    "True Wyckoff",
                    "Pred Wyckoff",
                    
                    "True Atom Types",
                    "Pred Atom Types",
                    
                    "True Free Pos",
                    "Pred Free Pos",
                    
                    "True Free Cell",
                    "Pred Free Cell",
                ]
            )
        
        pred_table_pyxtal = self.pyxtal_pred_table

        # ---- ADD SAMPLED ROWS ----
        for idx in chosen_idx:
            g = self.pyxtal_gt_list[idx]
            p = self.pyxtal_pred_list[idx]

            sample_idx = g["sample_idx"]
            id = p["id"]
            

            true_atom_types = " ".join(str(t) for t in g["atom_types"])
            pred_atom_types = " ".join(str(t) for t in p["atom_types"])

            true_wyckoff = " ".join(str(w) for w in g["wyckoff"])
            pred_wyckoff = " ".join(str(w) for w in p["wyckoff"])

            true_free_pos = " ".join(f"{l:.2f}" for l in g["pos_free"])
            true_free_cell  = " ".join(f"{a:.2f}" for a in g["cell_free"])

            pred_free_pos = " ".join(f"{l:.2f}" for l in p["pos_free"])
            pred_free_cell  = " ".join(f"{a:.2f}" for a in p["cell_free"])

            true_sg = g["sg"]


            pred_table_pyxtal.add_data(
                current_epoch,
                id,
                sample_idx,
                true_sg,
                true_wyckoff,
                pred_wyckoff,
                true_atom_types,
                pred_atom_types,
                true_free_pos,
                pred_free_pos,
                true_free_cell,
                pred_free_cell,
            )

        return pred_table_pyxtal


    


def array_dict_to_crystal(x: dict) -> Crystal:


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

    # Preserve the generated SG when PyXtal built successfully.
    # Otherwise retain the requested/conditioned SG.
    # if data.get("built_spacegroup") is not None:
    #     data["spacegroup"] = data["built_spacegroup"]
    # else:
    #     data["spacegroup"] = data["requested_spacegroup"]

    return Crystal(data)

