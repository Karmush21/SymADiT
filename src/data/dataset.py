"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, InMemoryDataset
from src.data.preprocessing import sym_preprocess


def get_num_workers():
    if hasattr(os, "sched_getaffinity"):
        num_workers = len(os.sched_getaffinity(0))
    else:
        num_workers = os.cpu_count() or 1

    # Respect SLURM CPU allocation if available
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus is not None:
        num_workers = min(num_workers, int(slurm_cpus))

    return max(1, num_workers)


class SymDataset(InMemoryDataset):
    """The dataset as a PyG InMemoryDataset.

    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:
    - InMemoryDataset.raw_file_names(): A list of files in the raw_dir which needs to be found in order to skip the download.
    - InMemoryDataset.processed_file_names(): A list of files in the processed_dir which needs to be found in order to skip the processing.
    - InMemoryDataset.download(): Downloads raw data into raw_dir.
    - InMemoryDataset.process(): Processes raw data and saves it into the processed_dir.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """


    def __init__(
        self,
        root: str,
        debug: bool,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:


        self.debug = debug
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0]) 


    
    @property
    def raw_file_names(self) -> List[str]:
        return ["raw.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self) -> None:
        raise NotImplementedError(
            f"Manually download the dataset and place it at {self.root}/raw."
        )

    
    def process(self) -> None:
        if os.path.exists(os.path.join(self.root, "all_ori.pt")):
            cached_data = torch.load(os.path.join(self.root, "all_ori.pt"), weights_only=False)
        
        else:
            if not self.debug:
                num_workers = get_num_workers()
                cached_data = sym_preprocess(input_file=os.path.join(self.root, "raw/raw.csv"), 
                num_workers=num_workers) 
            else:
                # NOTE material_ids let you choose certain files to debug
                mp_ids = [
                    #"mp-22862" # NaCL
                    "mp-1095443" # SG 194
                    #"mp-34284"
                    #...
                ]
                cached_data = sym_preprocess(input_file=os.path.join(self.root, "raw/raw.csv"), debug=True, material_ids= mp_ids, num_workers=1)
            
            torch.save(cached_data, os.path.join(self.root, "all_ori.pt"))
        
        data_list = []

        for data_dict in cached_data:
            graph_arrays = data_dict["graph_arrays"]

            N = len(graph_arrays["atom_types"])
            
            data = Data(
                id = data_dict["mp_id"],
                
                atom_types=torch.tensor(graph_arrays["atom_types"], dtype=torch.long),
                    
                num_atoms_conv=torch.tensor(graph_arrays["num_atoms_conv"], dtype=torch.long),
                num_orbits = torch.tensor(graph_arrays["num_orbits"], dtype=torch.long),
                
                
                wyckoff_ids=torch.tensor(graph_arrays["wyckoff_sites"], dtype=torch.long),        
                
                
                full_rep=torch.tensor(graph_arrays["full_rep"], dtype=torch.float32),      
                full_rep_mask=torch.tensor(graph_arrays["full_rep_mask"], dtype=torch.float32),   

                
                sg_number=torch.tensor([graph_arrays["space_group"]], dtype=torch.long),
                
                
                full_cell_rad=torch.tensor(graph_arrays["full_cell_rad"], dtype=torch.float32).unsqueeze(0), 
                full_cell_mask=torch.tensor(graph_arrays["full_cell_mask"], dtype=torch.float32).unsqueeze(0),  
                
                all_reps=torch.tensor(graph_arrays["all_reps"], dtype=torch.float32),
                all_reps_mask=torch.tensor(graph_arrays["all_reps_mask"], dtype=torch.bool),

                R=torch.tensor(graph_arrays["R"], dtype=torch.float32),
                t=torch.tensor(graph_arrays["t"], dtype=torch.float32),
                

                # For Pyxtal ground truth 
                x_cell_free_dof = graph_arrays["x_cell_free_dof"],
                x_pos_free_dof = graph_arrays["x_pos_free_dof"],

                volume=torch.tensor([graph_arrays["volume"]], dtype=torch.float32),


                num_dof_rep = torch.tensor(graph_arrays["num_dof_rep"], dtype=torch.float32),
                num_dof_cell = torch.tensor(graph_arrays["num_dof_cell"], dtype=torch.float32),

                A = torch.tensor(graph_arrays["A"], dtype=torch.float32),
                b = torch.tensor(graph_arrays["b"], dtype=torch.float32),


                token_idx=torch.arange(N),
                num_nodes=torch.tensor(N, dtype=torch.long),  
                
            )

            
            # NOTE ADiT did this, needed?
            # data.pos = torch.einsum(
            #     "bi,bij->bj",
            #     data.frac_coords,
            #     torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            # )


            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            

            data_list.append(data)

        self.save(data_list, os.path.join(self.root, "processed/data.pt"))
        

