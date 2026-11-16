import torch
from pymatgen.core import  Element

def z_to_symbol(atom_list: torch.Tensor):
    symbols = []
    for z in atom_list.tolist():  # convert tensor to Python list
        symbols.append(Element.from_Z(int(z)).symbol)
    return symbols