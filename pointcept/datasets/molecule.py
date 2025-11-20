"""
Datasets for molecule-style point clouds (PDB/SDF preprocessed scenes).
"""

import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class MoleculeDataset(DefaultDataset):
    """
    Dataset that loads directories containing coord.npy + atom_type.npy exported
    by preprocess_pdb.py / preprocess_sdf.py. Each subfolder corresponds to one
    molecule/ligand.
    """

    VALID_ASSETS = ["coord", "atom_type"]

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        if "atom_type" not in data_dict:
            raise FileNotFoundError(
                f"'atom_type.npy' is required in {data_dict['name']}."
            )
        data_dict["atom_type"] = data_dict["atom_type"].astype(np.float32)
        # Force atom_type to be tracked by index operations.
        data_dict["index_valid_keys"] = ["coord", "atom_type", "segment", "instance"]
        return data_dict
