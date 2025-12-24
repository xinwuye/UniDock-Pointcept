"""
LBA regression dataset for Pointcept-style molecule folders.

Expected folder structure:
  <data_root>/{train,val,test}/<sample_id>/
    coord.npy        (N,3) float32
    atom_type.npy    (N,C) float32 one-hot
    identity.npy     (N,2) float32 one-hot (protein vs ligand)
    affinity.npy     (1,)  float32 label (neglog_aff / pK)
"""

from __future__ import annotations

import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS


import os

# ... (imports)

@DATASETS.register_module()
class LBARegressionDataset(DefaultDataset):
    """Dataset that loads coord/atom_type/identity plus affinity label."""

    VALID_ASSETS = ["coord", "atom_type", "identity", "affinity", "bwms_order"]

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        
        # Inject atom_types.json path relative to data_root
        data_dict["atom_types_json_path"] = os.path.join(self.data_root, "atom_types.json")

        missing = [k for k in ("coord", "atom_type", "identity", "affinity") if k not in data_dict]
        if missing:
            raise FileNotFoundError(
                f"Missing required assets {missing} in sample {data_dict.get('name')} under split {data_dict.get('split')}."
            )

        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["atom_type"] = data_dict["atom_type"].astype(np.float32)
        data_dict["identity"] = data_dict["identity"].astype(np.float32)

        # affinity is a scalar label stored as a (1,) array; ensure float32.
        aff = np.asarray(data_dict["affinity"], dtype=np.float32).reshape(-1)
        if aff.size != 1:
            raise ValueError(
                f"Expected affinity.npy to contain a single value, got shape={np.shape(data_dict['affinity'])} in {data_dict.get('name')}"
            )
        data_dict["affinity"] = aff

        # Make sure indexing-based transforms also slice these point-wise arrays.
        data_dict["index_valid_keys"] = ["coord", "atom_type", "identity", "segment", "instance"]
        return data_dict


