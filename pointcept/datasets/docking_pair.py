"""
DockingPairDataset: load paired fixed/moved samples with matching names across two roots.
Applies recorded random rigid transforms and computes GT relative transform (R,t).
"""

import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as SR
from copy import deepcopy
from torch.utils.data import Dataset

from .builder import DATASETS
from .transform import Compose


def euler_to_matrix(rots):
    """Compose rotations around x,y,z (list of (axis, angle)) into 3x3 matrix.
    Order: apply in the given sequence.
    """
    R = np.eye(3, dtype=np.float32)
    for axis, angle in rots:
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'x':
            R_axis = np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
        elif axis == 'y':
            R_axis = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)
        else:
            R_axis = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
        R = R_axis @ R
    return R


@DATASETS.register_module()
class DockingPairDataset(Dataset):
    def __init__(self, split, fixed_root, moved_root, fixed_transform, moved_transform, test_mode=False):
        super().__init__()
        self.split = split
        self.fixed_root = os.path.join(fixed_root, split)
        self.moved_root = os.path.join(moved_root, split)
        self.fixed_transform = Compose(fixed_transform)
        self.moved_transform = Compose(moved_transform)
        self.test_mode = test_mode
        fixed_names = set([os.path.basename(p) for p in glob.glob(os.path.join(self.fixed_root, '*'))])
        moved_names = set([os.path.basename(p) for p in glob.glob(os.path.join(self.moved_root, '*'))])
        self.names = sorted(list(fixed_names & moved_names))

    def __len__(self):
        return len(self.names)

    def _load_npys(self, root, name):
        path = os.path.join(root, name)
        coord = np.load(os.path.join(path, 'coord.npy')).astype(np.float32)
        atom = np.load(os.path.join(path, 'atom_type.npy')).astype(np.float32)
        data = dict(coord=coord, atom_type=atom, name=name)
        return data

    def __getitem__(self, idx):
        name = self.names[idx]
        fixed = self._load_npys(self.fixed_root, name)
        moved = self._load_npys(self.moved_root, name)

        # apply transforms and record applied rotations/shifts
        fixed = self.fixed_transform(fixed)
        moved = self.moved_transform(moved)

        # Build GT relative transform R,t: move 'moved' to 'fixed'
        rots_f = fixed.get('applied_rot', [])
        rots_m = moved.get('applied_rot', [])
        Rf = euler_to_matrix(rots_f)
        Rm = euler_to_matrix(rots_m)
        R_gt = Rf @ Rm.T
        # Use recorded centers from CenterShiftRecord: t maps moved->fixed in column convention
        # Derivation (row/right-mul): Yf = Ym (Rm Rf^T) + (cm - cf) Rf^T
        # Thus in column/left-mul: R_gt = Rf Rm^T, t_gt = Rf (cm - cf)
        cf = fixed.get('applied_center', np.zeros(3, dtype=np.float32))
        cm = moved.get('applied_center', np.zeros(3, dtype=np.float32))
        t_gt = (Rf @ (cm - cf)).astype(np.float32)

        # Convert R_gt to quaternion (w,x,y,z), enforce w>=0 for canonical sign
        q_xyzw = SR.from_matrix(R_gt.astype(np.float64)).as_quat().astype(np.float32)  # (x,y,z,w)
        q_wxyz = np.concatenate([q_xyzw[3:4], q_xyzw[:3]], axis=0)
        if q_wxyz[0] < 0:
            q_wxyz = -q_wxyz

        import torch
        out = dict(
            name=name,
            coord_fixed=fixed['coord'],
            atom_type_fixed=fixed['atom_type'],
            offset_fixed=torch.tensor([fixed['coord'].shape[0]], dtype=torch.int32),
            coord_moved=moved['coord'],
            atom_type_moved=moved['atom_type'],
            offset_moved=torch.tensor([moved['coord'].shape[0]], dtype=torch.int32),
            rot_gt=R_gt.astype(np.float32),
            trans_gt=t_gt.astype(np.float32),
            quat_gt=q_wxyz.astype(np.float32),  # (w,x,y,z)
            R_fixed=Rf.astype(np.float32),
            R_moved=Rm.astype(np.float32),
            center_fixed=cf.astype(np.float32),
            center_moved=cm.astype(np.float32),
        )
        return out
