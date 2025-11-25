import os
import torch
import torch.nn as nn
from collections import OrderedDict

from pointcept.models.builder import MODELS, build_model
from pointcept.models.docking.transformer import DockingTransformer
from pointcept.models.utils.structure import Point


def load_backbone(backbone, weight_path, label="backbone"):
    if weight_path is None or not os.path.isfile(weight_path):
        print(f"[Docking] No checkpoint for {label}: {weight_path}")
        return False
    ckpt = torch.load(weight_path, map_location='cpu')
    weight = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if k.startswith('module.'):
            k = k[7:]
        # allow mapping if keys start with 'backbone.'
        if k.startswith('backbone.'):
            k = k[9:]
        weight[k] = v
    missing, unexpected = backbone.load_state_dict(weight, strict=False)
    if len(missing) > 0:
        print(f"[Docking] Missing keys: {missing}")
    if len(unexpected) > 0:
        print(f"[Docking] Unexpected keys: {unexpected}")
    print(f"[Docking] Loaded {label} encoder weights from: {weight_path}")
    return True


@MODELS.register_module()
class DockingWrapper(nn.Module):
    def __init__(self,
                 backbone_fixed,
                 backbone_moved,
                 transformer=dict(d_model=512, nhead=8, num_layers=2, pool='mean'),
                 weight_fixed=None,
                 weight_moved=None,
                 freeze_backbone=True,
                 loss_rot_weight=1.0,
                 loss_trans_weight=1.0):
        super().__init__()
        self.backbone_fixed = build_model(backbone_fixed)
        self.backbone_moved = build_model(backbone_moved)
        self.transformer = DockingTransformer(**transformer)

        load_backbone(self.backbone_fixed, weight_fixed, label="fixed")
        load_backbone(self.backbone_moved, weight_moved, label="moved")

        if freeze_backbone:
            for p in self.backbone_fixed.parameters():
                p.requires_grad = False
            for p in self.backbone_moved.parameters():
                p.requires_grad = False

        self.loss_rot_weight = loss_rot_weight
        self.loss_trans_weight = loss_trans_weight
        self.l1 = nn.SmoothL1Loss()

    def encode(self, backbone, batch):
        # Ensure required keys for Point
        if 'feat' not in batch and 'atom_type' in batch:
            batch = dict(batch)
            batch['feat'] = batch['atom_type']
        point = Point(batch)
        point.serialization(order=backbone.order, shuffle_orders=backbone.shuffle_orders)
        point.sparsify()
        point = backbone.embedding(point)
        point = backbone.enc(point)
        return point.feat, point.offset

    def forward(self, input_dict):
        # Expect keys: coord_fixed, atom_type_fixed, coord_moved, atom_type_moved, rot_gt (3x3), trans_gt (3)
        fixed = dict(
            coord=input_dict['coord_fixed'],
            grid_coord=input_dict.get('grid_coord_fixed'),
            atom_type=input_dict['atom_type_fixed'],
            offset=input_dict.get('offset_fixed'),
        )
        moved = dict(
            coord=input_dict['coord_moved'],
            grid_coord=input_dict.get('grid_coord_moved'),
            atom_type=input_dict['atom_type_moved'],
            offset=input_dict.get('offset_moved'),
        )
        # move tensors to same device
        dev = input_dict['coord_fixed'].device
        for d in (fixed, moved):
            for k in d:
                if isinstance(d[k], torch.Tensor):
                    d[k] = d[k].to(dev)

        feat_f, off_f = self.encode(self.backbone_fixed, fixed)
        feat_m, off_m = self.encode(self.backbone_moved, moved)
        rot_raw, trans_pred = self.transformer(feat_f, off_f, feat_m, off_m)
        # Normalize quaternion prediction to unit norm and canonicalize sign (w>=0)
        eps = 1e-8
        q_pred = rot_raw / (rot_raw.norm(dim=-1, keepdim=True) + eps)
        sign = torch.where(q_pred[:, :1] < 0, -1.0, 1.0)
        q_pred = q_pred * sign

        # Ground-truth quaternion (w,x,y,z) and translation
        q_gt = input_dict['quat_gt'].to(dev)
        t = input_dict['trans_gt'].to(dev)

        # Quaternion geodesic proxy loss: 1 - (|<q_pred, q_gt>|)^2
        dot = torch.sum(q_pred * q_gt, dim=-1).abs().clamp(0.0, 1.0)
        loss_rot = (1.0 - dot * dot).mean()
        loss_trans = self.l1(trans_pred, t)
        loss = self.loss_rot_weight * loss_rot + self.loss_trans_weight * loss_trans
        return dict(
            loss=loss,
            loss_rot=loss_rot,
            loss_trans=loss_trans,
            quat_pred=q_pred,
            trans_pred=trans_pred,
        )
