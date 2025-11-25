import os
import torch
import torch.nn as nn
from collections import OrderedDict

from pointcept.models.builder import MODELS, build_model
from pointcept.models.docking.transformer import DockingTransformer
from pointcept.models.utils.structure import Point


def load_backbone(backbone, weight_path):
    if weight_path is None or not os.path.isfile(weight_path):
        return
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

        load_backbone(self.backbone_fixed, weight_fixed)
        load_backbone(self.backbone_moved, weight_moved)

        if freeze_backbone:
            for p in self.backbone_fixed.parameters():
                p.requires_grad = False
            for p in self.backbone_moved.parameters():
                p.requires_grad = False

        self.loss_rot_weight = loss_rot_weight
        self.loss_trans_weight = loss_trans_weight
        self.l1 = nn.SmoothL1Loss()

    def encode(self, backbone, batch):
        point = Point(batch)
        point.serialization(order=backbone.order, shuffle_orders=backbone.shuffle_orders)
        point.sparsify()
        point = backbone.embedding(point)
        point = backbone.enc(point)
        return point.feat, point.offset

    def forward(self, input_dict):
        # Expect keys: coord_fixed, atom_type_fixed, coord_moved, atom_type_moved, rot_gt (3x3), trans_gt (3)
        fixed = dict(coord=input_dict['coord_fixed'], atom_type=input_dict['atom_type_fixed'])
        moved = dict(coord=input_dict['coord_moved'], atom_type=input_dict['atom_type_moved'])
        # move tensors to same device
        dev = input_dict['coord_fixed'].device
        for d in (fixed, moved):
            for k in d:
                if isinstance(d[k], torch.Tensor):
                    d[k] = d[k].to(dev)

        feat_f, off_f = self.encode(self.backbone_fixed, fixed)
        feat_m, off_m = self.encode(self.backbone_moved, moved)
        rot_pred, trans_pred = self.transformer(feat_f, off_f, feat_m, off_m)

        # Build gt rotation vector (extract Euler xyz from 3x3); here we use simple approximation via matrix log
        R = input_dict['rot_gt'].to(dev)
        t = input_dict['trans_gt'].to(dev)
        # convert rotation matrix to axis-angle then to xyz small-angle approx
        # trace-based angle
        B = R.size(0)
        rot_vec = []
        for b in range(B):
            Rb = R[b]
            angle = torch.acos(torch.clamp((Rb.trace() - 1) / 2, -1 + 1e-6, 1 - 1e-6))
            if angle.item() < 1e-6:
                rv = torch.zeros(3, device=dev)
            else:
                skew = (Rb - Rb.T) / (2 * torch.sin(angle))
                rv = angle * torch.tensor([skew[2,1], skew[0,2], skew[1,0]], device=dev)
            rot_vec.append(rv.unsqueeze(0))
        rot_vec = torch.cat(rot_vec, dim=0)

        loss_rot = self.l1(rot_pred, rot_vec)
        loss_trans = self.l1(trans_pred, t)
        loss = self.loss_rot_weight * loss_rot + self.loss_trans_weight * loss_trans
        return dict(loss=loss, loss_rot=loss_rot, loss_trans=loss_trans,
                    rot_pred=rot_pred, trans_pred=trans_pred)

