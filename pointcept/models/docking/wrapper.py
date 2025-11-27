import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from pointcept.models.builder import MODELS, build_model
from pointcept.models.docking.transformer import DockingTransformer, DockingTransformerFlow
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


def q_inverse(q):
    # q: (..., 4) [w, x, y, z]
    # inverse of unit quaternion is conjugate: [w, -x, -y, -z]
    q_inv = q.clone()
    q_inv[..., 1:] = -q_inv[..., 1:]
    return q_inv

def q_multiply(q1, q2):
    # q1, q2: (..., 4) [w, x, y, z]
    # Hamiltonian product
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)

def q_to_axis_angle(q):
    # q: (..., 4) [w, x, y, z]
    # Returns xi = theta * u (..., 3)
    # Ensure q is normalized and w >= 0 (double cover) handled by caller or here?
    # Caller usually ensures q_noise and q_gt are in same hemisphere.
    
    # Clip w to [-1, 1] to avoid NaN
    w = q[..., 0].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    norm_xyz = xyz.norm(dim=-1, keepdim=True)
    
    # theta = 2 * atan2(|xyz|, w)
    # if |xyz| ~ 0, theta ~ 0, axis is arbitrary.
    # u = xyz / |xyz|
    
    # Robust implementation using arctan2
    theta = 2.0 * torch.atan2(norm_xyz, w)
    
    # xi = theta * (xyz / norm_xyz) = (theta / norm_xyz) * xyz
    # Limit theta/norm_xyz as norm_xyz -> 0:
    # 2 * atan2(n, w) / n -> 2 * (n/w) / n = 2/w. If w=1, limit is 2.
    # Taylor expansion for sinc could be used, or just a mask.
    
    # Use small epsilon to avoid div by zero
    eps = 1e-6
    scale = torch.where((norm_xyz < eps).squeeze(-1),
                        2.0 / w.clamp(min=eps), # Approx for small angle
                        theta.squeeze(-1) / norm_xyz.squeeze(-1)).unsqueeze(-1)
    
    xi = scale * xyz
    return xi

def q_slerp(q0, q1, t):
    # q0, q1: (B, 4)
    # t: (B,) or scalar
    # Assumes q0, q1 normalized.
    
    # Ensure dot >= 0
    dot = (q0 * q1).sum(dim=-1)
    q1 = torch.where((dot < 0).unsqueeze(-1), -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # If theta ~ 0, use linear interpolation
    eps = 1e-6
    
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    
    # Where sin_theta is small, use linear interpolation weights
    linear_mask = sin_theta < eps
    w0[linear_mask] = 1 - t[linear_mask] if isinstance(t, torch.Tensor) else 1 - t
    w1[linear_mask] = t[linear_mask] if isinstance(t, torch.Tensor) else t
    
    return w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1


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
        # Debug: print fixed coords shape before encoding
        try:
            fc = fixed.get('coord')
            if isinstance(fc, torch.Tensor):
                print(f"[Docking] fixed coord shape: {tuple(fc.shape)}")
        except Exception:
            pass
        feat_f, off_f = self.encode(self.backbone_fixed, fixed)
        # Debug: print moved coords shape before encoding
        try:
            mc = moved.get('coord')
            if isinstance(mc, torch.Tensor):
                print(f"[Docking] moved coord shape: {tuple(mc.shape)}")
        except Exception:
            pass
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
        if self.training:
            return dict(
                loss=loss,
                loss_rot=loss_rot,
                loss_trans=loss_trans,
            )
        else:
            return dict(
                loss=loss,
                loss_rot=loss_rot,
                loss_trans=loss_trans,
                quat_pred=q_pred,
                trans_pred=trans_pred,
            )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


@MODELS.register_module()
class DockingWrapperFlow(nn.Module):
    def __init__(self,
                 backbone_fixed,
                 backbone_moved,
                 transformer=dict(d_model=512, nhead=8, num_layers=2, pool='mean'),
                 weight_fixed=None,
                 weight_moved=None,
                 freeze_backbone=True,
                 loss_rot_weight=1.0,
                 loss_trans_weight=1.0,
                 grid_size=0.5,
                 sigma_min=1e-4):
        super().__init__()
        self.backbone_fixed = build_model(backbone_fixed)
        self.backbone_moved = build_model(backbone_moved)
        
        # Set output dim to 6 for Flow Matching (3 rot vel + 3 trans vel)
        transformer = dict(transformer)
        transformer['out_dim'] = 6
        self.transformer = DockingTransformerFlow(**transformer)

        self.d_model = transformer.get('d_model', 512)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        load_backbone(self.backbone_fixed, weight_fixed, label="fixed")
        load_backbone(self.backbone_moved, weight_moved, label="moved")

        if freeze_backbone:
            for p in self.backbone_fixed.parameters():
                p.requires_grad = False
            for p in self.backbone_moved.parameters():
                p.requires_grad = False

        self.loss_rot_weight = loss_rot_weight
        self.loss_trans_weight = loss_trans_weight
        self.grid_size = grid_size
        self.sigma_min = sigma_min
        # Use MSE for velocity matching
        self.mse = nn.MSELoss()

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
        # input_dict keys: coord_fixed, atom_type_fixed, coord_moved, atom_type_moved, quat_gt, trans_gt
        dev = input_dict['coord_fixed'].device
        
        # 1. Prepare Ground Truth
        q_gt = input_dict['quat_gt'].to(dev) # (B, 4)
        t_gt = input_dict['trans_gt'].to(dev) # (B, 3)
        
        # 2. Sample Time and Noise
        B = q_gt.shape[0]
        if self.training:
            t = torch.rand(B, device=dev)
        else:
            t = input_dict.get('t', torch.rand(B, device=dev))
        
        # --- Translation Flow (Euclidean Rectified Flow) ---
        t_noise = torch.randn_like(t_gt)
        # Interpolate: X_t = t * X_1 + (1 - t) * X_0
        t_t = t[:, None] * t_gt + (1 - t[:, None]) * t_noise
        # Target Velocity: v = X_1 - X_0
        v_trans_target = t_gt - t_noise
        
        # --- Rotation Flow (SO(3) Geodesic Flow) ---
        # 1. Sample random quaternion on S^3
        q_noise = torch.randn_like(q_gt)
        q_noise = q_noise / (q_noise.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 2. Enforce shortest path (same hemisphere)
        dot = torch.sum(q_noise * q_gt, dim=-1, keepdim=True)
        q_noise = q_noise * torch.sign(dot)
        
        # 3. SLERP Interpolation for state q_t
        q_t = q_slerp(q_noise, q_gt, t)
        
        # 4. Target Velocity (Angular Velocity in SO(3) tangent space)
        # Compute relative rotation: q_rel = q_noise^{-1} * q_gt
        q_rel = q_multiply(q_inverse(q_noise), q_gt)
        
        # Extract axis-angle xi from q_rel
        # xi represents the log map Log_q_noise(q_gt)
        # For constant velocity flow on geodesic: omega = xi
        # (Strictly speaking for rectified flow on manifold, it is closely related to the log map)
        xi = q_to_axis_angle(q_rel)
        v_rot_target = xi # (B, 3)
        
        # 3. Apply Transformation to Moved Protein
        # Convert q_t to rotation matrix
        w, x, y, z = q_t.unbind(-1)
        R_t = torch.stack([
            1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
        ], dim=-1).reshape(B, 3, 3)
        
        coord_moved = input_dict['coord_moved'].to(dev)
        offset_moved = input_dict.get('offset_moved') # (B,)
        
        # Batch transform
        batch_idx = torch.zeros(coord_moved.shape[0], dtype=torch.long, device=dev)
        start = 0
        for i, end in enumerate(offset_moved):
            end = int(end.item())
            batch_idx[start:end] = i
            start = end
            
        R_expanded = R_t[batch_idx] # (N, 3, 3)
        t_expanded = t_t[batch_idx] # (N, 3)
        
        # X_t = R_t * X + t_t
        coord_moved_t = torch.bmm(R_expanded, coord_moved.unsqueeze(-1)).squeeze(-1) + t_expanded
        
        # 4. Encode
        fixed = dict(
            coord=input_dict['coord_fixed'].to(dev),
            grid_coord=input_dict.get('grid_coord_fixed'),
            atom_type=input_dict['atom_type_fixed'].to(dev),
            offset=input_dict.get('offset_fixed'),
        )
        if 'grid_coord' not in fixed:
             fixed['grid_size'] = self.grid_size
             
        moved = dict(
            coord=coord_moved_t,
            atom_type=input_dict['atom_type_moved'].to(dev),
            offset=offset_moved,
            grid_size=self.grid_size
        )
        
        feat_f, off_f = self.encode(self.backbone_fixed, fixed)
        feat_m, off_m = self.encode(self.backbone_moved, moved)
        
        # 5. Inject Time Embedding
        t_emb = self.time_mlp(t) # (B, C)
        t_emb_expanded = t_emb[batch_idx] # (N, C)
        
        feat_m = feat_m + t_emb_expanded
        
        # 6. Transformer
        # Output is (B, 6) -> split into (B, 3) and (B, 3)
        # v_rot_pred corresponds to angular velocity (axis-angle)
        preds = self.transformer(feat_f, off_f, feat_m, off_m)
        
        # So in this file I expect `preds` to be (B, 6).
        v_rot_pred = preds[:, :3]
        v_trans_pred = preds[:, 3:]
        
        # 7. Loss
        loss_trans = self.mse(v_trans_pred, v_trans_target)
        loss_rot = self.mse(v_rot_pred, v_rot_target)
        
        loss = self.loss_rot_weight * loss_rot + self.loss_trans_weight * loss_trans
        
        return dict(
            loss=loss,
            loss_rot=loss_rot,
            loss_trans=loss_trans,
            v_rot_pred=v_rot_pred,
            v_trans_pred=v_trans_pred,
            t=t
        )
