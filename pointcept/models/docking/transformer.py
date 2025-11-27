import torch
import torch.nn as nn


class CrossBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x_q, x_kv, mask_kv=None):
        # x_*: (B, N, C); mask_kv: (B, N) True indicates padding to ignore
        q = self.ln_q(x_q)
        kv = self.ln_kv(x_kv)
        out, _ = self.attn(q, kv, kv, key_padding_mask=mask_kv)
        x = x_q + out
        x = x + self.ffn(x)
        return x


class DockingTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2, pool='mean'):
        super().__init__()
        # Use separate layers for fixed and moved streams to avoid weight sharing
        # Fixed stream only needs N-1 layers because the N-th layer output is unused
        self.layers_fixed = nn.ModuleList([CrossBlock(d_model, nhead) for _ in range(num_layers - 1)])
        self.layers_moved = nn.ModuleList([CrossBlock(d_model, nhead) for _ in range(num_layers)])
        self.pool = pool
        # Output: 4 for quaternion (w,x,y,z) + 3 for translation
        self.head = nn.Sequential(
            # nn.LayerNorm(d_model * 2),
            # nn.Linear(d_model * 2, 256),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 7),
        )

    @staticmethod
    def _pad_batch(feat, offset):
        """
        Convert flat (N,C) with CSR offset to padded (B, N_max, C) and mask (B, N_max).
        mask=True denotes padded positions to ignore in attention and pooling.
        """
        # offset is cumulative counts per sample, len(offset) == B
        indptr = torch.nn.functional.pad(offset, (1, 0))
        B = offset.numel()
        lengths = (indptr[1:] - indptr[:-1]).tolist()
        N_max = max(lengths) if lengths else 0
        C = feat.size(-1)
        device = feat.device
        x_pad = feat.new_zeros((B, N_max, C))
        mask = torch.ones((B, N_max), dtype=torch.bool, device=device)
        for b in range(B):
            s, e = indptr[b].item(), indptr[b + 1].item()
            n = e - s
            if n >= 0:
                x_pad[b, :n] = feat[s:e]
                mask[b, :n] = False
        return x_pad, mask

    def forward(self, feat_fixed, offset_fixed, feat_moved, offset_moved):
        # Build padded batches
        xf, mask_f = self._pad_batch(feat_fixed, offset_fixed)
        xm, mask_m = self._pad_batch(feat_moved, offset_moved)
        print('shape of xf: ', xf.shape)
        print('shape of xm: ', xm.shape)
        
        # Cross-attention with padding masks
        # Iterate through layers. layers_moved has 1 more layer than layers_fixed
        num_common_layers = len(self.layers_fixed)
        
        # 1. Run common layers (both update)
        for i in range(num_common_layers):
            layer_f = self.layers_fixed[i]
            layer_m = self.layers_moved[i]
            
            xf_new = layer_f(xf, xm, mask_kv=mask_m)
            xm_new = layer_m(xm, xf, mask_kv=mask_f)
            
            xf, xm = xf_new, xm_new
            
        # 2. Run the last moved layer (only xm updates, using previous xf)
        last_layer_m = self.layers_moved[-1]
        xm = last_layer_m(xm, xf, mask_kv=mask_f)
            
        # Masked mean pooling per sample
        def masked_mean(x, mask):
            valid = (~mask).unsqueeze(-1)  # (B,N,1)
            x_sum = (x * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            return x_sum / denom
        # zf = masked_mean(xf, mask_f)
        zm = masked_mean(xm, mask_m)
        # z = torch.cat([zf, zm], dim=-1)
        z = zm
        pred = self.head(z)  # (B,7)
        rot = pred[:, :4]    # quaternion raw
        trans = pred[:, 4:]
        return rot, trans


class DockingTransformerFlow(DockingTransformer):
    def __init__(self, d_model=512, nhead=8, num_layers=2, pool='mean', out_dim=6):
        super().__init__(d_model, nhead, num_layers, pool)
        # Override head for Flow Matching output dim
        # Output: out_dim (default 6 for 3D rot vel + 3D trans vel)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, feat_fixed, offset_fixed, feat_moved, offset_moved):
        # Re-implement forward to avoid splitting 7 dims
        # Copy-paste logic from parent but change return
        
        # Build padded batches
        xf, mask_f = self._pad_batch(feat_fixed, offset_fixed)
        xm, mask_m = self._pad_batch(feat_moved, offset_moved)
        # print('shape of xf: ', xf.shape)
        # print('shape of xm: ', xm.shape)
        
        num_common_layers = len(self.layers_fixed)
        
        for i in range(num_common_layers):
            layer_f = self.layers_fixed[i]
            layer_m = self.layers_moved[i]
            
            xf_new = layer_f(xf, xm, mask_kv=mask_m)
            xm_new = layer_m(xm, xf, mask_kv=mask_f)
            
            xf, xm = xf_new, xm_new
            
        last_layer_m = self.layers_moved[-1]
        xm = last_layer_m(xm, xf, mask_kv=mask_f)
            
        def masked_mean(x, mask):
            valid = (~mask).unsqueeze(-1)
            x_sum = (x * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            return x_sum / denom
            
        zm = masked_mean(xm, mask_m)
        z = zm
        pred = self.head(z)  # (B, out_dim)
        
        # Return full prediction tensor (B, 6)
        return pred
