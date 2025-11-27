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
        self.layers_fixed = nn.ModuleList([CrossBlock(d_model, nhead) for _ in range(num_layers)])
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
        # Cross-attention with padding masks, parallel update without weight sharing
        for layer_f, layer_m in zip(self.layers_fixed, self.layers_moved):
            # Parallel computation: compute new states based on current states
            xf_new = layer_f(xf, xm, mask_kv=mask_m)
            xm_new = layer_m(xm, xf, mask_kv=mask_f)
            # Update states simultaneously
            xf, xm = xf_new, xm_new
            
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
