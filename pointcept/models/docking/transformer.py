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

    def forward(self, x_q, x_kv):
        # x_*: (B, N, C)
        q = self.ln_q(x_q)
        kv = self.ln_kv(x_kv)
        out, _ = self.attn(q, kv, kv)
        x = x_q + out
        x = x + self.ffn(x)
        return x


class DockingTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2, pool='mean'):
        super().__init__()
        self.layers = nn.ModuleList([CrossBlock(d_model, nhead) for _ in range(num_layers)])
        self.pool = pool
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 256), nn.GELU(), nn.Linear(256, 6))

    def _pool(self, x, offset):
        # x: (N,C) concat batches via offset; pool to (B,C)
        if offset is None:
            return x.mean(dim=0, keepdim=True)
        indptr = torch.nn.functional.pad(offset, (1, 0))
        outs = []
        for b in range(indptr.numel() - 1):
            s, e = indptr[b].item(), indptr[b + 1].item()
            xb = x[s:e]
            if self.pool == 'max':
                outs.append(xb.max(dim=0, keepdim=True).values)
            else:
                outs.append(xb.mean(dim=0, keepdim=True))
        return torch.cat(outs, dim=0)

    def forward(self, feat_fixed, offset_fixed, feat_moved, offset_moved):
        # pack as (B,N,C) chunked by batch
        # For simplicity, process per-batch via splitting
        # Build CSR pointers
        B = offset_fixed[-1].item() + 1
        indptr_f = torch.nn.functional.pad(offset_fixed, (1, 0))
        indptr_m = torch.nn.functional.pad(offset_moved, (1, 0))
        outs = []
        for b in range(B):
            sf, ef = indptr_f[b].item(), indptr_f[b + 1].item()
            sm, em = indptr_m[b].item(), indptr_m[b + 1].item()
            xf = feat_fixed[sf:ef].unsqueeze(0)  # (1,Nf,C)
            xm = feat_moved[sm:em].unsqueeze(0)
            for layer in self.layers:
                xf = layer(xf, xm)
                xm = layer(xm, xf)
            zf = xf.mean(dim=1)
            zm = xm.mean(dim=1)
            outs.append(torch.cat([zf, zm], dim=-1))
        z = torch.cat(outs, dim=0)
        pred = self.head(z)  # (B,6)
        rot = pred[:, :3]
        trans = pred[:, 3:]
        return rot, trans

