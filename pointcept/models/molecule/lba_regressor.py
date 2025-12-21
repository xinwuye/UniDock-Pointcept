from __future__ import annotations

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point


def _mean_pool(feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool point features into per-sample features.

    Args:
      feat: (N, C)
      offset: (B,) int tensor of cumulative counts (end indices).
    Returns:
      pooled: (B, C)
    """
    if offset.numel() == 0:
        return feat.new_zeros((0, feat.shape[-1]))
    indptr = torch.nn.functional.pad(offset.to(torch.long), (1, 0), value=0)
    out = []
    for b in range(indptr.numel() - 1):
        s = int(indptr[b].item())
        e = int(indptr[b + 1].item())
        if e <= s:
            out.append(feat.new_zeros((feat.shape[-1],)))
        else:
            out.append(feat[s:e].mean(dim=0))
    return torch.stack(out, dim=0)


@MODELS.register_module()
class LBAPTV3Regressor(nn.Module):
    """
    Regression model for LBA:
      PT-v3 encoder -> pooling -> MLP head -> affinity scalar
    """

    def __init__(
        self,
        backbone: dict,
        head: dict | None = None,
        loss: dict | None = None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)

        head = head or {}
        in_dim = int(head.get("in_dim", 512))
        hidden = int(head.get("hidden_dim", 256))
        dropout = float(head.get("dropout", 0.0))
        self.pool = str(head.get("pool", "mean"))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

        loss = loss or {"type": "mse"}
        loss_type = str(loss.get("type", "mse")).lower()
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type in {"l1", "mae"}:
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          feat: (N, C)
          offset: (B,)
        """
        point = Point(batch)
        point.serialization(order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders)
        point.sparsify()
        point = self.backbone.embedding(point)
        point = self.backbone.enc(point)
        return point.feat, point.offset

    def forward(self, input_dict: dict) -> dict:
        # Expect: coord, grid_coord (optional), feat, offset, affinity (label)
        dev = input_dict["coord"].device

        batch = dict(
            coord=input_dict["coord"],
            grid_coord=input_dict.get("grid_coord"),
            feat=input_dict.get("feat"),
            offset=input_dict.get("offset"),
            atom_type=input_dict.get("atom_type"),
            identity=input_dict.get("identity"),
        )
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(dev)

        feat, offset = self.encode(batch)

        pooled = _mean_pool(feat, offset) if self.pool == "mean" else _mean_pool(feat, offset)
        pred = self.mlp(pooled).squeeze(-1)  # (B,)

        # IMPORTANT:
        # During training, Pointcept's default hooks (e.g. InformationWriter)
        # assume every tensor in model_output_dict is a scalar and call .item().
        # Therefore we only return scalar values in training mode.
        if "affinity" in input_dict:
            y = input_dict["affinity"]
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, device=dev)
            y = y.to(dev).view(-1).float()
            loss = self.loss_fn(pred.float(), y)
        else:
            y = None
            loss = None

        if self.training:
            out = {}
            if loss is not None:
                out["loss"] = loss
                # optional scalar metrics for logging
                if y is not None:
                    out["mae"] = torch.mean(torch.abs(pred.float() - y))
            return out

        out = {"pred": pred}
        if loss is not None:
            out["loss"] = loss
        return out


