#!/usr/bin/env python3
"""
Infer VAE3 embeddings over train/val/test subsets and save per-sample arrays.

Usage:
  python tools/vae3_infer.py \
    --config-file configs/molecule/pdbbind_ligand_ptv3_vae3.py \
    --weight exp/molecule/pdbbind2020r1-ligands-ptv3-vae3/model/model_best.pth \
    --data-root data/pdbbind2020r1/ligands \
    --save-root out/embeddings/pdbbind2020r1/ligand/molecule/pdbbind2020r1-ligands-ptv3-vae3 \
    --filename moved

Notes:
- filename must be either 'fixed' or 'moved'. The script saves one .npy per sample
  as: {save_root}/{subset}/{sample}/{filename}.npy
- Skips subsets that do not exist.
"""

import argparse
import os
import os.path as osp
import sys
import numpy as np
import torch
from collections import OrderedDict
from tqdm import tqdm

 # Add repo root to sys.path so `pointcept` is importable when running from anywhere
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointcept.utils.config import Config
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.models.utils.structure import Point


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-file', required=True, help='Path to config .py')
    ap.add_argument('--weight', required=True, help='Path to checkpoint .pth')
    ap.add_argument('--data-root', required=True, help='Dataset root containing train/val/test')
    ap.add_argument('--save-root', required=True, help='Output root to save embeddings')
    ap.add_argument('--filename', required=True, choices=['fixed', 'moved'], help="Output filename (without extension)")
    ap.add_argument('--device', default='cuda', help='cuda or cpu')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=8)
    return ap.parse_args()


def build_model_from_cfg(cfg, device, weight_path):
    model = build_model(cfg.model)
    model = model.to(device)
    assert osp.isfile(weight_path), f"Weight not found: {weight_path}"
    ckpt = torch.load(weight_path, map_location='cpu')
    weight = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        # strip possible 'module.' prefix
        if k.startswith('module.'):
            k = k[7:]
        weight[k] = v
    missing, unexpected = model.load_state_dict(weight, strict=False)
    if len(missing) > 0:
        print(f"[Warn] Missing keys: {missing}")
    if len(unexpected) > 0:
        print(f"[Warn] Unexpected keys: {unexpected}")
    model.eval()
    return model


@torch.no_grad()
def encode_backbone(backbone, batch, device):
    # Build Point and run encoder as in VAE forward
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=True)
    point = Point(batch)
    # serialization & sparsify as PTv3 expects
    point.serialization(order=backbone.order, shuffle_orders=backbone.shuffle_orders)
    point.sparsify()
    point = backbone.embedding(point)
    point = backbone.enc(point)
    # point.feat is encoder deepest token features (N4 x C)
    return point.feat.detach().cpu().numpy(), batch['offset'].detach().cpu().numpy()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    model = build_model_from_cfg(cfg, device, args.weight)
    # Obtain backbone (supports both VAE2/VAE3 wrappers or direct backbone in cfg)
    backbone = getattr(model, 'backbone', model)

    subsets = ['train', 'val', 'test']
    for subset in subsets:
        subset_root = osp.join(args.data_root, subset)
        if not osp.isdir(subset_root):
            print(f"[Info] Skip subset '{subset}' (not found: {subset_root})")
            continue

        # Reuse test/eval transform pipeline from cfg if present; fallback to identity
        ds_cfg = cfg.data.test if 'test' in cfg.data else cfg.data.val
        # clone and override split/data_root
        ds_cfg = dict(ds_cfg)
        ds_cfg.update(dict(split=subset, data_root=args.data_root, test_mode=False))
        dataset = build_dataset(ds_cfg)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        for sample in tqdm(loader, total=len(loader), desc=f"{subset}"):
            names = sample['name']
            if isinstance(names, str):
                names = [names]
            emb, offset = encode_backbone(backbone, sample, device)
            # Build CSR pointer from offset
            indptr = np.concatenate([[0], np.cumsum(offset)])
            assert len(names) == len(offset), "Mismatch between names and offset length"
            for i, name in enumerate(names):
                s, e = indptr[i], indptr[i + 1]
                out_dir = osp.join(args.save_root, subset, name)
                ensure_dir(out_dir)
                out_path = osp.join(out_dir, f"{args.filename}.npy")
                np.save(out_path, emb[s:e].astype(np.float32))
        print(f"[{subset}] Saved embeddings to {osp.join(args.save_root, subset)}")


if __name__ == '__main__':
    main()
