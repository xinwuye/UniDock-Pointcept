_base_ = ["../_base_/default_runtime.py"]

# Misc
batch_size = 192
batch_size_val = 512
num_worker = 8
mix_prob = 0.0
empty_cache = False
enable_amp = True
evaluate = True


# Dataset root (produced by convert_atom3d_lba_to_pointcept.py)
data_root = "data/lba/split-by-sequence-identity-30processed"

# Atom types count in: data/lba/split-by-sequence-identity-30processed/atom_types.json
# (kept as a constant here to avoid imports inside config)
num_atom_types = 26


grid_size = 0.1
train_tf = [
    dict(type="CenterShiftMolecule"),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", p=1.0),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", p=1.0),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", p=1.0),
    dict(type="Copy", keys_dict={"coord": "coord_aug_before_voxel"}),
    dict(type="GridSampleAccumulate", grid_size=grid_size, feat_keys=["atom_type", "identity"]),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "atom_type", "identity", "affinity"),
        feat_keys=("atom_type", "identity", "coord"),
    ),
]

eval_tf = [
    dict(type="CenterShiftMolecule"),
    dict(type="Copy", keys_dict={"coord": "coord_aug_before_voxel"}),
    dict(type="GridSampleAccumulate", grid_size=grid_size, feat_keys=["atom_type", "identity"]),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "atom_type", "identity", "affinity"),
        feat_keys=("atom_type", "identity", "coord"),
    ),
]

data = dict(
    train=dict(
        type="LBARegressionDataset",
        split="train",
        data_root=data_root,
        transform=train_tf,
        test_mode=False,
    ),
    val=dict(
        type="LBARegressionDataset",
        split="val",
        data_root=data_root,
        transform=eval_tf,
        test_mode=False,
    ),
    test=dict(
        type="LBARegressionDataset",
        split="test",
        data_root=data_root,
        transform=eval_tf,
        test_mode=False,
    ),
)


# Model: PT-v3 encoder + pooling + MLP head
model = dict(
    type="LBAPTV3Regressor",
    backbone=dict(
        type="PT-v3m1",
        in_channels=num_atom_types + 2 + 3,  # atom_type + identity + coord
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2, 2, 2),
        enc_depths=(2, 2, 2, 2, 2, 2, 2),
        enc_channels=(64, 128, 256, 512, 512, 512, 512),
        enc_num_head=(2, 4, 8, 16, 32, 32, 32),
        # enc_patch_size=(256, 256, 256, 256, 256, 256, 256),
        enc_patch_size=(128, 128, 128, 128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=True,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("Ligand",),
    ),
    head=dict(in_dim=512, hidden_dim=256, dropout=0.0, pool="mean"),
    loss=dict(type="mse"),
)


# Training schedule
epoch = 500
eval_epoch = 500
lr = 0.00005
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)
# scheduler = dict(
#     type="OneCycleLR",
#     max_lr=lr,
#     pct_start=0.05,
#     anneal_strategy="cos",
#     div_factor=10.0,
#     final_div_factor=1000.0,
# )
scheduler = dict(type="ExpLR", gamma=1.0)


hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LBARegressionEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="DefaultTrainer")


