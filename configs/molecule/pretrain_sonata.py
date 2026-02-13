"""
Sonata pretraining on molecule point clouds.

This config keeps Sonata's student-teacher pretraining objective, but switches
the data pipeline from scene RGB/normal to molecule coord/atom_type features
and molecule-style augmentations.
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
batch_size = 4  # total batch size across all GPUs
num_worker = 32
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# Expected layout:
#   <data_root>/{train,val,test}/<sample_id>/coord.npy, atom_type.npy
#   <data_root>/atom_types.json
data_root = "data/afdb_swissprot10000"
num_atom_types = 28

grid_size = 0.1

train_transform = [
    dict(type="CenterShiftMolecule"),
    dict(
        type="GridSampleAccumulate",
        grid_size=grid_size,
        feat_keys=["atom_type"],
    ),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "atom_type"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[],
        global_transform=[
            dict(type="CenterShiftMolecule"),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=1.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=1.0),
        ],
        local_transform=[
            dict(type="CenterShiftMolecule"),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=1.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=1.0),
        ],
        max_size=65536,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_atom_type",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_atom_type",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_atom_type"),
        local_feat_keys=("local_coord", "local_atom_type"),
    ),
]

data = dict(
    train=dict(
        type="MoleculeDataset",
        split=["train", "val", "test"],
        data_root=data_root,
        transform=train_transform,
        test_mode=False,
        loop=1,
    ),
)

# -----------------------------------------------------------------------------
# Model (Sonata-v1m1)
# -----------------------------------------------------------------------------
model = dict(
    type="Sonata-v1m1",
    backbone=dict(
        type="PT-v3m2",
        in_channels=num_atom_types + 3,  # atom_type + coord
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=1088,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)

# -----------------------------------------------------------------------------
# Optimizer / Scheduler
# -----------------------------------------------------------------------------
epoch = 200
base_lr = 0.004
lr_decay = 0.9  # layer-wise lr decay
base_wd = 0.04
final_wd = 0.2

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=5),
]
