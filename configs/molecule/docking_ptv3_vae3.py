_base_ = ["../_base_/default_runtime.py"]

# Misc
batch_size = 64
num_worker = 8
# batch_size = 16
mix_prob = 0.0
empty_cache = False
enable_amp = True
evaluate = True

# Roots
fixed_root = "data/pdbbind2020r1/proteins"
moved_root = "data/pdbbind2020r1/ligands"

# Build paired dataset with recorded rigid augments
grid_size = 0.5
# grid_size = 0.05
fixed_train_tf = [
    dict(type="ResetAugmentOps"),
    dict(type="CenterShiftMoleculeRecordSeq"),
    dict(type="RandomRotateRecordSeq", angle=[-1, 1], axis="z", p=1.0),
    dict(type="RandomRotateRecordSeq", angle=[-1, 1], axis="y", p=1.0),
    dict(type="RandomRotateRecordSeq", angle=[-1, 1], axis="x", p=1.0),
    dict(type="RandomShiftRecordSeq", shift=((-250.0, 250.0), (-250.0, 250.0), (-250.0, 250.0)), p=0.7),
    dict(type="Copy", keys_dict={"coord": "coord_aug_before_voxel"}),
    dict(type="GridSampleAccumulate", grid_size=grid_size, feat_keys=["atom_type"]),
    dict(type="ToTensor"),
]
moved_train_tf = fixed_train_tf

fixed_eval_tf = [
    dict(type="ResetAugmentOps"),
    dict(type="CenterShiftMoleculeRecordSeq"),
    dict(type="RandomRotateRecordSeq", angle=[-1, 1], axis="z", p=1.0),
    dict(type="RandomRotateRecordSeq", angle=[-1, 1], axis="y", p=1.0),
    dict(type="RandomRotateRecordSeq", angle=[-1, 1], axis="x", p=1.0),
    dict(type="RandomShiftRecordSeq", shift=((-250.0, 250.0), (-250.0, 250.0), (-250.0, 250.0)), p=0.7),
    dict(type="Copy", keys_dict={"coord": "coord_aug_before_voxel"}),
    dict(type="GridSampleAccumulate", grid_size=grid_size, feat_keys=["atom_type"]),
    dict(type="ToTensor"),
]
moved_eval_tf = fixed_eval_tf

data = dict(
    train=dict(
        type="DockingPairDataset",
        split="train",
        fixed_root=fixed_root,
        moved_root=moved_root,
        fixed_transform=fixed_train_tf,
        moved_transform=moved_train_tf,
        test_mode=False,
    ),
    val=dict(
        type="DockingPairDataset",
        split="val",
        fixed_root=fixed_root,
        moved_root=moved_root,
        fixed_transform=fixed_eval_tf,
        moved_transform=moved_eval_tf,
        test_mode=False,
    ),
    test=dict(
        type="DockingPairDataset",
        split="test",
        fixed_root=fixed_root,
        moved_root=moved_root,
        fixed_transform=fixed_eval_tf,
        moved_transform=moved_eval_tf,
        test_mode=False,
    ),
)

# Model: two encoders + cross-attn docking head
model = dict(
    type="DockingWrapper",
    backbone_fixed=dict(
        type="PT-v3m1",
        in_channels=45,  # proteins
        order=("z", "z-trans"),
        stride=(2,2,2,1),
        enc_depths=(2,2,2,6,2),
        enc_channels=(64,128,256,512,512),
        enc_num_head=(2,4,8,16,32),
        enc_patch_size=(256,256,256,256,256),
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
    backbone_moved=dict(
        type="PT-v3m1",
        in_channels=29,  # ligands
        order=("z", "z-trans"),
        stride=(2,2,2,1),
        enc_depths=(2,2,2,6,2),
        enc_channels=(64,128,256,512,512),
        enc_num_head=(2,4,8,16,32),
        enc_patch_size=(256,256,256,256,256),
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
    transformer=dict(d_model=512, nhead=8, num_layers=4, pool='mean'),
    # weight_fixed="exp/molecule/pdbbind2020r1-proteins-ptv3-vae3/model/model_best.pth",
    # weight_moved="exp/molecule/pdbbind2020r1-ligands-ptv3-vae3/model/model_best.pth",
    # freeze_backbone=True,
    freeze_backbone=False,
    loss_rot_weight=1.0,
    loss_trans_weight=1.0,
)

# Training schedule
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0001,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="DockingEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="DefaultTrainer")
