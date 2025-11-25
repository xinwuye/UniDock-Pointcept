_base_ = ["../_base_/default_runtime.py"]

# Misc
batch_size = 1024
num_worker = 8
mix_prob = 0.0
empty_cache = False
enable_amp = True
evaluate = True

data_root = "data/pdbbind2020r1/ligands"

num_atom_types = 29

model = dict(
    type="PointTransformerVAE3",
    feat_channels=num_atom_types,
    coord_weight=1.0,
    feat_weight=1.0,
    decoder_cfg=dict(
        # enc_channels=(32, 64, 128, 256, 512),
        enc_channels=(64, 128, 256, 512, 1024),
        # dec_channels=(64, 64, 128, 256),
        dec_channels=(64, 128, 256, 512),
        dec_out_channels=64,
        dec_depths=(2, 2, 2, 2),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(256, 256, 256, 256),
        drop_path=0.1,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        order=("z", "z-trans"),
    ),
    backbone=dict(
        type="PT-v3m1",
        in_channels=num_atom_types,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        # enc_channels=(32, 64, 128, 256, 512),
        enc_channels=(64, 128, 256, 512, 1024),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
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
)

epoch = 200
eval_epoch = 40
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0005, 0.00005],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.00005)]

dataset_type = "MoleculeDataset"
grid_size = 0.5

train_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", p=1.),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", p=1.),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", p=1.),
    dict(type="GridSampleAccumulate", grid_size=grid_size, feat_keys=["atom_type"]),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "atom_type"), feat_keys=("atom_type",)),
]

eval_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="GridSampleAccumulate", grid_size=grid_size, feat_keys=["atom_type"]),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "atom_type"), feat_keys=("atom_type",)),
]

data = dict(
    num_atom_types=num_atom_types,
    ignore_index=-1,
    train=dict(type=dataset_type, split="train", data_root=data_root, transform=train_transform, test_mode=False),
    val=dict(type=dataset_type, split="val", data_root=data_root, transform=eval_transform, test_mode=False),
    test=dict(type=dataset_type, split="test", data_root=data_root, transform=eval_transform, test_mode=False),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ReconstructionEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=False)
