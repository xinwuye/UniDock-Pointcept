import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch, offset2bincount
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PTv3NoSkipDecoder
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module()
class PointTransformerVAE(nn.Module):
    """
    VAE wrapper around PT-v3 that reconstructs coordinates and input features.
    """

    def __init__(
        self,
        backbone_out_channels,
        feat_channels,
        latent_dim=64,
        coord_weight=1.0,
        feat_weight=1.0,
        kl_weight=1e-4,
        backbone=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.mu_head = nn.Linear(backbone_out_channels, latent_dim)
        self.logvar_head = nn.Linear(backbone_out_channels, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.coord_head = nn.Linear(latent_dim, 3)
        self.feat_head = nn.Linear(latent_dim, feat_channels)
        self.coord_weight = coord_weight
        self.feat_weight = feat_weight
        self.kl_weight = kl_weight

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        mu = self.mu_head(feat)
        logvar = self.logvar_head(feat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        hidden = self.decoder(z)
        recon_coord = self.coord_head(hidden)
        recon_feat = self.feat_head(hidden)

        coord_target = input_dict["coord"]
        feat_target = input_dict["atom_type"]

        coord_loss = F.mse_loss(recon_coord, coord_target)
        feat_loss = F.mse_loss(recon_feat, feat_target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = (
            self.coord_weight * coord_loss
            + self.feat_weight * feat_loss
            + self.kl_weight * kl_loss
        )

        output = dict(loss=loss)
        if not self.training:
            output.update(
                recon_coord=recon_coord,
                recon_feat=recon_feat,
                mu=mu,
                logvar=logvar,
            )
        return output


@MODELS.register_module()
class PointTransformerVAE2(nn.Module):
    """
    Standard VAE-style wrapper over PTv3:
    - Encoder: use PTv3 embedding + encoder to get low-resolution tokens
    - Global pooling to a single latent per-sample (max pooling)
    - Sample z, repeat to token count, concatenate noise, project to decoder input dim
    - Decoder: use PTv3 decoder to upsample back (no manual skip concatenation here)
    - Heads: predict coord + feature; compute reconstruction + KL losses
    """

    def __init__(
        self,
        encoder_out_channels,  # channels at encoder output (e.g., enc_channels[-1])
        decoder_in_channels,   # channels expected by decoder input
        feat_channels,         # output feature channels to reconstruct (e.g., num_atom_types)
        latent_dim=64,
        noise_dim=32,
        coord_weight=1.0,
        feat_weight=1.0,
        kl_weight=1e-4,
        backbone=None,
        decoder_cfg=None,  # dict: enc_channels, dec_channels, dec_depths, dec_num_head, dec_patch_size, plus flags
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.encoder_out_channels = encoder_out_channels
        self.decoder_in_channels = decoder_in_channels
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim

        # Latent heads on pooled encoder features
        self.mu_head = nn.Linear(encoder_out_channels, latent_dim)
        self.logvar_head = nn.Linear(encoder_out_channels, latent_dim)

        # Project repeated latent (+ noise) to decoder input channels
        self.z_to_dec = nn.Sequential(
            nn.Linear(latent_dim + noise_dim, decoder_in_channels),
            nn.GELU(),
            nn.Linear(decoder_in_channels, decoder_in_channels),
            nn.GELU(),
        )

        # Reconstruction heads on decoder top features
        self.coord_head = nn.Linear(decoder_in_channels, 3)
        self.feat_head = nn.Linear(decoder_in_channels, feat_channels)

        self.coord_weight = coord_weight
        self.feat_weight = feat_weight
        self.kl_weight = kl_weight
        # Build no-skip decoder if cfg provided
        self.decoder_noskip = None
        self.decoder_cfg = decoder_cfg or {}
        if self.decoder_cfg:
            self.decoder_noskip = PTv3NoSkipDecoder(
                enc_channels=self.decoder_cfg.get("enc_channels"),
                dec_channels=self.decoder_cfg.get("dec_channels"),
                dec_depths=self.decoder_cfg.get("dec_depths"),
                dec_num_head=self.decoder_cfg.get("dec_num_head"),
                dec_patch_size=self.decoder_cfg.get("dec_patch_size"),
                drop_path=self.decoder_cfg.get("drop_path", 0.0),
                norm_layer=self.decoder_cfg.get("norm_layer", None),
                act_layer=self.decoder_cfg.get("act_layer", nn.GELU),
                pre_norm=self.decoder_cfg.get("pre_norm", True),
                enable_rpe=self.decoder_cfg.get("enable_rpe", False),
                enable_flash=self.decoder_cfg.get("enable_flash", True),
                upcast_attention=self.decoder_cfg.get("upcast_attention", False),
                upcast_softmax=self.decoder_cfg.get("upcast_softmax", False),
                order=self.decoder_cfg.get("order", ("z", "z-trans")),
            )

    def forward(self, input_dict):
        # Build point
        point = Point(input_dict)

        # Prepare serialization & sparse tensor as PTv3 expects
        point.serialization(order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders)
        point.sparsify()
        # Run PTv3 encoder path explicitly (embedding + encoder only)
        point = self.backbone.embedding(point)
        point = self.backbone.enc(point)

        # Global pooling (max) over tokens per sample
        # indptr from offset: pad with leading 0
        indptr = nn.functional.pad(point.offset, (1, 0))
        pooled = torch_scatter.segment_csr(
            src=point.feat,
            indptr=indptr,
            reduce="max",
        )  # [B, C_enc]

        # Latent parameters & reparameterization
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # [B, latent_dim]

        # Repeat z to token-level and add noise
        batch_idx = offset2batch(point.offset)
        z_rep = z[batch_idx]  # [N, latent_dim]
        noise = torch.randn(z_rep.size(0), self.noise_dim, device=z_rep.device, dtype=z_rep.dtype)
        fused = torch.cat([z_rep, noise], dim=-1)

        # Project to decoder input channels; decode with no-skip PTv3-style decoder if available
        point.feat = self.z_to_dec(fused)
        point = self.decoder_noskip(point)
        dec_feat = point.feat

        recon_coord = self.coord_head(dec_feat)
        recon_feat = self.feat_head(dec_feat)

        # Losses against current batch tokens
        coord_loss = F.mse_loss(recon_coord, input_dict["coord"])
        feat_loss = F.mse_loss(recon_feat, input_dict["atom_type"])
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = self.coord_weight * coord_loss + self.feat_weight * feat_loss + self.kl_weight * kl_loss

        out = dict(loss=loss)
        if not self.training:
            out.update(recon_coord=recon_coord, recon_feat=recon_feat, mu=mu, logvar=logvar)
        return out
