import re
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import cfg
from lib.models.decoder.decoder_dense import DPTHead
from lib.models.decoder.decoder_ground import GroundNormalHead
from lib.models.backbone.fcn import FCNResNetDecoder
from lib.models.decoder.randomization import ContentRandomization, StyleRandomization
from lib.models.decoder.pro_rand_conv import ProRandConvModule
from lib.utils.train_utils import tmp_freeze_module


class FECO(nn.Module):
    def __init__(self):
        super(FECO, self).__init__()

        # RandConv
        if 'resnet' in cfg.MODEL.backbone_type:
            data_mean = cfg.MODEL.img_mean
            data_std  = cfg.MODEL.img_std
        elif 'vit' in cfg.MODEL.backbone_type:
            data_mean = cfg.MODEL.img_mean_vit
            data_std  = cfg.MODEL.img_std_vit
        elif 'hrnet' in cfg.MODEL.backbone_type:
            data_mean = cfg.MODEL.img_mean_hrnet
            data_std  = cfg.MODEL.img_std_hrnet
        else:
            raise NotImplementedError

        # Low-level randomization with ProRandConv
        self.randconv = ProRandConvModule(
            in_channels=3,
            out_channels=3,
            mixing=True,
            identity_prob=0.1,
            data_mean=data_mean,
            data_std=data_std,
            clamp_output=True,
            use_deformable=True,
            L_max=10,
            b_delta=0.5
        )

        # Main contact backbone & decoder
        self.backbone = get_backbone_network(type=cfg.MODEL.backbone_type)
        self.decoder = get_decoder_network(type=cfg.MODEL.backbone_type)

        if cfg.MODEL.backbone_type in ['resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152']:
            context_dim_list = list(self.backbone.channels)
            context_dim = context_dim_list[-1]
        elif cfg.MODEL.backbone_type in ['vit-h-14']:
            context_dim = 1280
            context_dim_list = [context_dim]*4
        elif cfg.MODEL.backbone_type in ['vit-l-16']:
            context_dim = 1024
            context_dim_list = [context_dim]*4
        elif cfg.MODEL.backbone_type in ['vit-b-16']:
            context_dim = 768
            context_dim_list = [context_dim]*4
        elif cfg.MODEL.backbone_type in ['vit-s-16']:
            context_dim = 384
            context_dim_list = [context_dim]*4
        else:
            raise NotImplementedError

        self.context_dim = context_dim
        self.context_dim_list = context_dim_list
        self._chan2levels = {}
        for i, c in enumerate(self.context_dim_list):
            self._chan2levels.setdefault(c, []).append(i)
        self.default_level = max(0, min(getattr(self, "style_stage", 3) - 1, len(self.context_dim_list) - 1))

        # Ground feature encoder
        self.ground_feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(c, c, kernel_size=1),
                nn.ReLU(),
            ) for c in context_dim_list
        ])

        # Spatial attention and an adaptar for mixing main feature and ground feature
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(context_dim + context_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(256, 2, kernel_size=1) # 2 logits
        )
        self.contact_feat_adapter = nn.Sequential(
            nn.Conv2d(context_dim, context_dim, kernel_size=1),
            nn.ReLU()
        )

        # Randomization from SagNets
        self.content_randomization = ContentRandomization()
        self.style_randomization = StyleRandomization()

        # Adaptors for adversarial training (serves the role of layer1, layer2, layer3, layer4, bn1 in SagNets)
        self.adv_gamma1 = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in self.context_dim_list])
        self.adv_gamma2 = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in self.context_dim_list])
        self.style_adv_adapter1 = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size=3, padding=1) for c in self.context_dim_list]
        )
        self.style_adv_adapter2 = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size=3, padding=1) for c in self.context_dim_list]
        )
        for m in list(self.style_adv_adapter1) + list(self.style_adv_adapter2):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for g in self.adv_gamma1:
            g.data.fill_(0.02)
        for g in self.adv_gamma2:
            g.data.fill_(0.02)

        # Style decoder
        self.style_decoder = get_decoder_network(type=cfg.MODEL.backbone_type)

        # Pixel height map decoder & Foot segmentation mask decoder
        if 'resnet' in cfg.MODEL.backbone_type:
            resnet_type = int(re.findall(r'\d+', cfg.MODEL.backbone_type)[0])

            self.pixel_height_decoder = FCNResNetDecoder(resnet_type=resnet_type, num_classes=1)
            self.mask_decoder = FCNResNetDecoder(resnet_type=resnet_type, num_classes=1)
        elif 'vit' in cfg.MODEL.backbone_type:
            self.pixel_height_decoder = DPTHead(
                in_channels=context_dim, features=256, use_bn=False,
                out_channels=[context_dim, context_dim, context_dim, context_dim],
                use_clstoken=False
            )
            self.mask_decoder = DPTHead(
                in_channels=context_dim, features=256, use_bn=False,
                out_channels=[context_dim, context_dim, context_dim, context_dim],
                use_clstoken=False
            )
        else:
            raise NotImplementedError

        # Ground normal decoder
        self.ground_normal_decoder = GroundNormalHead(context_dim)

        self.eps = 1e-6
        self.temp = 1.5

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _check_pyramid(self, feats):
        for i, f in enumerate(feats):
            C = f.shape[1]
            expected = self.context_dim_list[i]
            assert C == expected, f"pyramid[{i}] channels {C} != expected {expected}"

    def _select_level_by_channels(self, C: int) -> int:
        levels = self._chan2levels.get(C, None)
        if levels is None or len(levels) == 0:
            raise ValueError(f"No adapter level matches channel size {C}. "
                            f"Available: {self.context_dim_list}")
        # prefer default_level if present, else take the last
        return self.default_level if self.default_level in levels else levels[-1]

    def _adv_residual_before(self, x: torch.Tensor, level: int = None) -> torch.Tensor:
        if level is None:
            level = self._select_level_by_channels(x.shape[1])
        return x + self.adv_gamma1[level].view(1, -1, 1, 1) * self.style_adv_adapter1[level](x)

    def _adv_residual_after(self, x: torch.Tensor, level: int = None) -> torch.Tensor:
        if level is None:
            level = self._select_level_by_channels(x.shape[1])
        return x + self.adv_gamma2[level].view(1, -1, 1, 1) * self.style_adv_adapter2[level](x)

    def mix_contact(self, ground_feat_, main_feat_):
        logits  = self.spatial_attention(torch.cat([ground_feat_.detach(), main_feat_], dim=1))
        weights = F.softmax(logits / self.temp, dim=1)
        weights = weights.clamp_min(self.eps)
        weights = weights / weights.sum(dim=1, keepdim=True)
        w_g, w_m = weights[:, 0:1], weights[:, 1:2]
        return w_g * ground_feat_.detach() + w_m * main_feat_

    def forward(self, inputs, mode='test'):
        image = inputs['input']['image'].to(self.device)
        if mode == 'train':
            image_shoe = inputs['input_shoe']['image'].to(self.device)

        if 'vit' in cfg.MODEL.backbone_type:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            if mode == 'train':
                image_shoe = F.interpolate(image_shoe, size=(224, 224), mode='bilinear', align_corners=False)

        ##################### Low-level randomization #####################
        if mode == 'train':
            self.randconv.randomize()
            rand_image1 = self.randconv(image)
            self.randconv.randomize()
            rand_image2 = self.randconv(image)
        ##################### Low-level randomization #####################

        _, _, img_h, img_w = image.shape
        img_size = max(img_h, img_w)

        ############################ Backbone #############################
        feat = self.backbone(image, return_intermediate=True)

        if mode == 'train':
            feat1 = self.backbone(rand_image1, return_intermediate=True)
            feat2 = self.backbone(rand_image2, return_intermediate=True)
        if mode == 'train':
            with torch.no_grad():
                feat_shoe = self.backbone(image_shoe, return_intermediate=True)
        ############################ Backbone #############################


        ######################## Foot Segmentation ########################
        if 'resnet' in cfg.MODEL.backbone_type:
            mask_foot_out = self.mask_decoder(feat['intermediate_feats'])
        elif 'vit' in cfg.MODEL.backbone_type:
            mask_foot_out = self.mask_decoder([(f, None) for f in feat['intermediate_feats']], patch_h=14, patch_w=14)
        mask_foot_out = F.interpolate(mask_foot_out, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]
        mask_prob  = torch.sigmoid(mask_foot_out.detach()[:, None])
        mask_foot_patch = F.interpolate(mask_prob, size=(feat['final_feat'].shape[2], feat['final_feat'].shape[3]), mode="area")
        mask_foot_patch = 1.0 * (mask_foot_patch > 0.5)
        mask_foot_patchs = [F.interpolate(mask_prob, size=(lf.shape[2], lf.shape[3]), mode="area") for lf in feat['intermediate_feats']]
        mask_foot_patchs = [1.0 * (p > 0.5) for p in mask_foot_patchs]

        if mode == 'train':
            if 'resnet' in cfg.MODEL.backbone_type:
                mask_foot_out1 = self.mask_decoder(feat1['intermediate_feats'])
                mask_foot_out2 = self.mask_decoder(feat2['intermediate_feats'])
            elif 'vit' in cfg.MODEL.backbone_type:
                mask_foot_out1 = self.mask_decoder([(f, None) for f in feat1['intermediate_feats']], patch_h=14, patch_w=14)
                mask_foot_out2 = self.mask_decoder([(f, None) for f in feat2['intermediate_feats']], patch_h=14, patch_w=14)
            mask_foot_out1 = F.interpolate(mask_foot_out1, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]
            mask_foot_out2 = F.interpolate(mask_foot_out2, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]        
            mask_prob1 = torch.sigmoid(mask_foot_out1.detach()[:, None])
            mask_prob2 = torch.sigmoid(mask_foot_out2.detach()[:, None])
            mask_foot_patch1 = F.interpolate(mask_prob1, size=(feat1['final_feat'].shape[2], feat1['final_feat'].shape[3]), mode="area")
            mask_foot_patch1 = 1.0 * (mask_foot_patch1 > 0.5)
            mask_foot_patch2 = F.interpolate(mask_prob2, size=(feat2['final_feat'].shape[2], feat2['final_feat'].shape[3]), mode="area")
            mask_foot_patch2 = 1.0 * (mask_foot_patch2 > 0.5)        
            mask_foot_patchs1 = [F.interpolate(mask_prob1, size=(lf.shape[2], lf.shape[3]), mode="area") for lf in feat1['intermediate_feats']]
            mask_foot_patchs1 = [1.0 * (p > 0.5) for p in mask_foot_patchs1]
            mask_foot_patchs2 = [F.interpolate(mask_prob2, size=(lf.shape[2], lf.shape[3]), mode="area") for lf in feat2['intermediate_feats']]
            mask_foot_patchs2 = [1.0 * (p > 0.5) for p in mask_foot_patchs2]
        ######################## Foot Segmentation ########################


        in_style_feat = self._adv_residual_before(feat['final_feat'].detach()) # We detach to only train _adv_residual_before and _adv_residual_after with adv_loss
        in_style_feats = [self._adv_residual_before(f.detach()) for f in feat['intermediate_feats']] # We detach to only train _adv_residual_before and _adv_residual_after with adv_loss
        if mode == 'train':
            in_style_feat1 = self._adv_residual_before(feat1['final_feat'].detach()) # We detach to only train _adv_residual_before and _adv_residual_after with adv_loss
            in_style_feat2 = self._adv_residual_before(feat2['final_feat'].detach()) # We detach to only train _adv_residual_before and _adv_residual_after with adv_loss
            in_style_feats1 = [self._adv_residual_before(f1.detach()) for f1 in feat1['intermediate_feats']] # We detach to only train _adv_residual_before and _adv_residual_after with adv_loss
            in_style_feats2 = [self._adv_residual_before(f2.detach()) for f2 in feat2['intermediate_feats']] # We detach to only train _adv_residual_before and _adv_residual_after with adv_loss


        ################# Shoe Content Randomization ######################
        if mode == 'train':
            in_shoe_feat = self._adv_residual_before(feat_shoe['final_feat'])
            in_shoe_feats = [self._adv_residual_before(f.detach()) for f in feat_shoe['intermediate_feats']]

            # Content Randomization
            # Multi-level features
            style_rand_feats = [self.content_randomization(b * m, s, mask=m) for b, m, s in zip(in_style_feats, mask_foot_patchs, in_shoe_feats)]
            style_rand_feats1 = [self.content_randomization(b * m, s, mask=m) for b, m, s in zip(in_style_feats1, mask_foot_patchs1, in_shoe_feats)]
            style_rand_feats2 = [self.content_randomization(b * m, s, mask=m) for b, m, s in zip(in_style_feats2, mask_foot_patchs2, in_shoe_feats)]
            # Final features
            style_rand_final = self.content_randomization(in_style_feat * mask_foot_patch, in_shoe_feat, mask=mask_foot_patch)
            style_rand_final1 = self.content_randomization(in_style_feat1 * mask_foot_patch1, in_shoe_feat, mask=mask_foot_patch1)
            style_rand_final2 = self.content_randomization(in_style_feat2 * mask_foot_patch2, in_shoe_feat, mask=mask_foot_patch2)

            # Adversarial Training: Train _adv_residual_before and _adv_residual_after with adv loss
            style_main_feats = [self._adv_residual_after(rf) * m for rf, m in zip(style_rand_feats, mask_foot_patchs)]
            style_main_feats1 = [self._adv_residual_after(rf1) * m1 for rf1, m1 in zip(style_rand_feats1, mask_foot_patchs1)]
            style_main_feats2 = [self._adv_residual_after(rf2) * m2 for rf2, m2 in zip(style_rand_feats2, mask_foot_patchs2)]
            style_main_feat = self._adv_residual_after(style_rand_final) * mask_foot_patch
            style_main_feat1 = self._adv_residual_after(style_rand_final1) * mask_foot_patch1
            style_main_feat2 = self._adv_residual_after(style_rand_final2) * mask_foot_patch2
        else:
            # Inference: use adapter outputs as-is
            style_main_feats = [self._adv_residual_after(f) for f in in_style_feats]
            style_main_feat = self._adv_residual_after(in_style_feat)
        ################# Shoe Content Randomization ######################


        ################# Ground feature encoding - Style ##################
        style_ground_feats = [enc(f) for enc, f in zip(self.ground_feature_encoders, style_main_feats)]
        style_ground_feat = style_ground_feats[-1]
        if mode == 'train':
            style_ground_feats1 = [enc(f) for enc, f in zip(self.ground_feature_encoders, style_main_feats1)]
            style_ground_feats2 = [enc(f) for enc, f in zip(self.ground_feature_encoders, style_main_feats2)]
            style_ground_feat1 = style_ground_feats1[-1]
            style_ground_feat2 = style_ground_feats2[-1]
        ################# Ground feature encoding - Style ##################


        ################# Ground-aware prediction - Style ##################
        # Pixel Height Map
        if 'resnet' in cfg.MODEL.backbone_type:
            pixel_height_style = self.pixel_height_decoder(style_ground_feats) * img_size
        elif 'vit' in cfg.MODEL.backbone_type:
            pixel_height_style = self.pixel_height_decoder([(f, None) for f in style_ground_feats], patch_h=14, patch_w=14) * img_size
        pixel_height_style = F.interpolate(pixel_height_style, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]
        if mode == 'train':
            if 'resnet' in cfg.MODEL.backbone_type:
                pixel_height_style1 = self.pixel_height_decoder(style_ground_feats1) * img_size
                pixel_height_style2 = self.pixel_height_decoder(style_ground_feats2) * img_size
            elif 'vit' in cfg.MODEL.backbone_type:
                pixel_height_style1 = self.pixel_height_decoder([(f, None) for f in style_ground_feats1], patch_h=14, patch_w=14) * img_size
                pixel_height_style2 = self.pixel_height_decoder([(f, None) for f in style_ground_feats2], patch_h=14, patch_w=14) * img_size
            pixel_height_style1 = F.interpolate(pixel_height_style1, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]
            pixel_height_style2 = F.interpolate(pixel_height_style2, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]

        # Ground Normal
        ground_normal_style = self.ground_normal_decoder(style_ground_feat * (1.0 - mask_foot_patch))
        if mode == 'train':
            ground_normal_style1 = self.ground_normal_decoder(style_ground_feat1 * (1.0 - mask_foot_patch1))
            ground_normal_style2 = self.ground_normal_decoder(style_ground_feat2 * (1.0 - mask_foot_patch2))
        ################# Ground-aware prediction - Style ##################


        ################### Spatial attention - Style ######################
        style_contact_feat = self.mix_contact(style_ground_feat, style_main_feat)
        style_contact_feat = self.contact_feat_adapter(style_contact_feat)
        if mode == 'train':
            style_contact_feat1 = self.mix_contact(style_ground_feat1, style_main_feat1)
            style_contact_feat2 = self.mix_contact(style_ground_feat2, style_main_feat2)        
            style_contact_feat1 = self.contact_feat_adapter(style_contact_feat1)
            style_contact_feat2 = self.contact_feat_adapter(style_contact_feat2)
        ################### Spatial attention - Style ######################


        ################## Foot contact decoder - Style ####################
        # Non-Adversarial Training
        contact_style_out, contact_joint_style_out, contact_joint_openpose_style_out, contact_per_foot_openpose_style_out = self.style_decoder(style_contact_feat.detach()) # Detach feature because SagNets only train style decoder with style loss
        if mode == 'train':
            contact_style_out1, contact_joint_style_out1, contact_joint_openpose_style_out1, contact_per_foot_openpose_style_out1 = self.style_decoder(style_contact_feat1.detach()) # Detach feature because SagNets only train style decoder with style loss
            contact_style_out2, contact_joint_style_out2, contact_joint_openpose_style_out2, contact_per_foot_openpose_style_out2 = self.style_decoder(style_contact_feat2.detach()) # Detach feature because SagNets only train style decoder with style loss
        
        # Adversarial Training
        with tmp_freeze_module(self.style_decoder): # We freeze style_decoder as SagNets only train _adv_residual_before and _adv_residual_after with adv_loss
            contact_adv_out,  contact_joint_adv_out,  contact_joint_openpose_adv_out,  contact_per_foot_openpose_adv_out  = self.style_decoder(style_main_feat)
            if mode == 'train':
                contact_adv_out1, contact_joint_adv_out1, contact_joint_openpose_adv_out1, contact_per_foot_openpose_adv_out1 = self.style_decoder(style_main_feat1)
                contact_adv_out2, contact_joint_adv_out2, contact_joint_openpose_adv_out2, contact_per_foot_openpose_adv_out2 = self.style_decoder(style_main_feat2)
        ################## Foot contact decoder - Style ####################


        in_feats = [self._adv_residual_before(f) for f in feat['intermediate_feats']] # We train backbone and all other modules with main loss
        in_feat = self._adv_residual_before(feat['final_feat']) # We train backbone and all other modules with main loss
        if mode == 'train':
            in_feats1 = [self._adv_residual_before(f) for f in feat1['intermediate_feats']] # We train backbone and all other modules with main loss
            in_feats2 = [self._adv_residual_before(f) for f in feat2['intermediate_feats']] # We train backbone and all other modules with main loss
            in_feat1 = self._adv_residual_before(feat1['final_feat']) # We train backbone and all other modules with main loss
            in_feat2 = self._adv_residual_before(feat2['final_feat']) # We train backbone and all other modules with main loss


        ################### Shoe Style Randomization ######################
        # Shoe Style Randomization for the main path
        if mode == 'train':
            alpha = torch.rand(image.shape[0], 1, 1, device=self.device)

            # Style Randomization (Only style randomize foot region)
            # Multi-level features
            rand_feats = [self.style_randomization(f * m, d, alpha) * m + f * (1 - m) for f, m, d in zip(in_feats, mask_foot_patchs, in_shoe_feats)]
            rand_feats1 = [self.style_randomization(f * m, d, alpha) * m + f * (1 - m) for f, m, d in zip(in_feats1, mask_foot_patchs1, in_shoe_feats)]
            rand_feats2 = [self.style_randomization(f * m, d, alpha) * m + f * (1 - m) for f, m, d in zip(in_feats2, mask_foot_patchs2, in_shoe_feats)]
            # Final features
            rand_final = self.style_randomization(in_feat * mask_foot_patch, in_shoe_feat, alpha) * mask_foot_patch + in_feat * (1 - mask_foot_patch)
            rand_final1 = self.style_randomization(in_feat1 * mask_foot_patch1, in_shoe_feat, alpha) * mask_foot_patch1 + in_feat1 * (1 - mask_foot_patch1)
            rand_final2 = self.style_randomization(in_feat2 * mask_foot_patch2, in_shoe_feat, alpha) * mask_foot_patch2 + in_feat2 * (1 - mask_foot_patch2)

            # Adaptar after Style Randomization
            main_feats = [self._adv_residual_after(rf) for rf in rand_feats]
            main_feats1 = [self._adv_residual_after(rf1) for rf1 in rand_feats1]
            main_feats2 = [self._adv_residual_after(rf2) for rf2 in rand_feats2]
            main_feat = self._adv_residual_after(rand_final)
            main_feat1 = self._adv_residual_after(rand_final1)
            main_feat2 = self._adv_residual_after(rand_final2)
        else:
            # Inference: use adapter outputs as-is
            main_feats = [self._adv_residual_after(f) for f in in_feats]
            main_feat = self._adv_residual_after(in_feat)
        ################### Shoe Style Randomization ######################


        ##################### Ground feature encoding #####################
        ground_feats = [enc(f) for enc, f in zip(self.ground_feature_encoders, main_feats)]
        ground_feat = ground_feats[-1]
        if mode == 'train':
            ground_feats1 = [enc(f) for enc, f in zip(self.ground_feature_encoders, main_feats1)]
            ground_feats2 = [enc(f) for enc, f in zip(self.ground_feature_encoders, main_feats2)]        
            ground_feat1 = ground_feats1[-1]
            ground_feat2 = ground_feats2[-1]
        ##################### Ground feature encoding #####################


        ##################### Ground-aware prediction #####################
        # Pixel Height Map
        if 'resnet' in cfg.MODEL.backbone_type:
            pixel_height_out = self.pixel_height_decoder(ground_feats) * img_size
        elif 'vit' in cfg.MODEL.backbone_type:
            pixel_height_out = self.pixel_height_decoder([(f, None) for f in ground_feats], patch_h=14, patch_w=14) * img_size
        pixel_height_out = F.interpolate(pixel_height_out, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]
        if mode == 'train':
            if 'resnet' in cfg.MODEL.backbone_type:
                pixel_height_out1 = self.pixel_height_decoder(ground_feats1) * img_size
                pixel_height_out2 = self.pixel_height_decoder(ground_feats2) * img_size
            elif 'vit' in cfg.MODEL.backbone_type:
                pixel_height_out1 = self.pixel_height_decoder([(f, None) for f in ground_feats1], patch_h=14, patch_w=14) * img_size
                pixel_height_out2 = self.pixel_height_decoder([(f, None) for f in ground_feats2], patch_h=14, patch_w=14) * img_size
            pixel_height_out1 = F.interpolate(pixel_height_out1, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]
            pixel_height_out2 = F.interpolate(pixel_height_out2, size=cfg.MODEL.input_img_shape, mode="bilinear", align_corners=True)[:, 0]

        # Ground Normal
        ground_normal_out = self.ground_normal_decoder(ground_feat *  (1.0 - mask_foot_patch))
        if mode == 'train':
            ground_normal_out1 = self.ground_normal_decoder(ground_feat1 * (1.0 - mask_foot_patch1))
            ground_normal_out2 = self.ground_normal_decoder(ground_feat2 * (1.0 - mask_foot_patch2))
        ##################### Ground-aware prediction #####################


        ######################## Spatial attention #########################
        contact_feat = self.mix_contact(ground_feat, main_feat)
        contact_feat = self.contact_feat_adapter(contact_feat)
        if mode == 'train':
            contact_feat1 = self.mix_contact(ground_feat1, main_feat1)
            contact_feat2 = self.mix_contact(ground_feat2, main_feat2)
            contact_feat1 = self.contact_feat_adapter(contact_feat1)
            contact_feat2 = self.contact_feat_adapter(contact_feat2)
        ######################## Spatial attention #########################


        ###################### Foot contact decoder #######################
        contact_out, contact_joint_out, contact_joint_openpose_out, contact_per_foot_out = self.decoder(contact_feat)
        if mode == 'train':
            contact_out1, contact_joint_out1, contact_joint_openpose_out1, contact_per_foot_out1 = self.decoder(contact_feat1)
            contact_out2, contact_joint_out2, contact_joint_openpose_out2, contact_per_foot_out2 = self.decoder(contact_feat2)
        ###################### Foot contact decoder #######################


        if mode == 'train':
            return dict(
                contact_style_out=contact_style_out, contact_joint_style_out=contact_joint_style_out, contact_joint_openpose_style_out=contact_joint_openpose_style_out,
                contact_style_out1=contact_style_out1, contact_joint_style_out1=contact_joint_style_out1, contact_joint_openpose_style_out1=contact_joint_openpose_style_out1,
                contact_style_out2=contact_style_out2, contact_joint_style_out2=contact_joint_style_out2, contact_joint_openpose_style_out2=contact_joint_openpose_style_out2,
                contact_adv_out=contact_adv_out, contact_joint_adv_out=contact_joint_adv_out, contact_joint_openpose_adv_out=contact_joint_openpose_adv_out,
                contact_adv_out1=contact_adv_out1, contact_joint_adv_out1=contact_joint_adv_out1, contact_joint_openpose_adv_out1=contact_joint_openpose_adv_out1,
                contact_adv_out2=contact_adv_out2, contact_joint_adv_out2=contact_joint_adv_out2, contact_joint_openpose_adv_out2=contact_joint_openpose_adv_out2,
                pixel_height_style_out=pixel_height_style, ground_normal_style_out=ground_normal_style,
                pixel_height_style_out1=pixel_height_style1, ground_normal_style_out1=ground_normal_style1,
                pixel_height_style_out2=pixel_height_style2, ground_normal_style_out2=ground_normal_style2,
                contact_out=contact_out, contact_joint_out=contact_joint_out, contact_joint_openpose_out=contact_joint_openpose_out,
                contact_out1=contact_out1, contact_joint_out1=contact_joint_out1, contact_joint_openpose_out1=contact_joint_openpose_out1,
                contact_out2=contact_out2, contact_joint_out2=contact_joint_out2, contact_joint_openpose_out2=contact_joint_openpose_out2,
                pixel_height_out=pixel_height_out, mask_foot_out=mask_foot_out, ground_normal_out=ground_normal_out,
                pixel_height_out1=pixel_height_out1, mask_foot_out1=mask_foot_out1, ground_normal_out1=ground_normal_out1,
                pixel_height_out2=pixel_height_out2, mask_foot_out2=mask_foot_out2, ground_normal_out2=ground_normal_out2,
            )
        else:
            return dict(
                contact_out=contact_out, contact_joint_out=contact_joint_out, contact_joint_openpose_out=contact_joint_openpose_out, contact_per_foot_out=contact_per_foot_out,
                pixel_height_out=pixel_height_out, mask_foot_out=mask_foot_out, ground_normal_out=ground_normal_out,
            )


def get_backbone_network(type='vit-h-14'):
    if type in ['resnet-18']:
        from lib.models.backbone.fcn import FCNResNetEncoder
        backbone = FCNResNetEncoder(resnet_type=18, style_stage=3)
        backbone.init_weights()
    elif type in ['resnet-34']:
        from lib.models.backbone.fcn import FCNResNetEncoder
        backbone = FCNResNetEncoder(resnet_type=34, style_stage=3)
        backbone.init_weights()
    elif type in ['resnet-50']:
        from lib.models.backbone.fcn import FCNResNetEncoder
        backbone = FCNResNetEncoder(resnet_type=50, style_stage=3)
        backbone.init_weights()
    elif type in ['resnet-101']:
        from lib.models.backbone.fcn import FCNResNetEncoder
        backbone = FCNResNetEncoder(resnet_type=101, style_stage=3)
        backbone.init_weights()
    elif type in ['resnet-152']:
        from lib.models.backbone.fcn import FCNResNetEncoder
        backbone = FCNResNetEncoder(resnet_type=152, style_stage=3)
        backbone.init_weights()
    elif type in ['hrnet-w32']:
        from lib.models.backbone.hrnet import HighResolutionNet
        from lib.utils.func_utils import load_config
        config = load_config(cfg.MODEL.hrnet_w32_backbone_config_path)
        pretrained = cfg.MODEL.hrnet_w32_backbone_pretrained_path
        backbone = HighResolutionNet(config)
        backbone.init_weights(pretrained=pretrained)
    elif type in ['hrnet-w48']:
        from lib.models.backbone.hrnet import HighResolutionNet
        from lib.utils.func_utils import load_config
        config = load_config(cfg.MODEL.hrnet_w48_backbone_config_path)
        pretrained = cfg.MODEL.hrnet_w48_backbone_pretrained_path
        backbone = HighResolutionNet(config)
        backbone.init_weights(pretrained=pretrained)
    elif type in ['vit-s-16']:
        from lib.models.backbone.vit import ViTBackbone
        backbone = ViTBackbone(model_name='vit_small_patch16_224', pretrained=True, out_indices=(2, 5, 8, 11))
    elif type in ['vit-b-16']:
        from lib.models.backbone.vit import ViTBackbone
        backbone = ViTBackbone(model_name='vit_base_patch16_224', pretrained=True, out_indices=(2, 5, 8, 11))
    elif type in ['vit-l-16']:
        from lib.models.backbone.vit import ViTBackbone
        backbone = ViTBackbone(model_name='vit_large_patch16_224', pretrained=True, out_indices=(4, 11, 17, 23))
    elif type in ['vit-h-14']:
        from lib.models.backbone.vit import ViTBackbone
        backbone = ViTBackbone(model_name='vit_huge_patch14_224', pretrained=True, out_indices=(2, 11, 21, 31))
    elif type in ['dinov2_vitb16']:
        backbone = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
    else:
        raise NotImplementedError

    return backbone


def get_decoder_network(type='vit-h-14'):
    from lib.models.decoder.decoder_hamer_style import ContactTransformerDecoderHead
    decoder = ContactTransformerDecoderHead()
    return decoder