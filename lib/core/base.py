import numpy as np

import torch
import torch.nn.functional as F

from lib.core.config import cfg
from lib.core.loss import ClsLoss, DiceLoss, CosineLoss, AdvLoss, NormLoss

J_regressor_foot = torch.tensor(np.load(cfg.MODEL.J_regressor_foot_path), dtype=torch.float32).to('cuda')
J_regressor_foot_openpose = torch.tensor(np.load(cfg.MODEL.J_regressor_foot_openpose_path), dtype=torch.float32).to('cuda')

# Loss function
class_loss = ClsLoss()
dice_loss = DiceLoss()
adv_loss = AdvLoss()
cos_loss = CosineLoss()
norm_loss = NormLoss()


# ---- top-level helpers (no nested defs) ----
def to_two_class_probs_from_logits(x, eps=1e-7):
    """
    x: logits of shape (B, N) or (B, 1, H, W) or (B, H, W) or (B, C, H, W) with C==1
    returns probs of shape (B, 2, N_flat)
    """
    p = torch.sigmoid(x)
    if p.dim() > 2:
        p = p.flatten(1)  # (B, N)
    p = p.clamp(eps, 1 - eps)
    return torch.stack([1 - p, p], dim=1)  # (B, 2, N)

def jsd3_from_logits(clean, aug1, aug2, eps=1e-7):
    """
    Implements:
      p_mixture = clamp((p_clean + p_aug1 + p_aug2)/3, eps, 1).log()
      inv = (KL(p_mixture, p_clean) + KL(p_mixture, p_aug1) + KL(p_mixture, p_aug2))/3
    using batchmean reduction.
    """
    p_clean = to_two_class_probs_from_logits(clean, eps)
    p_aug1  = to_two_class_probs_from_logits(aug1,  eps)
    p_aug2  = to_two_class_probs_from_logits(aug2,  eps)

    p_mixture_log = ((p_clean + p_aug1 + p_aug2) / 3.0).clamp(eps, 1.0).log()
    inv = (
        F.kl_div(p_mixture_log, p_clean, reduction='batchmean') +
        F.kl_div(p_mixture_log, p_aug1,  reduction='batchmean') +
        F.kl_div(p_mixture_log, p_aug2,  reduction='batchmean')
    ) / 3.0
    return inv
# --------------------------------------------

def compute_contact_loss(preds, targets):
    total_loss = 0

    is_3D = targets['contact_data']['is_3D']
    is_2D = 1 - is_3D

    contact_f_mesh = targets['contact_data']['contact_f'].float()
    contact_f_joint = 1 * (torch.mm(contact_f_mesh, J_regressor_foot.T) > 0)
    contact_f_joint_openpose_3d = 1 * (torch.mm(contact_f_mesh, J_regressor_foot_openpose.T) > 0)
    contact_f_joint_openpose_2d = targets['contact_data']['contact_f_joint_openpose_2d']

    pixel_height_map = targets['pixel_height_map']
    valid_height_mask = targets['valid_height_mask']
    ground_normal = targets['ground_data']['ground_normal']

    # Task loss - clean branch
    contact_mesh_loss = class_loss(preds['contact_out'], contact_f_mesh, is_3D)
    contact_joint_loss = class_loss(preds['contact_joint_out'], contact_f_joint, is_3D)
    contact_joint_openpose_loss = class_loss(preds['contact_joint_openpose_out'], contact_f_joint_openpose_3d, is_3D)
    contact_joint_openpose_2d_loss = class_loss(preds['contact_joint_openpose_out'], contact_f_joint_openpose_2d, is_2D)
    main_contact_loss = cfg.TRAIN.contact_loss_weight * (contact_mesh_loss + contact_joint_loss + contact_joint_openpose_loss + contact_joint_openpose_2d_loss)

    contact_mesh_loss1 = class_loss(preds['contact_out1'], contact_f_mesh, is_3D)
    contact_joint_loss1 = class_loss(preds['contact_joint_out1'], contact_f_joint, is_3D)
    contact_joint_openpose_loss1 = class_loss(preds['contact_joint_openpose_out1'], contact_f_joint_openpose_3d, is_3D)
    contact_joint_openpose_2d_loss1 = class_loss(preds['contact_joint_openpose_out1'], contact_f_joint_openpose_2d, is_2D)
    main_contact_loss1 = cfg.TRAIN.contact_loss_weight * (contact_mesh_loss1 + contact_joint_loss1 + contact_joint_openpose_loss1 + contact_joint_openpose_2d_loss1)

    contact_mesh_loss2 = class_loss(preds['contact_out2'], contact_f_mesh, is_3D)
    contact_joint_loss2 = class_loss(preds['contact_joint_out2'], contact_f_joint, is_3D)
    contact_joint_openpose_loss2 = class_loss(preds['contact_joint_openpose_out2'], contact_f_joint_openpose_3d, is_3D)
    contact_joint_openpose_2d_loss2 = class_loss(preds['contact_joint_openpose_out2'], contact_f_joint_openpose_2d, is_2D)
    main_contact_loss2 = cfg.TRAIN.contact_loss_weight * (contact_mesh_loss2 + contact_joint_loss2 + contact_joint_openpose_loss2 + contact_joint_openpose_2d_loss2)

    # Style loss
    contact_style_mesh_loss = class_loss(preds['contact_style_out'], contact_f_mesh, is_3D)
    contact_style_joint_loss = class_loss(preds['contact_joint_style_out'], contact_f_joint, is_3D)
    contact_style_joint_openpose_loss = class_loss(preds['contact_joint_openpose_style_out'], contact_f_joint_openpose_3d, is_3D)
    contact_style_joint_openpose_2d_loss = class_loss(preds['contact_joint_openpose_style_out'], contact_f_joint_openpose_2d, is_2D)
    style_contact_loss = cfg.TRAIN.contact_loss_weight * (contact_style_mesh_loss + contact_style_joint_loss + contact_style_joint_openpose_loss + contact_style_joint_openpose_2d_loss)

    contact_style_mesh_loss1 = class_loss(preds['contact_style_out1'], contact_f_mesh, is_3D)
    contact_style_joint_loss1 = class_loss(preds['contact_joint_style_out1'], contact_f_joint, is_3D)
    contact_style_joint_openpose_loss1 = class_loss(preds['contact_joint_openpose_style_out1'], contact_f_joint_openpose_3d, is_3D)
    contact_style_joint_openpose_2d_loss1 = class_loss(preds['contact_joint_openpose_style_out1'], contact_f_joint_openpose_2d, is_2D)
    style_contact_loss1 = cfg.TRAIN.contact_loss_weight * (contact_style_mesh_loss1 + contact_style_joint_loss1 + contact_style_joint_openpose_loss1 + contact_style_joint_openpose_2d_loss1)

    contact_style_mesh_loss2 = class_loss(preds['contact_style_out2'], contact_f_mesh, is_3D)
    contact_style_joint_loss2 = class_loss(preds['contact_joint_style_out2'], contact_f_joint, is_3D)
    contact_style_joint_openpose_loss2 = class_loss(preds['contact_joint_openpose_style_out2'], contact_f_joint_openpose_3d, is_3D)
    contact_style_joint_openpose_2d_loss2 = class_loss(preds['contact_joint_openpose_style_out2'], contact_f_joint_openpose_2d, is_2D)
    style_contact_loss2 = cfg.TRAIN.contact_loss_weight * (contact_style_mesh_loss2 + contact_style_joint_loss2 + contact_style_joint_openpose_loss2 + contact_style_joint_openpose_2d_loss2)

    # Adversarial loss (same entropy loss but gradients update early layers)
    contact_adv_mesh_loss = adv_loss(preds['contact_adv_out'], is_3D)
    contact_adv_joint_loss = adv_loss(preds['contact_joint_adv_out'], is_3D)
    contact_adv_joint_openpose_loss = adv_loss(preds['contact_joint_openpose_adv_out'], is_3D)
    contact_adv_joint_openpose_2d_loss = adv_loss(preds['contact_joint_openpose_adv_out'], is_2D)
    adv_contact_loss = cfg.TRAIN.contact_loss_weight * (contact_adv_mesh_loss + contact_adv_joint_loss + contact_adv_joint_openpose_loss + contact_adv_joint_openpose_2d_loss)

    contact_adv_mesh_loss1 = adv_loss(preds['contact_adv_out1'], is_3D)
    contact_adv_joint_loss1 = adv_loss(preds['contact_joint_adv_out1'], is_3D)
    contact_adv_joint_openpose_loss1 = adv_loss(preds['contact_joint_openpose_adv_out1'], is_3D)
    contact_adv_joint_openpose_2d_loss1 = adv_loss(preds['contact_joint_openpose_adv_out1'], is_2D)
    adv_contact_loss1 = cfg.TRAIN.contact_loss_weight * (contact_adv_mesh_loss1 + contact_adv_joint_loss1 + contact_adv_joint_openpose_loss1 + contact_adv_joint_openpose_2d_loss1)

    contact_adv_mesh_loss2 = adv_loss(preds['contact_adv_out2'], is_3D)
    contact_adv_joint_loss2 = adv_loss(preds['contact_joint_adv_out2'], is_3D)
    contact_adv_joint_openpose_loss2 = adv_loss(preds['contact_joint_openpose_adv_out2'], is_3D)
    contact_adv_joint_openpose_2d_loss2 = adv_loss(preds['contact_joint_openpose_adv_out2'], is_2D)
    adv_contact_loss2 = cfg.TRAIN.contact_loss_weight * (contact_adv_mesh_loss2 + contact_adv_joint_loss2 + contact_adv_joint_openpose_loss2 + contact_adv_joint_openpose_2d_loss2)

    # Ground - clean branch
    foot_mask_loss = 0.5 * class_loss(preds['mask_foot_out'], valid_height_mask, is_3D) + 0.5 * dice_loss(preds['mask_foot_out'], valid_height_mask, is_3D)
    pixel_height_loss = norm_loss(preds['pixel_height_out'], pixel_height_map, is_3D)
    ground_normal_loss = cos_loss(preds['ground_normal_out'], ground_normal, is_3D)
    ground_loss = foot_mask_loss + pixel_height_loss + ground_normal_loss

    foot_mask_loss1 = 0.5 * class_loss(preds['mask_foot_out1'], valid_height_mask, is_3D) + 0.5 * dice_loss(preds['mask_foot_out1'], valid_height_mask, is_3D)
    pixel_height_loss1 = norm_loss(preds['pixel_height_out1'], pixel_height_map, is_3D)
    ground_normal_loss1 = cos_loss(preds['ground_normal_out1'], ground_normal, is_3D)
    ground_loss1 = foot_mask_loss1 + pixel_height_loss1 + ground_normal_loss1

    foot_mask_loss2 = 0.5 * class_loss(preds['mask_foot_out2'], valid_height_mask, is_3D) + 0.5 * dice_loss(preds['mask_foot_out2'], valid_height_mask, is_3D)
    pixel_height_loss2 = norm_loss(preds['pixel_height_out2'], pixel_height_map, is_3D)
    ground_normal_loss2 = cos_loss(preds['ground_normal_out2'], ground_normal, is_3D)
    ground_loss2 = foot_mask_loss2 + pixel_height_loss2 + ground_normal_loss2

    # Style-branch ground-aware losses (symmetry with main path)
    pixel_height_style_loss = norm_loss(preds['pixel_height_style_out'], pixel_height_map, is_3D)
    ground_normal_style_loss = cos_loss( preds['ground_normal_style_out'], ground_normal, is_3D)
    style_ground_loss = pixel_height_style_loss + ground_normal_style_loss

    pixel_height_style_loss1 = norm_loss(preds['pixel_height_style_out1'], pixel_height_map, is_3D)
    ground_normal_style_loss1 = cos_loss( preds['ground_normal_style_out1'], ground_normal, is_3D)
    style_ground_loss1 = pixel_height_style_loss1 + ground_normal_style_loss1

    pixel_height_style_loss2 = norm_loss(preds['pixel_height_style_out2'], pixel_height_map, is_3D)
    ground_normal_style_loss2 = cos_loss( preds['ground_normal_style_out2'], ground_normal, is_3D)
    style_ground_loss2 = pixel_height_style_loss2 + ground_normal_style_loss2

    total_foot_mask_loss = (foot_mask_loss + foot_mask_loss1 + foot_mask_loss2) / 3
    total_pixel_height_loss = (pixel_height_loss + pixel_height_loss1 + pixel_height_loss2) / 3
    total_ground_normal_loss = (ground_normal_loss + ground_normal_loss1 + ground_normal_loss2) / 3


    total_main_loss = cfg.TRAIN.contact_loss_weight * (main_contact_loss + main_contact_loss1 + main_contact_loss2)/3
    total_style_loss = cfg.TRAIN.style_contact_loss_weight * (style_contact_loss + style_contact_loss1 + style_contact_loss2)/3
    total_adv_loss = cfg.TRAIN.adv_loss_weight * (adv_contact_loss + adv_contact_loss1 + adv_contact_loss2)/3
    total_ground_loss = cfg.TRAIN.ground_loss_weight * (ground_loss + ground_loss1 + ground_loss2)/3
    total_style_ground_loss = cfg.TRAIN.style_ground_loss_weight * (style_ground_loss + style_ground_loss1 + style_ground_loss2)/3
    total_loss = total_main_loss + total_style_loss + total_adv_loss + total_ground_loss + total_style_ground_loss

    loss_dict = dict(
        total_loss=total_loss,
        main_contact_loss=total_main_loss,
        style_contact_loss=total_style_loss,
        adv_contact_loss=total_adv_loss,
        ground_loss=total_ground_loss,
        style_ground_loss=total_style_ground_loss,
        foot_mask_loss=total_foot_mask_loss,
        pixel_height_loss=total_pixel_height_loss,
        ground_normal_loss=total_ground_normal_loss
    )
    return total_loss, loss_dict