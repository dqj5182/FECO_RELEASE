import os
import torch
import numpy as np
from easydict import EasyDict as edict

from lib.core.logger import ColorLogger
from lib.utils.log_utils import init_dirs


cfg = edict()


""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.train_name = ['BEHAVE', 'EgoBody', 'Hi4D', 'InterCap', 'MMVP', 'MotionPro', 'MOYO', 'PROX', 'RICH', 'OpenPose', 'InstaVariety', 'PennAction', 'MPII']
cfg.DATASET.test_name = 'MMVP' # ONLY TEST ONE DATASET AT A TIME
cfg.DATASET.train_shoe_style_name = ['UTZap50K']
cfg.DATASET.test_shoe_style_name = 'UTZap50K'
cfg.DATASET.workers = 4
cfg.DATASET.random_seed = 314
cfg.DATASET.ho_bbox_expand_ratio = 1.3
cfg.DATASET.hand_bbox_expand_ratio = 1.3
cfg.DATASET.body_bbox_expand_ratio = 1.3
cfg.DATASET.ho_big_bbox_expand_ratio = 2.0
cfg.DATASET.foot_bbox_expand_ratio = 2.0
cfg.DATASET.foot_scene_bbox_expand_ratio = 2.5
cfg.DATASET.foot_scene_distorted_bbox_expand_ratio = 3.0 # InterCap has camera distortion, so bigger bbox
cfg.DATASET.foot_motionpro_bbox_expand_ratio = 4.0 # InterCap has camera distortion, so bigger bbox
cfg.DATASET.obj_bbox_expand_ratio = 1.5


""" Model - HMR """
cfg.MODEL = edict()
cfg.MODEL.seed = 314
cfg.MODEL.input_img_shape = (256, 256)
cfg.MODEL.img_mean = (0.485, 0.456, 0.406)
cfg.MODEL.img_std = (0.229, 0.224, 0.225)
cfg.MODEL.img_mean_vit = (0.485, 0.456, 0.406)
cfg.MODEL.img_std_vit = (0.229, 0.224, 0.225)
cfg.MODEL.img_mean_dino = (0.5, 0.5, 0.5)
cfg.MODEL.img_std_dino = (0.5, 0.5, 0.5)
# MANO
cfg.MODEL.human_model_path = 'data/base_data/human_models'
# Contact
cfg.MODEL.c_thres = 0.03
cfg.MODEL.c_thres_ground = 0.1
cfg.MODEL.c_thres_flat_ground = 0.05
cfg.MODEL.c_thres_moyo = 0.01
cfg.MODEL.c_thres_motionpro = 0.02
cfg.MODEL.c_thres_fitted_ground = 0.05
# Backbone
cfg.MODEL.backbone_type = 'vit-h-14'
# Multi-level joint regressor
cfg.MODEL.J_regressor_foot_path = 'data/base_data/foot_data/J_regressor_foot.npy'
cfg.MODEL.J_regressor_foot_openpose_path = 'data/base_data/foot_data/J_regressor_foot_openpose.npy'
# Loss
cfg.MODEL.loss_type = 'ce'


""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.batch = 4
cfg.TRAIN.epoch = 10
cfg.TRAIN.lr = 1e-5
cfg.TRAIN.weight_decay = 0.0001
cfg.TRAIN.milestones = (5, 10)
cfg.TRAIN.gamma = 0.9
cfg.TRAIN.betas = (0.9, 0.95)
cfg.TRAIN.print_freq = 1

cfg.TRAIN.contact_loss_weight = 1.0
cfg.TRAIN.style_contact_loss_weight = 0.1
cfg.TRAIN.adv_loss_weight = 0.1
cfg.TRAIN.ground_loss_weight = 1.0
cfg.TRAIN.style_ground_loss_weight = 0.1


""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch = 1


""" CAMERA """
cfg.CAMERA = edict()

np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)
torch.backends.cudnn.benchmark = True
logger = None


def update_config(backbone_type='', test_name='', exp_dir='', ckpt_path=''):
    if backbone_type == '':
        backbone_type = 'vit-h-14'
    cfg.MODEL.backbone_type = backbone_type
    if test_name == '':
        test_name = 'MMVP'
    cfg.DATASET.test_name = test_name

    global logger
    log_dir = os.path.join(exp_dir, 'log')
    try:
        init_dirs([log_dir])
        logger = ColorLogger(log_dir)
        logger.info("Logger initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize logger: {e}")
        logger = None