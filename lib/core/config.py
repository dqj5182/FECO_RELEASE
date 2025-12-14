import os
import torch
import numpy as np
from easydict import EasyDict as edict

from lib.core.logger import ColorLogger
from lib.utils.log_utils import init_dirs


cfg = edict()


""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.workers = 4
cfg.DATASET.random_seed = 314


""" Model - HMR """
cfg.MODEL = edict()
cfg.MODEL.seed = 314
cfg.MODEL.input_img_shape = (256, 256)
cfg.MODEL.img_mean = (0.485, 0.456, 0.406)
cfg.MODEL.img_std = (0.229, 0.224, 0.225)
cfg.MODEL.img_mean_vit = (0.485, 0.456, 0.406)
cfg.MODEL.img_std_vit = (0.229, 0.224, 0.225)
# Human model
cfg.MODEL.human_model_path = 'data/base_data/human_models'
# Contact
cfg.MODEL.contact_data_path = 'data/base_data/contact_data/all/contact_data_all.npy'
cfg.MODEL.contact_means_path = 'data/base_data/contact_data/all/contact_means_all.npy'
# Backbone
cfg.MODEL.backbone_type = 'vit-h-14'
# Multi-level joint regressor
cfg.MODEL.J_regressor_foot_path = 'data/base_data/foot_data/J_regressor_foot.npy'
cfg.MODEL.J_regressor_foot_openpose_path = 'data/base_data/foot_data/J_regressor_foot_openpose.npy'
# Loss
cfg.MODEL.loss_type = 'ce'


""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch = 1


""" CAMERA """
cfg.CAMERA = edict()

np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)
torch.backends.cudnn.benchmark = True
logger = None


def update_config(backbone_type='', exp_dir='', ckpt_path=''):
    if backbone_type == '':
        backbone_type = 'hamer'
    cfg.MODEL.backbone_type = backbone_type

    global logger
    log_dir = os.path.join(exp_dir, 'log')
    try:
        init_dirs([log_dir])
        logger = ColorLogger(log_dir)
        logger.info("Logger initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize logger: {e}")
        logger = None