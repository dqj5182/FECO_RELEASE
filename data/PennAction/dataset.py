import os
import numpy as np
import cv2
import glob
import scipy.io as sio
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

import sys
sys.path.append(os.getcwd())

from lib.core.config import cfg
from lib.utils.func_utils import load_img, path_natural_key, get_bbox_pennaction
from lib.utils.preprocessing import augmentation_contact



class PennAction(Dataset):
    def __init__(self, transform, data_split):
        super(PennAction, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.dataset_name = 'pennaction'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'PennAction')
        self.data_dir = os.path.join(self.root_path, 'data')

        split_sample_id_file_path = os.path.join(self.root_path, 'preprocessed_data', data_split, 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding="utf-8") as f:
            self.db = [line.strip() for line in f if line.strip()]


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        video_id = aid.split('-')[0]
        image_id = aid.split('-')[1]
        sample_id = f'{video_id}-{image_id}'

        orig_img_path = os.path.join(self.data_dir, 'frames', video_id, f'{image_id}.jpg')
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        foot_valid = np.ones((1), dtype=np.float32)
        is_3D = np.zeros((1), dtype=np.float32)

        # 2D keypoint
        keypoints_path = os.path.join(self.data_dir, 'keypoints', video_id, f'{image_id}.npy')
        keypoints_2d = np.load(keypoints_path)
        keypoints_2d_body = keypoints_2d[:, :2]
        keypoints_2d_body_valid = keypoints_2d[:, 2]


        # 2D bounding box
        bbox_foot_l, bbox_foot_r = get_bbox_pennaction(keypoints_2d_body, keypoints_2d_body_valid, r_ankle_idx=11, l_ankle_idx=12, foot_frac=0.4, image_size=(img_h, img_w))
        

        # Contact Annotation
        save_contact_r_label_path = os.path.join(self.data_dir, f'contacts', video_id, f'{image_id}_r.npy')
        os.makedirs(os.path.dirname(save_contact_r_label_path), exist_ok=True)


        contact_f = np.zeros(265, dtype=int)
        contact_f_r_joint_openpose_2d = np.load(save_contact_r_label_path).astype(np.float64)
        if self.data_split == 'train':
            contact_data = dict(contact_f=contact_f, contact_f_joint_openpose_2d=contact_f_r_joint_openpose_2d[1:], is_3D=is_3D)
        else:
            contact_data = dict(contact_f=contact_f, contact_f_joint_openpose_2d=contact_f_r_joint_openpose_2d, is_3D=is_3D)


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, self.data_split, enforce_flip=False)
        crop_img = img.copy()

        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform(img)
        ############################### PROCESS CROP AND AUGMENTATION ################################


        crop_pixel_height_map = np.zeros(cfg.MODEL.input_img_shape, dtype=np.int16)
        crop_valid_height_mask = np.zeros(cfg.MODEL.input_img_shape, dtype=np.float64)


        ground_normal = np.zeros(3, dtype=np.float64)
        ground_data = dict(ground_normal=ground_normal)


        if self.data_split == 'train':
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data, ground_data=ground_data, pixel_height_map=crop_pixel_height_map, valid_height_mask=crop_valid_height_mask)
            meta_info = dict(sample_id=sample_id, foot_valid=foot_valid, dataset_name=self.dataset_name)
        else:
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data, ground_data=ground_data, pixel_height_map=crop_pixel_height_map, valid_height_mask=crop_valid_height_mask)
            meta_info = dict(sample_id=sample_id, foot_valid=foot_valid, dataset_name=self.dataset_name)


        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)