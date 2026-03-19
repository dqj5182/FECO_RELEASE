import os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from lib.core.config import cfg
from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact, apply_augmentation_mask_direct_from_crop, apply_augmentation_height_direct_from_crop, apply_augmentation_ground_normal


class Hi4D(Dataset):
    def __init__(self, transform, data_split, task=None):
        super(Hi4D, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.dataset_name = 'hi4d'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'Hi4D')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.aug_split = data_split

        split_sample_id_file_path = os.path.join(self.root_path, 'preprocessed_data', data_split, 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding="utf-8") as f:
            self.db = [line.strip() for line in f if line.strip()]

        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        self.pixel_height_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'pixel_height_data')
        self.ground_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'ground_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)
        os.makedirs(os.path.join(self.pixel_height_data_path, 'map'), exist_ok=True)
        os.makedirs(os.path.join(self.pixel_height_data_path, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(self.ground_data_path, 'normal'), exist_ok=True)


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        pair_name = aid.split('-')[0]
        action_name = aid.split('-')[1]
        cam_name = aid.split('-')[2]
        img_name = aid.split('-')[3]
        pid = aid.split('-')[4]
        sample_id = f'{pair_name}-{action_name}-{cam_name}-{img_name}-{pid}'

        orig_img_path = os.path.join(self.data_dir, pair_name, action_name, 'images', cam_name, f'{img_name}.jpg')
        orig_img = load_img(orig_img_path)

        foot_valid = np.ones((1), dtype=np.float32)
        is_3D = np.ones((1), dtype=np.float32)

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        contact_f_path = os.path.join(self.contact_data_path, f'{sample_id}.npy')
        ground_normal_data_path = os.path.join(self.ground_data_path, 'normal', f'{sample_id}.npy')
        pixel_height_data_path = os.path.join(self.pixel_height_data_path, 'map', f'{sample_id}.npy')
        valid_height_mask_path = os.path.join(self.pixel_height_data_path, 'mask', f'{sample_id}.png')

        annot_data = np.load(annot_data_path, allow_pickle=True)
        bbox_foot_r = annot_data['bbox_foot_r']
        inv_trans_foot_r = annot_data['inv_trans_foot_r']

        # Contact data
        contact_f = np.load(contact_f_path)
        contact_f_joint_openpose_2d = np.zeros(4)
        contact_data = dict(contact_f=contact_f, contact_f_joint_openpose_2d=contact_f_joint_openpose_2d[1:], is_3D=is_3D)

        # Pixel height map
        pixel_height_map = np.load(pixel_height_data_path) # pixel_height_map: (256, 256)
        valid_height_mask = cv2.imread(valid_height_mask_path) # valid_height_mask: (256, 256, 3)
        valid_height_mask = valid_height_mask[:, :, 0]

        # Ground data
        ground_normal = np.load(ground_normal_data_path)
        ground_normal = -1 * ground_normal


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale, do_extreme_crop, extreme_crop_lvl = augmentation_contact(orig_img.copy(), bbox_foot_r, self.aug_split, enforce_flip=False)
        crop_img = img.copy()

        crop_pixel_height_map = apply_augmentation_height_direct_from_crop(height_map_cropped=pixel_height_map, inv_trans_cropped_to_full=inv_trans_foot_r, img2bb_trans_full_to_out=img2bb_trans, do_flip=do_flip, full_image_shape=orig_img.shape[:2], out_shape=tuple(cfg.MODEL.input_img_shape[:2]), apply_extreme_crop=do_extreme_crop, extreme_crop_lvl=extreme_crop_lvl)
        crop_valid_height_mask = apply_augmentation_mask_direct_from_crop(valid_height_mask, inv_trans_cropped_to_full=inv_trans_foot_r, img2bb_trans_full_to_out=img2bb_trans, do_flip=do_flip, full_image_shape=orig_img.shape[:2], out_shape=tuple(cfg.MODEL.input_img_shape[:2]), apply_extreme_crop=do_extreme_crop, extreme_crop_lvl=extreme_crop_lvl)
        ground_normal = apply_augmentation_ground_normal(ground_normal, rot_deg=rot, do_flip=do_flip)
        ground_data = dict(ground_normal=ground_normal)

        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform(img)
        ############################### PROCESS CROP AND AUGMENTATION ################################


        if self.data_split == 'train':
            crop_valid_height_mask = 1. * (crop_valid_height_mask > 128)
        else:
            crop_valid_height_mask = 1. * (crop_valid_height_mask > 0.5)


        if self.data_split == 'train':
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data, ground_data=ground_data, pixel_height_map=crop_pixel_height_map, valid_height_mask=crop_valid_height_mask)
            meta_info = dict(sample_id=sample_id, foot_valid=foot_valid, dataset_name=self.dataset_name)
        else:
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data, ground_data=ground_data, pixel_height_map=crop_pixel_height_map, valid_height_mask=crop_valid_height_mask)
            meta_info = dict(sample_id=sample_id, foot_valid=foot_valid, dataset_name=self.dataset_name)


        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)