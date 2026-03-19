import os
import copy
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from torch.utils.data import Dataset

from lib.core.config import cfg
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.preprocessing import augmentation_contact


def get_filtered_coco(db, data_split):
    kept_annotations = []

    for ann_id, ann in db.anns.items():
        if ann.get('iscrowd', 0):
            continue
        if 'keypoints' not in ann or ann['keypoints'] is None:
            continue

        kpts = np.array(ann['keypoints'], dtype=float).reshape(-1, 3)
        kpts_xy = kpts[:, :2]

        if data_split == 'train':
            feet = kpts_xy[15:]
            k2d_foot_l = feet[[0, 2, 3, 4]]
            k2d_foot_r = feet[[1, 5, 6, 7]]
        else:
            feet = kpts_xy
            k2d_foot_l = feet[:3]
            k2d_foot_r = feet[3:]

        valid_l = (~np.all(k2d_foot_l == 0, axis=1)).astype(int)
        valid_r = (~np.all(k2d_foot_r == 0, axis=1)).astype(int)

        if valid_l.sum() <= 1 or valid_r.sum() <= 1:
            continue

        kept_annotations.append(ann)

    if len(kept_annotations) == 0:
        filtered_db = COCO()
        filtered_db.dataset = {
            "images": [],
            "annotations": [],
            "categories": copy.deepcopy(db.dataset.get('categories', [])),
        }
        filtered_db.createIndex()
        return filtered_db

    kept_image_ids = {ann['image_id'] for ann in kept_annotations}
    kept_images = [img for img in db.dataset['images'] if img['id'] in kept_image_ids]

    filtered_data = {
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": copy.deepcopy(db.dataset.get('categories', [])),
    }

    filtered_db = COCO()
    filtered_db.dataset = filtered_data
    filtered_db.createIndex()
    return filtered_db


class OpenPose(Dataset):
    def __init__(self, transform, data_split):
        super(OpenPose, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.dataset_name = 'openpose'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'OpenPose')
        self.data_dir = os.path.join(self.root_path, 'data')

        split_sample_id_file_path = os.path.join(self.root_path, 'preprocessed_data', data_split, 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding="utf-8") as f:
            self.db = [line.strip() for line in f if line.strip()]

        if data_split == 'train':
            self.annot_split = 'train2017'
        else:
            self.annot_split = 'val2017'
        ann = COCO(os.path.join(self.data_dir, f'person_keypoints_{self.annot_split}_foot_v1.json'))
        self.ann = get_filtered_coco(ann, data_split)

        # allowed ids from self.db
        allowed_image_ids = set()
        for sid in self.db:
            head = sid.split('_')[0]
            try:
                allowed_image_ids.add(int(head))
            except ValueError:
                pass
        
        # keep only ann ids whose image_id is in the list
        self.kept_ann_ids = [aid for aid, a in self.ann.anns.items() if a['image_id'] in allowed_image_ids]


    def __len__(self):
        return len(self.kept_ann_ids)


    def __getitem__(self, index):
        aid = self.kept_ann_ids[index]
        ann = self.ann.anns[aid]
        image_id = ann['image_id']
        img = self.ann.loadImgs(image_id)[0]
        sample_id = str(image_id)

        orig_img_path = os.path.join(self.data_dir, self.annot_split, img['file_name'])
        orig_img = load_img(orig_img_path)

        foot_valid = np.ones((1), dtype=np.float32)
        is_3D = np.zeros((1), dtype=np.float32)

        # 2D keypoint
        keypoints_2d_body = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
        
        if self.data_split == 'train':
            keypoints_2d_feet = keypoints_2d_body[15:]
            keypoints_2d_foot_r = keypoints_2d_feet[[1, 5, 6, 7]]
        else:
            keypoints_2d_feet = keypoints_2d_body
            keypoints_2d_foot_r = keypoints_2d_feet[3:]

        keypoints_2d_foot_valid_r = (~np.all(keypoints_2d_foot_r == 0, axis=1)).astype(int)

        # 2D bounding box
        bbox_foot_r = get_bbox(keypoints_2d_foot_r[keypoints_2d_foot_valid_r == 1], np.ones(keypoints_2d_foot_valid_r.sum()), expansion_factor=cfg.DATASET.foot_bbox_expand_ratio)

        # Contact Annotation
        save_contact_r_label_path = os.path.join(self.data_dir, f'{self.annot_split}_contact', f'{sample_id}_r.npy')

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