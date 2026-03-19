import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact


def split_zap50k_by_category(image_paths, split='train', test_category_ratio=0.1, seed=12):
    assert split in ['train', 'test']
    random.seed(seed)

    # Group images by category path (3-level directory structure)
    category_to_images = defaultdict(list)
    for path in image_paths:
        category_key = '/'.join(path.split(os.sep)[-5:-1])
        category_to_images[category_key].append(path)

    # Shuffle categories and split
    categories = sorted(category_to_images)
    random.shuffle(categories)
    split_point = int(len(categories) * (1 - test_category_ratio))
    selected_categories = categories[:split_point] if split == 'train' else categories[split_point:]

    return [img for cat in selected_categories for img in category_to_images[cat]]


class UTZap50K(Dataset):
    def __init__(self, transform, data_split):
        super(UTZap50K, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'utzap50k'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'UTZap50K')
        self.data_dir = os.path.join(self.root_path, 'data')

        meta_data_path = os.path.join(self.data_dir, 'ut-zap50k-data', 'meta-data-bin.csv')
        self.meta_data = pd.read_csv(meta_data_path)
        valid_cids = set(self.meta_data['CID'].values)

        db = glob.glob(os.path.join(self.data_dir, 'ut-zap50k-images', '*', '*', '*', '*.jpg'))
        self.db = []
        for each_db in db:
            img_name = each_db.split('/')[-1].split('.jpg')[0]
            image_id = img_name.replace('.', '-')
            if image_id not in valid_cids:
                continue
            self.db.append(each_db)

        self.db = split_zap50k_by_category(self.db, split=data_split)

        all_columns = [col for col in self.meta_data.columns if col != 'CID']
        prefix_groups = ['Category', 'SubCategory', 'HeelHeight', 'Insole', 'Closure', 'Gender', 'Material', 'ToeStyle']

        self.hierarchy_labels = {
            group: sorted([col for col in all_columns if col.startswith(f"{group}.")])
            for group in prefix_groups
        }


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        category_name1 = aid.split('/')[-4]
        category_name2 = aid.split('/')[-3]
        category_name3 = aid.split('/')[-2]
        img_name = aid.split('/')[-1].split('.jpg')[0]
        image_id = img_name.replace('.', '-')
        sample_id = f'{category_name1}-{category_name2}-{category_name3}-{img_name}'

        orig_img_path = aid
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        # Extract metadata row for the given image ID
        meta_row_df = self.meta_data[self.meta_data['CID'] == image_id]

        # Remove CID column to isolate label values
        label_values_df = meta_row_df.drop(columns=['CID'])

        class_index_list = []
        group_validity_flags = []
        attribute_groups = list(self.hierarchy_labels.keys())

        # Process each label group
        for group_name in attribute_groups:
            group_columns = self.hierarchy_labels[group_name]
            group_one_hot = label_values_df[group_columns].astype(np.float32).values

            is_group_annotated = group_one_hot.sum() > 0
            group_validity_flags.append(int(is_group_annotated))

            if is_group_annotated:
                class_index = group_one_hot.argmax(axis=1).item()
            else:
                class_index = -1

            class_index_list.append(class_index)

        target_class_indices = torch.tensor(class_index_list, dtype=torch.long)
        target_validity_mask = torch.tensor(group_validity_flags, dtype=torch.long)

        bbox_foot_r = [0, 0, img_w, img_h]


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale, do_extreme_crop, extreme_crop_lvl = augmentation_contact(orig_img.copy(), bbox_foot_r, self.data_split, enforce_flip=False)
        crop_img = img.copy()

        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform(img)
        ############################### PROCESS CROP AND AUGMENTATION ################################


        if self.data_split == 'train':
            input_data = dict(image=img)
            targets_data = dict(class_label=target_class_indices, class_valid=target_validity_mask)
            meta_info = dict(sample_id=sample_id)
        else:
            input_data = dict(image=img)
            targets_data = dict(class_label=target_class_indices, class_valid=target_validity_mask)
            meta_info = dict(sample_id=sample_id)


        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)