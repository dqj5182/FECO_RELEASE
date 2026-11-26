import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import cv2
import json
import pickle
import trimesh
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lib.core.config import cfg
from lib.utils.human_models import smpl_x
from lib.utils.transforms import cam2pixel
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.preprocessing import augmentation_contact
from lib.utils.pixel_height_utils import render_foot_depth, compute_image_pixel_height_along_ground_direction
from lib.utils.mesh_utils import compute_ground_normal
from lib.utils.mesh_utils import extract_partial_mesh
from lib.utils.vis_utils import ContactRenderer

contact_renderer = ContactRenderer()


class MMVP(Dataset):
    def __init__(self, transform, style_transform, data_split, task=None):
        super(MMVP, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.style_transform = style_transform
        self.output_path = Path('data') / 'MMVP' / 'preprocessed_data' / data_split
        self.dataset_name = 'mmvp'
        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'MMVP')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.annotation_dir = os.path.join(self.data_dir, 'annotations')
        self.contact_dir = os.path.join(self.data_dir, 'contacts')
        self.mesh_body_dir = os.path.join(self.data_dir, 'meshes')
        split_sample_id_file_path = str(Path('data/MMVP/preprocessed_data') / data_split / 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]


        self.ground_plane_coeffs = {
                                    'S01': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.25330174},
                                    'S02': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.24544096},
                                    'S03': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.24262941},
                                    'S04': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.24720883},
                                    'S05': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.21958303},
                                    'S06': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.25487947},
                                    'S07': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.25587881},
                                    'S09': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.23999989},
                                    'S10': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.2669698},
                                    'S11': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.2237581},
                                    'S12': {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 1.24692881},
                                    }
        smpl_to_smplx_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smpl_to_smplx.pkl')
        with open(smpl_to_smplx_mapping_path, 'rb') as f:
            self.smpl_to_smplx_mapping = pickle.load(f)


        with open('data/base_data/conversions/smplx_vert_segmentation.json', 'r') as f:
            self.smplx_part_seg = json.load(f)
            self.smplx_leftfoot_idxs = self.smplx_part_seg['leftFoot']
            self.smplx_rightfoot_idxs = self.smplx_part_seg['rightFoot']
            self.smplx_lefttoe_idxs = self.smplx_part_seg['leftToeBase']
            self.smplx_righttoe_idxs = self.smplx_part_seg['rightToeBase']
            self.smplx_leftfoot_idxs = np.array(self.smplx_leftfoot_idxs + self.smplx_lefttoe_idxs)
            self.smplx_rightfoot_idxs = np.array(self.smplx_rightfoot_idxs + self.smplx_righttoe_idxs)


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        date_name = aid.split('-')[0]
        subject_id = aid.split('-')[1]
        seq_name = aid.split('-')[2]
        frame_id = aid.split('-')[3]
        sample_id = f'{date_name}-{subject_id}-{seq_name}-{frame_id}'


        orig_img_path = os.path.join(self.data_dir, 'images', date_name, subject_id, seq_name, 'color', f'{frame_id}.jpg')
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape


        calibration_path = os.path.join(self.image_dir, date_name, subject_id, seq_name, 'calibration.npy')
        calibration = np.load(calibration_path, allow_pickle=True).item()['color_Intr']
        cam_param = {'focal': [calibration['fx'], calibration['fy']], 'princpt': [calibration['cx'], calibration['cy']]}


        mesh_body_path = os.path.join(self.mesh_body_dir, date_name, subject_id, seq_name, f'{frame_id}.obj')
        mesh_body_cam = trimesh.load(mesh_body_path)
        mesh_smplx_cam = trimesh.Trimesh(np.matmul(self.smpl_to_smplx_mapping['matrix'], mesh_body_cam.vertices), smpl_x.face, process=False)
        mesh_foot_r_cam = extract_partial_mesh(mesh_smplx_cam, self.smplx_rightfoot_idxs)
        mesh_foot_r_img = cam2pixel(mesh_foot_r_cam.vertices, cam_param['focal'], cam_param['princpt'])
        bbox_foot_r = get_bbox(mesh_foot_r_img, np.ones(len(mesh_foot_r_img)), expansion_factor=cfg.DATASET.foot_scene_bbox_expand_ratio)
        _, _, inv_trans_foot_r, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)
        ground_plane_slope_x_cam, ground_plane_slope_z_cam, ground_plane_intercept_cam = (self.ground_plane_coeffs[subject_id]['slope_x'], self.ground_plane_coeffs[subject_id]['slope_z'], self.ground_plane_coeffs[subject_id]['intercept'])


        ground_normal = compute_ground_normal(ground_plane_slope_x_cam, ground_plane_slope_z_cam)


        contact_path = os.path.join(self.contact_dir, date_name, subject_id, seq_name, f'{frame_id}.npy')
        contact_smpl = np.load(contact_path)
        contact_smplx = 1 * (np.matmul(self.smpl_to_smplx_mapping['matrix'], contact_smpl) > 0.5)
        contact_f = contact_smplx[self.smplx_rightfoot_idxs]
        rendered_img, depth_map, valid_mask = render_foot_depth(
            img=orig_img[..., ::-1],
            mesh=mesh_foot_r_cam.vertices,
            face=contact_renderer.mesh_foot_r.faces,
            cam_param=cam_param
        )
        pixel_height_map, valid_height_mask = compute_image_pixel_height_along_ground_direction(
            depth=depth_map,
            cam_param=cam_param,
            ground_plane_slope_x=ground_plane_slope_x_cam,
            ground_plane_slope_z=ground_plane_slope_z_cam,
            ground_plane_intercept=ground_plane_intercept_cam
        )


        output = Path(self.output_path)
        _, trans, inv_trans, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)
        height_3ch = np.stack([pixel_height_map] * 3, axis=-1)
        height_3ch, _, _, _, _, _, _, _ = augmentation_contact(height_3ch, bbox_foot_r, data_split='test', enforce_flip=False)
        height = height_3ch[:, :, 0]
        mask_3ch = np.stack([valid_height_mask] * 3, axis=-1)
        mask_3ch, _, _, _, _, _, _, _ = augmentation_contact(mask_3ch, bbox_foot_r, data_split='test', enforce_flip=False)
        valid = mask_3ch[:, :, 0]
        annot = dict(sample_id=sample_id, cam_param=cam_param, bbox_foot_r=bbox_foot_r, inv_trans_foot_r=inv_trans)


        np.savez(output / 'annot_data' / (sample_id + '.npz'), **annot)
        np.save(output / 'contact_data' / (sample_id + '.npy'), np.asarray(contact_f, dtype=int))
        np.save(output / 'ground_data/normal' / (sample_id + '.npy'), ground_normal)
        np.save(output / 'pixel_height_data/map' / (sample_id + '.npy'), np.round(height).astype(np.int16))
        if not cv2.imwrite(str(output / 'pixel_height_data/mask' / (sample_id + '.png')), (valid * 255).astype(np.uint8)):
            raise OSError('Could not write mask for ' + sample_id)
        return sample_id


if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test'])
    parser.add_argument('--output-dir', type=Path, default=Path('data/MMVP/preprocessed_data'))
    args = parser.parse_args()
    split_root = Path('data/MMVP/preprocessed_data')
    manifests = [split_root / args.split / 'split_sample_ids.txt'] if args.split else sorted(split_root.glob('*/split_sample_ids.txt'))
    if not manifests:
        raise FileNotFoundError(f'No split_sample_ids.txt found under {split_root}')
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f'Missing split manifest: {manifest}')
        if manifest.parent.name not in ('train', 'test'):
            continue
        output = args.output_dir / manifest.parent.name
        output.mkdir(parents=True, exist_ok=True)
        if manifest.resolve() != (output / 'split_sample_ids.txt').resolve():
            shutil.copyfile(manifest, output / 'split_sample_ids.txt')
        for folder in ['annot_data', 'contact_data', 'ground_data/normal', 'pixel_height_data/map', 'pixel_height_data/mask']:
            (output / folder).mkdir(parents=True, exist_ok=True)
        transform = transforms.ToTensor()
        style_transform = transforms.ToTensor()
        dataset = MMVP(transform, style_transform, manifest.parent.name, task='annotate')
        dataset.output_path = output
        count = len(dataset)
        with torch.no_grad():
            for index in tqdm(range(count), desc='MMVP ' + manifest.parent.name):
                dataset[index]
        print('MMVP ' + manifest.parent.name + ': ' + str(count) + ' samples complete')