import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import cv2
import json
import trimesh
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lib.core.config import cfg
from lib.utils.transforms import cam2pixel, apply_homogeneous_transformation_np
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.preprocessing import augmentation_contact, apply_augmentation_height, apply_augmentation_mask
from lib.utils.pixel_height_utils import render_foot_depth, compute_image_pixel_height_along_ground_direction
from lib.utils.mesh_utils import compute_ground_normal
from lib.utils.mesh_utils import extract_partial_mesh
from lib.utils.vis_utils import ContactRenderer

contact_renderer = ContactRenderer()


class PROX(Dataset):
    def __init__(self, transform, style_transform, data_split, task=None):
        super(PROX, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.style_transform = style_transform
        self.output_path = Path('data') / 'PROX' / 'preprocessed_data' / data_split
        self.dataset_name = 'prox'
        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'PROX')
        self.data_dir = os.path.join(self.root_path, 'data')
        split_sample_id_file_path = str(Path('data/PROX/preprocessed_data') / data_split / 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]

        self.ground_plane_coeffs = {'slope_x': -0.006376034695736329, 'slope_z': -0.000384573845447686, 'intercept': 1.0347856813163174}


        self.ground_plane_coeffs_world = {'slope_x': 0.0055426240711963694, 'slope_y': 0.02106784185668942, 'intercept': -0.11605666625584592}

        with open('data/base_data/conversions/smplx_vert_segmentation.json', 'r') as f:
            self.smplx_part_seg = json.load(f)
            self.smplx_leftfoot_idxs = self.smplx_part_seg['leftFoot']
            self.smplx_rightfoot_idxs = self.smplx_part_seg['rightFoot']
            self.smplx_lefttoe_idxs = self.smplx_part_seg['leftToeBase']
            self.smplx_righttoe_idxs = self.smplx_part_seg['rightToeBase']
            self.smplx_leftfoot_idxs = np.array(self.smplx_leftfoot_idxs + self.smplx_lefttoe_idxs)
            self.smplx_rightfoot_idxs = np.array(self.smplx_rightfoot_idxs + self.smplx_righttoe_idxs)
        with open(os.path.join(self.data_dir, 'quantitative/calibration/Color.json'), 'r') as f:
            calibration = json.load(f)
            self.cam_param = {'focal': calibration['f'], 'princpt': calibration['c']}


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        seq_name = aid.split('-')[0]
        img_name = aid.split('-')[1]
        sample_id = f'{seq_name}-{img_name}'

        orig_img_path = os.path.join(os.path.join(self.data_dir, 'quantitative', 'recordings', seq_name, 'Color', f'{img_name}.jpg'))
        orig_img = load_img(orig_img_path)
        orig_img = cv2.flip(orig_img, 1)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape
        cam_param = self.cam_param
        with open(os.path.join(self.data_dir, 'quantitative', 'cam2world', f'vicon.json'), 'r') as f:
            c2w_homo = np.array(json.load(f))
            w2c_homo = np.linalg.inv(c2w_homo)

        mesh_body_path = os.path.join(self.data_dir, 'quantitative', 'fittings', 'mosh', seq_name, 'meshes', img_name, '000.ply')
        mesh_body = trimesh.load(mesh_body_path)
        mesh_body_cam = mesh_body.copy()
        mesh_foot_r_cam = mesh_body_cam.vertices[self.smplx_rightfoot_idxs]
        mesh_foot_r_img = cam2pixel(mesh_foot_r_cam, self.cam_param['focal'], self.cam_param['princpt'])
        bbox_foot_r = get_bbox(mesh_foot_r_img, np.ones(len(mesh_foot_r_img)), expansion_factor=cfg.DATASET.foot_scene_bbox_expand_ratio)
        mesh_body_world = apply_homogeneous_transformation_np(mesh_body.vertices, c2w_homo)
        mesh_body = trimesh.Trimesh(mesh_body_world, mesh_body.faces, process=False)
        mesh_foot_r = extract_partial_mesh(mesh_body, self.smplx_rightfoot_idxs)
        mesh_foot_r_cam = extract_partial_mesh(mesh_body_cam, self.smplx_rightfoot_idxs)
        scene_mesh_path = os.path.join(self.data_dir, 'quantitative', 'scenes', f'vicon.ply')
        scene_mesh = trimesh.load(scene_mesh_path)
        ground_plane_slope_x_cam, ground_plane_slope_z_cam, ground_plane_intercept_cam = (self.ground_plane_coeffs['slope_x'], self.ground_plane_coeffs['slope_z'], self.ground_plane_coeffs['intercept'])

        ground_normal = compute_ground_normal(ground_plane_slope_x_cam, ground_plane_slope_z_cam)
        rendered_img, depth_map, valid_mask = render_foot_depth(
            img=orig_img[..., ::-1],
            mesh=mesh_foot_r_cam.vertices,
            face=contact_renderer.mesh_foot_r.faces,
            cam_param=self.cam_param
        )
        pixel_height_map, valid_height_mask = compute_image_pixel_height_along_ground_direction(
            depth=depth_map,
            cam_param=self.cam_param,
            ground_plane_slope_x=ground_plane_slope_x_cam,
            ground_plane_slope_z=ground_plane_slope_z_cam,
            ground_plane_intercept=ground_plane_intercept_cam
        )
        _, _, inv_trans_foot_r, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)
        contact_f, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_foot_r, scene_mesh, cfg.MODEL.c_thres)

        output = Path(self.output_path)
        _, trans, inv_trans, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)
        height = apply_augmentation_height(pixel_height_map, trans, False, cfg.MODEL.input_img_shape)
        valid = apply_augmentation_mask(valid_height_mask, trans, False, cfg.MODEL.input_img_shape)
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
    parser.add_argument('--output-dir', type=Path, default=Path('data/PROX/preprocessed_data'))
    args = parser.parse_args()
    split_root = Path('data/PROX/preprocessed_data')
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
        dataset = PROX(transform, style_transform, manifest.parent.name, task='annotate')
        dataset.output_path = output
        count = len(dataset)
        with torch.no_grad():
            for index in tqdm(range(count), desc='PROX ' + manifest.parent.name):
                dataset[index]
        print('PROX ' + manifest.parent.name + ': ' + str(count) + ' samples complete')