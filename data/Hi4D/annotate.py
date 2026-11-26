import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import json
import pickle
import shutil

import cv2
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.core.config import cfg
from lib.utils.func_utils import get_bbox, load_img
from lib.utils.human_models import smpl
from lib.utils.mesh_utils import (compute_ground_normal, compute_signed_distances_to_plane,
                                  create_ground_plane_mesh, extract_ground_plane)
from lib.utils.pixel_height_utils import (compute_image_pixel_height_along_ground_direction,
                                          render_foot_depth)
from lib.utils.preprocessing import (apply_augmentation_height, apply_augmentation_mask,
                                     augmentation_contact)
from lib.utils.transforms import cam2pixel
from lib.utils.vis_utils import ContactRenderer


class Hi4D(Dataset):
    def __init__(self, data_split, output_dir):
        self.root_path = Path('data/Hi4D')
        self.data_dir = self.root_path / 'data'
        self.output_path = Path(output_dir) / data_split
        manifest = self.root_path / 'preprocessed_data' / data_split / 'split_sample_ids.txt'
        with manifest.open(encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]
        with open('data/base_data/conversions/smpl_to_smplx.pkl', 'rb') as f:
            self.smpl_to_smplx_mapping = pickle.load(f)['matrix']
        with open('data/base_data/conversions/smplx_vert_segmentation.json') as f:
            parts = json.load(f)
        self.rightfoot_idxs = np.array(parts['rightFoot'] + parts['rightToeBase'])
        self.ground_plane_coeffs = (-0.010825328882025708, 0.011382879201886676, 0.2741792256433256)
        self.renderer = ContactRenderer()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        sample_id = self.db[index]
        pair, action, camera, frame, person = sample_id.split('-')
        sequence = self.data_dir / pair / action
        image = load_img(str(sequence / 'images' / camera / f'{frame}.jpg'))
        camera_data = np.load(sequence / 'cameras/rgb_cameras.npz')
        camera_index = camera_data['ids'].tolist().index(int(camera))
        extrinsic = camera_data['extrinsics'][camera_index]
        rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
        intrinsic = camera_data['intrinsics'][camera_index]
        cam_param = {'focal': [intrinsic[0, 0], intrinsic[1, 1]],
                     'princpt': [intrinsic[0, 2], intrinsic[1, 2]]}

        smpl_data = np.load(sequence / 'smpl' / f'{frame}.npz')
        vertices = smpl_data['verts'][int(person)]
        body = trimesh.Trimesh(vertices, smpl.face, process=False)
        slope_x, slope_z, intercept = self.ground_plane_coeffs
        distances = compute_signed_distances_to_plane(vertices, slope_x, slope_z, intercept)
        ground_contact = (np.abs(distances) < cfg.MODEL.c_thres_fitted_ground).astype(np.uint8)
        ground_contact = self.smpl_to_smplx_mapping @ ground_contact
        ground_contact = ground_contact[self.rightfoot_idxs] > 0.5
        body_contact = (smpl_data['contact'][int(person)] > 0).astype(int)
        body_contact = (self.smpl_to_smplx_mapping @ body_contact)[self.rightfoot_idxs]
        contact = np.logical_or(ground_contact, body_contact).astype(int)

        center = (vertices[:, 0].mean(), 0.0, vertices[:, 2].mean())
        ground_mesh = create_ground_plane_mesh(slope_x, slope_z, intercept,
                                               center=center, size=(5.0, 5.0), resolution=50)
        ground_mesh.vertices = (rotation @ ground_mesh.vertices.T).T + translation
        camera_ground = extract_ground_plane(ground_mesh)
        ground_normal = compute_ground_normal(*camera_ground[:2])

        camera_vertices = (rotation @ vertices.T).T + translation
        smplx_vertices = self.smpl_to_smplx_mapping @ camera_vertices
        foot_vertices = smplx_vertices[self.rightfoot_idxs]
        foot_pixels = cam2pixel(foot_vertices, cam_param['focal'], cam_param['princpt'])
        bbox = get_bbox(foot_pixels, np.ones(len(foot_pixels)),
                        expansion_factor=cfg.DATASET.foot_scene_bbox_expand_ratio)
        _, trans, inv_trans, _, _, _, _, _ = augmentation_contact(
            image.copy(), bbox, data_split='test', enforce_flip=False)
        _, depth, _ = render_foot_depth(img=image[..., ::-1], mesh=foot_vertices,
                                        face=self.renderer.mesh_foot_r.faces, cam_param=cam_param)
        height_map, valid_mask = compute_image_pixel_height_along_ground_direction(
            depth=depth, cam_param=cam_param, ground_plane_slope_x=camera_ground[0],
            ground_plane_slope_z=camera_ground[1], ground_plane_intercept=camera_ground[2],
            positive_y_up=True)
        height = apply_augmentation_height(height_map, trans, False, cfg.MODEL.input_img_shape)
        valid = apply_augmentation_mask(valid_mask, trans, False, cfg.MODEL.input_img_shape)

        output = self.output_path
        np.savez(output / 'annot_data' / f'{sample_id}.npz', sample_id=sample_id,
                 contact_f=contact, bbox_foot_r=bbox, cam_param=cam_param,
                 inv_trans_foot_r=inv_trans)
        np.save(output / 'contact_data' / f'{sample_id}.npy', contact)
        np.save(output / 'ground_data/normal' / f'{sample_id}.npy', ground_normal)
        np.save(output / 'pixel_height_data/map' / f'{sample_id}.npy',
                np.round(height).astype(np.int16))
        if not cv2.imwrite(str(output / 'pixel_height_data/mask' / f'{sample_id}.png'),
                           (valid * 255).astype(np.uint8)):
            raise OSError(f'Could not write mask for {sample_id}')
        return sample_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test'])
    parser.add_argument('--output-dir', type=Path, default=Path('data/Hi4D/preprocessed_data'))
    parser.add_argument('--limit', type=int, help='Process only the first N samples for validation')
    args = parser.parse_args()
    source = Path('data/Hi4D/preprocessed_data')
    splits = [args.split] if args.split else ['train', 'test']
    for split in splits:
        output = args.output_dir / split
        output.mkdir(parents=True, exist_ok=True)
        manifest = source / split / 'split_sample_ids.txt'
        if manifest.resolve() != (output / 'split_sample_ids.txt').resolve():
            shutil.copyfile(manifest, output / 'split_sample_ids.txt')
        for folder in ('annot_data', 'contact_data', 'ground_data/normal',
                       'pixel_height_data/map', 'pixel_height_data/mask'):
            (output / folder).mkdir(parents=True, exist_ok=True)
        dataset = Hi4D(split, args.output_dir)
        count = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
        with torch.no_grad():
            for index in tqdm(range(count), desc=f'Hi4D {split}'):
                dataset[index]