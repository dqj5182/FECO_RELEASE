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
from lib.utils.mesh_utils import (compute_ground_normal, create_ground_plane_mesh_xy,
                                  extract_ground_plane, extract_partial_mesh)
from lib.utils.moyo_utils import smplx_to_mesh
from lib.utils.pixel_height_utils import (compute_image_pixel_height_along_ground_direction,
                                          render_foot_depth)
from lib.utils.preprocessing import (apply_augmentation_height, apply_augmentation_mask,
                                     augmentation_contact)
from lib.utils.transforms import cam2pixel
from lib.utils.vis_utils import ContactRenderer


VICON_FPS = 60
IOI_FPS = 30
CALIBRATION_SESSION = {'220923': 'Afternoon', '220926': 'Morning', '221004': ''}


class MOYO(Dataset):
    def __init__(self, data_split, output_dir):
        self.root_path = Path('data/MOYO')
        self.data_dir = self.root_path / 'data'
        self.output_path = Path(output_dir) / data_split
        self.annotation_split = 'train' if data_split == 'train' else 'val'
        with (self.root_path / 'preprocessed_data' / data_split / 'split_sample_ids.txt').open(encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]
        with open('data/base_data/conversions/smplx_vert_segmentation.json') as f:
            parts = json.load(f)
        self.rightfoot_idxs = np.array(parts['rightFoot'] + parts['rightToeBase'])
        self.renderer = ContactRenderer()
        self.cached_sequence = None
        self.cached_vertices = None
        self.cached_faces = None

    def __len__(self):
        return len(self.db)

    def _body_mesh(self, sequence, frame_index):
        if sequence != self.cached_sequence:
            path = self.data_dir / 'mosh' / self.annotation_split / f'{sequence}_stageii.pkl'
            with path.open('rb') as f:
                mosh_data = pickle.load(f)
            body_output, _, _, faces = smplx_to_mesh(
                mosh_data, 'data/base_data/human_models', 'smplx', gender='neutral')
            self.cached_vertices = body_output.vertices.detach().cpu().numpy()
            self.cached_faces = faces
            self.cached_sequence = sequence
        if frame_index >= len(self.cached_vertices):
            raise IndexError(f'MOYO motion frame {frame_index} unavailable for {sequence}')
        return trimesh.Trimesh(self.cached_vertices[frame_index], self.cached_faces, process=False)

    def __getitem__(self, index):
        sample_id = self.db[index]
        fields = sample_id.split('-')
        if len(fields) == 4:
            seq_count, camera_index = 2, 2
        elif len(fields) == 7:
            seq_count, camera_index = 3, 3
        elif len(fields) == 9:
            seq_count, camera_index = 4, 4
        elif len(fields) == 11:
            seq_count, camera_index = 5, 5
        else:
            seq_count, camera_index = 2, 2
        sequence = '-'.join(fields[:seq_count])
        camera_name = fields[camera_index]
        image_name = '-'.join(fields[camera_index + 1:])
        session = sequence.split('_')[0]
        camera_id = 'cam_' + str(int(camera_name.split('_')[-1]))
        image_frame = int(image_name.split('_')[-1])
        motion_frame = image_frame * (VICON_FPS // IOI_FPS)

        image = load_img(str(self.data_dir / sequence / camera_name / f'{image_name}.jpg'))
        calibration_file = (self.data_dir / 'cameras' / f'20{session}' /
                            f'{session}_{CALIBRATION_SESSION[session]}_PROCESSED_CAMERA_PARAMS' /
                            'cameras_param.json')
        with calibration_file.open() as f:
            calibration = json.load(f)[camera_id]
        cam_param = {'focal': [calibration['focal'] / 2, calibration['focal'] / 2],
                     'princpt': [calibration['princpt'][0] / 2, calibration['princpt'][1] / 2]}
        rotation = np.array(calibration['rotation'])
        translation = (-1 / 1000) * (rotation @ np.array(calibration['position']).reshape(3, 1))[:, 0]

        body_world = self._body_mesh(sequence, motion_frame)
        body_cam = body_world.copy()
        body_cam.vertices = body_cam.vertices @ rotation.T + translation
        foot_cam = extract_partial_mesh(body_cam, self.rightfoot_idxs)
        foot_pixels = cam2pixel(foot_cam.vertices, cam_param['focal'], cam_param['princpt'])
        bbox = get_bbox(foot_pixels, np.ones(len(foot_pixels)),
                        expansion_factor=cfg.DATASET.foot_scene_bbox_expand_ratio)
        _, trans, inv_trans, _, _, _, _, _ = augmentation_contact(
            image.copy(), bbox, data_split='test', enforce_flip=False)
        contact = (body_world.vertices[:, 2] < cfg.MODEL.c_thres_moyo).astype(int)[self.rightfoot_idxs]

        center = (body_world.vertices[:, 0].mean(), body_world.vertices[:, 1].mean(), 0.0)
        ground_mesh = create_ground_plane_mesh_xy(0.0, 0.0, 0.0,
                                                  center=center, size=(5.0, 5.0), resolution=50)
        ground_mesh.vertices = (rotation @ ground_mesh.vertices.T).T + translation
        camera_ground = extract_ground_plane(ground_mesh)
        ground_normal = compute_ground_normal(*camera_ground[:2])
        _, depth, _ = render_foot_depth(img=image[..., ::-1], mesh=foot_cam.vertices,
                                        face=self.renderer.mesh_foot_r.faces, cam_param=cam_param)
        height_map, valid_mask = compute_image_pixel_height_along_ground_direction(
            depth=depth, cam_param=cam_param, ground_plane_slope_x=camera_ground[0],
            ground_plane_slope_z=camera_ground[1], ground_plane_intercept=camera_ground[2])
        height = apply_augmentation_height(height_map, trans, False, cfg.MODEL.input_img_shape)
        valid = apply_augmentation_mask(valid_mask, trans, False, cfg.MODEL.input_img_shape)

        output = self.output_path
        np.savez(output / 'annot_data' / f'{sample_id}.npz', sample_id=sample_id,
                 cam_param=cam_param, bbox_foot_r=bbox, inv_trans_foot_r=inv_trans)
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
    parser.add_argument('--output-dir', type=Path, default=Path('data/MOYO/preprocessed_data'))
    parser.add_argument('--limit', type=int, help='Process only the first N samples for validation')
    args = parser.parse_args()
    source = Path('data/MOYO/preprocessed_data')
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
        dataset = MOYO(split, args.output_dir)
        count = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
        with torch.no_grad():
            for index in tqdm(range(count), desc=f'MOYO {split}'):
                dataset[index]