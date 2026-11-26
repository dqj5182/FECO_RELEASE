import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import json
import numpy as np
import trimesh
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lib.core.config import cfg
from lib.utils.transforms import cam2pixel
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.preprocessing import augmentation_contact, apply_augmentation_height, apply_augmentation_mask
from lib.utils.mesh_utils import extract_partial_mesh, create_ground_plane_mesh, compute_signed_distances_to_plane, extract_ground_plane, compute_ground_normal
from lib.utils.pixel_height_utils import render_foot_depth, compute_image_pixel_height_along_ground_direction
from lib.utils.vis_utils import ContactRenderer

contact_renderer = ContactRenderer()


class InterCap(Dataset):
    def __init__(self, transform, style_transform, data_split, task=None):
        super(InterCap, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.style_transform = style_transform
        self.dataset_name = 'intercap'
        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'InterCap')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.annotation_dir = os.path.join(self.data_dir, 'Res')
        self.calibration_path = os.path.join(self.root_path, 'calibration')
        self.output_path = Path(root_path) / 'preprocessed_data' / data_split

        split_sample_id_file_path = os.path.join(self.root_path, 'preprocessed_data', data_split, 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]

        self.calibration = {}
        for cam_id in range(1, 7):
            filename = 'Color.json' if cam_id == 1 else f'Color_{cam_id}.json'
            with open(os.path.join(self.calibration_path, filename), 'r') as f:
                camera = json.load(f)
            rotation, _ = cv2.Rodrigues(np.array(camera['R']))
            self.calibration[cam_id] = dict(intrinsic=camera, rotation=rotation, translation=np.array(camera['T']).reshape(1, 3))

        self.ground_plane_coeffs = {'slope_x': 0.027502683551041015, 'slope_z': -0.1343937121149727, 'intercept': 1.4542306532646525}
        with open('data/base_data/conversions/smplx_vert_segmentation.json', 'r') as f:
            part_seg = json.load(f)
        self.smplx_rightfoot_idxs = np.array(part_seg['rightFoot'] + part_seg['rightToeBase'])


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        subject_name, seq_name, sub_seq_name, cam_name, frame_id = aid.split('-')
        sample_id = f'{subject_name}-{seq_name}-{sub_seq_name}-{cam_name}-{frame_id}'
        cam_id = int(cam_name.split('Frames_Cam')[-1])

        orig_img_path = os.path.join(self.data_dir, 'RGBD_Images', subject_name, seq_name, sub_seq_name, cam_name, 'color', f'{frame_id}.jpg')
        orig_img = load_img(orig_img_path)

        camera = self.calibration[cam_id]
        K = np.array(camera['intrinsic']['camera_mtx'])
        cam_param = {'focal': [K[0, 0].item(), K[1, 1].item()], 'princpt': [K[0, 2].item(), K[1, 2].item()]}
        rot_target_cam, trans_target_cam = camera['rotation'], camera['translation']

        mesh_dir = os.path.join(self.annotation_dir, subject_name, seq_name, sub_seq_name, 'Mesh')
        mesh_body_world = trimesh.load(os.path.join(mesh_dir, f'{frame_id}_second.ply'))
        mesh_obj_world = trimesh.load(os.path.join(mesh_dir, f'{frame_id}_second_obj.ply'))
        mesh_foot_r_world = extract_partial_mesh(mesh_body_world, self.smplx_rightfoot_idxs)

        ground_plane_slope_x = self.ground_plane_coeffs['slope_x']
        ground_plane_slope_z = self.ground_plane_coeffs['slope_z']
        ground_plane_intercept = self.ground_plane_coeffs['intercept']
        distances = compute_signed_distances_to_plane(mesh_body_world.vertices, ground_plane_slope_x, ground_plane_slope_z, ground_plane_intercept)
        ground_contact_smplx = (np.abs(distances) < cfg.MODEL.c_thres_fitted_ground).astype(np.uint8)
        ground_contact_f = ground_contact_smplx[self.smplx_rightfoot_idxs]
        obj_contact_f, _, _, _ = get_ho_contact_and_offset(mesh_foot_r_world, mesh_obj_world, cfg.MODEL.c_thres)
        contact_f = np.array(ground_contact_f.astype(int) | obj_contact_f.astype(int))

        verts_ho_world = np.concatenate([mesh_body_world.vertices, mesh_obj_world.vertices], axis=0)
        center_xz_world = (verts_ho_world[:, 0].mean(), 0.0, verts_ho_world[:, 2].mean())
        ground_mesh_world = create_ground_plane_mesh(
            ground_plane_slope_x,
            ground_plane_slope_z,
            ground_plane_intercept,
            center=center_xz_world,
            size=(5.0, 5.0),
            resolution=50
        )
        ground_mesh_cam = ground_mesh_world.copy()
        ground_mesh_cam.vertices = (rot_target_cam @ ground_mesh_cam.vertices.T).T + trans_target_cam
        ground_plane_slope_x_cam, ground_plane_slope_z_cam, ground_plane_intercept_cam = extract_ground_plane(ground_mesh_cam)
        ground_normal = compute_ground_normal(ground_plane_slope_x_cam, ground_plane_slope_z_cam)

        mesh_foot_r_cam = mesh_foot_r_world.copy()
        mesh_foot_r_cam.vertices = (rot_target_cam @ mesh_foot_r_cam.vertices.T).T + trans_target_cam
        mesh_foot_r_img = cam2pixel(mesh_foot_r_cam.vertices, cam_param['focal'], cam_param['princpt'])
        bbox_foot_r = get_bbox(mesh_foot_r_img, np.ones(len(mesh_foot_r_img)), expansion_factor=cfg.DATASET.foot_scene_distorted_bbox_expand_ratio)

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

        _, trans, inv_trans, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)
        height = apply_augmentation_height(pixel_height_map, trans, False, cfg.MODEL.input_img_shape)
        valid = apply_augmentation_mask(valid_height_mask, trans, False, cfg.MODEL.input_img_shape)
        annot_data = dict(sample_id=sample_id, cam_param=cam_param, bbox_foot_r=bbox_foot_r, inv_trans_foot_r=inv_trans)

        output = Path(self.output_path)
        np.savez(output / 'annot_data' / f'{sample_id}.npz', **annot_data)
        np.save(output / 'contact_data' / f'{sample_id}.npy', contact_f)
        np.save(output / 'ground_data/normal' / f'{sample_id}.npy', ground_normal)
        np.save(output / 'pixel_height_data/map' / f'{sample_id}.npy', np.round(height).astype(np.int16))
        if not cv2.imwrite(str(output / 'pixel_height_data/mask' / f'{sample_id}.png'), (valid * 255).astype(np.uint8)):
            raise OSError(f'Could not write mask for {sample_id}')
        return sample_id


if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train'])
    parser.add_argument('--output-dir', type=Path, default=Path('data/InterCap/preprocessed_data'))
    args = parser.parse_args()
    split_root = Path('data/InterCap/preprocessed_data')
    manifests = [split_root / args.split / 'split_sample_ids.txt'] if args.split else sorted(split_root.glob('*/split_sample_ids.txt'))
    if not manifests:
        raise FileNotFoundError(f'No split_sample_ids.txt found under {split_root}')
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f'Missing split manifest: {manifest}')
        output = args.output_dir / manifest.parent.name
        output.mkdir(parents=True, exist_ok=True)
        if manifest.resolve() != (output / 'split_sample_ids.txt').resolve():
            shutil.copyfile(manifest, output / 'split_sample_ids.txt')
        for folder in ['annot_data', 'contact_data', 'ground_data/normal', 'pixel_height_data/map', 'pixel_height_data/mask']:
            (output / folder).mkdir(parents=True, exist_ok=True)
        transform = transforms.ToTensor()
        style_transform = transforms.ToTensor()
        dataset = InterCap(transform, style_transform, manifest.parent.name, task='annotate')
        dataset.output_path = output
        with torch.no_grad():
            for index in tqdm(range(len(dataset)), desc='InterCap ' + manifest.parent.name):
                dataset[index]