import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import cv2
import json
import smplx
import pickle
import trimesh
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lib.core.config import cfg
from lib.utils.human_models import smpl_x
from lib.utils.transforms import cam2pixel, apply_homogeneous_transformation_np
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.preprocessing import augmentation_contact
from lib.utils.pixel_height_utils import render_foot_depth, compute_image_pixel_height_along_ground_direction
from lib.utils.mesh_utils import create_ground_plane_mesh_xz, compute_ground_normal, extract_ground_plane
from lib.utils.vis_utils import ContactRenderer

contact_renderer = ContactRenderer()


smpl_model = smplx.create('data/base_data/human_models/smpl/SMPL_NEUTRAL.pkl')


class MotionPro(Dataset):
    def __init__(self, transform, style_transform, data_split, task=None):
        super(MotionPro, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.style_transform = style_transform
        self.output_path = Path('data') / 'MotionPro' / 'preprocessed_data' / data_split
        self.dataset_name = 'motionpro'
        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'MotionPro')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.camera_pose_path = os.path.join(self.root_path, 'camera_pose')
        split_sample_id_file_path = str(Path('data/MotionPro/preprocessed_data') / data_split / 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]
        self.cam_intr_data = {
                            'cam0': {'focal': [607.460, 607.295],'princpt': [642.213, 363.547]},
                            'cam1': {'focal': [606.187, 606.015], 'princpt': [640.228, 366.479]},
                            'cam2': {'focal': [609.233, 609.125], 'princpt': [640.599, 365.104]},
                            'cam3': {'focal': [608.881, 608.737], 'princpt': [641.453, 366.285]}
                            }
        self.img_res = [720, 1280]
        self.ground_level = 0.0


        self.ground_plane_coeffs_world = {'slope_x': 0.0, 'slope_z': 0.0, 'intercept': 0.0}
        smpl_smplx_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smpl_to_smplx.pkl')
        with open(smpl_smplx_mapping_path, 'rb') as f:
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
        seq_name = aid.split('-')[1]
        sub_seq_name = aid.split('-')[2]
        camera_id = aid.split('-')[3]
        camera_name = 'cam' + camera_id
        img_name = aid.split('-')[4]
        image_id = int(img_name) - 1
        sample_id = f'{date_name}-{seq_name}-{sub_seq_name}-{camera_id}-{img_name}'

        orig_img_path = os.path.join(self.data_dir, date_name, seq_name, sub_seq_name.replace('_', '-'), camera_id, f'{img_name}.jpg')
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape
        cam_param = self.cam_intr_data[camera_name]
        cam_extr_data_path = os.path.join(self.camera_pose_path, f'{date_name}.npy')
        cam_extr_data = np.load(cam_extr_data_path, allow_pickle=True).item()
        cam_extr_cam2world = cam_extr_data[int(camera_id)]
        cam_extr = np.linalg.inv(cam_extr_cam2world)

        mesh_body_path = os.path.join(self.data_dir, date_name, seq_name, sub_seq_name.replace('_', '-'), 'smpl.npy')
        mesh_body_data = np.load(mesh_body_path, allow_pickle=True).item()
        betas = torch.tensor(mesh_body_data['betas'][:10], dtype=torch.float32)[None]
        body_pose = torch.tensor(mesh_body_data['body_pose'], dtype=torch.float32)[image_id].reshape(-1, 23, 3)
        global_orient = torch.tensor(mesh_body_data['global_orient'], dtype=torch.float32)[image_id][None, None]
        transl = torch.tensor(mesh_body_data['transl'], dtype=torch.float32)[image_id][None]
        mesh_body_out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        mesh_body_world = trimesh.Trimesh(mesh_body_out.vertices[0], smpl_model.faces)
        angle_radians = np.radians(-90)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]])
        mesh_body_world.vertices = mesh_body_world.vertices @ rotation_matrix.T
        mesh_body_cam = mesh_body_world.copy()
        mesh_body_cam.vertices = apply_homogeneous_transformation_np(mesh_body_cam.vertices, cam_extr)
        mesh_smplx_cam = trimesh.Trimesh(np.matmul(self.smpl_to_smplx_mapping['matrix'], mesh_body_cam.vertices), smpl_x.face, process=False)
        mesh_foot_r_cam = mesh_smplx_cam.vertices[self.smplx_rightfoot_idxs]
        mesh_foot_r_img = cam2pixel(mesh_foot_r_cam, cam_param['focal'], cam_param['princpt'])
        bbox_foot_r = get_bbox(mesh_foot_r_img, np.ones(len(mesh_foot_r_img)), expansion_factor=cfg.DATASET.foot_motionpro_bbox_expand_ratio)
        _, _, inv_trans_foot_r, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)
        mesh_smplx_world = trimesh.Trimesh(np.matmul(self.smpl_to_smplx_mapping['matrix'], mesh_body_world.vertices), smpl_x.face, process=False)
        mesh_foot_world = trimesh.Trimesh(mesh_smplx_world.vertices[self.smplx_rightfoot_idxs], contact_renderer.mesh_foot_r.faces, process=False)
        contact_f = 1 * (mesh_foot_world.vertices[:, 2] - self.ground_level < cfg.MODEL.c_thres_motionpro)
        cam_R, cam_t = (cam_extr[:3, :3], cam_extr[:3, 3])
        rot_target_cam, trans_target_cam = (cam_R, cam_t)
        ground_plane_slope_x, ground_plane_slope_z, ground_plane_intercept = (self.ground_plane_coeffs_world['slope_x'], self.ground_plane_coeffs_world['slope_z'], self.ground_plane_coeffs_world['intercept'])
        x_center_world = mesh_body_world.vertices[:, 0].mean()
        z_center_world = mesh_body_world.vertices[:, 2].mean()
        center_xz_world = (x_center_world, 0.0, z_center_world)
        ground_mesh_world = create_ground_plane_mesh_xz(
            ground_plane_slope_x,
            ground_plane_slope_z,
            ground_plane_intercept,
            center=center_xz_world,
            size=(5.0, 5.0),
            resolution=50
        )
        ground_mesh_world_transformed = ground_mesh_world.copy()
        ground_mesh_world_transformed.vertices = (rot_target_cam @ ground_mesh_world_transformed.vertices.T).T + trans_target_cam
        ground_plane_slope_x_cam, ground_plane_slope_z_cam, ground_plane_intercept_cam = extract_ground_plane(ground_mesh_world_transformed)

        ground_normal = compute_ground_normal(ground_plane_slope_x_cam, ground_plane_slope_z_cam)
        rendered_img, depth_map, valid_mask = render_foot_depth(
            img=orig_img[..., ::-1],
            mesh=mesh_foot_r_cam,
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
    parser.add_argument('--output-dir', type=Path, default=Path('data/MotionPro/preprocessed_data'))
    args = parser.parse_args()
    split_root = Path('data/MotionPro/preprocessed_data')
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
        dataset = MotionPro(transform, style_transform, manifest.parent.name, task='annotate')
        dataset.output_path = output
        count = len(dataset)
        with torch.no_grad():
            for index in tqdm(range(count), desc='MotionPro ' + manifest.parent.name):
                dataset[index]
        print('MotionPro ' + manifest.parent.name + ': ' + str(count) + ' samples complete')