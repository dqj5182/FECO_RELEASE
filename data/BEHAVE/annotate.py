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
from lib.utils.transforms import cam2pixel
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.human_models import smpl_x
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.preprocessing import augmentation_contact, apply_augmentation_height, apply_augmentation_mask
from lib.utils.pixel_height_utils import render_foot_depth, compute_image_pixel_height_along_ground_direction
from lib.utils.mesh_utils import create_ground_plane_mesh, compute_ground_normal, compute_signed_distances_to_plane, extract_ground_plane
from lib.utils.mesh_utils import extract_partial_mesh
from lib.utils.vis_utils import ContactRenderer

contact_renderer = ContactRenderer()


def rotate_yaxis(R, t):
    "rotate the transformation matrix around z-axis by 180 degree ==>> let y-axis point up"
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    global_trans = np.eye(4)
    global_trans[0, 0] = global_trans[1, 1] = -1  # rotate around z-axis by 180
    rotated = np.matmul(global_trans, transform)
    return rotated[:3, :3], rotated[:3, 3]


def load_kinect_poses(config_folder, kids):
    pose_calibs = [json.load(open(os.path.join(config_folder, f"{x}/config.json"))) for x in kids]
    rotations = [np.array(pose_calibs[x]['rotation']).reshape((3, 3)) for x in kids]
    translations = [np.array(pose_calibs[x]['translation']) for x in kids]
    return rotations, translations


def load_kinect_poses_back(config_folder, kids, rotate=False):
    """
    backward transform
    rotate: kinect y-axis pointing down, if rotate, then return a transform that make y-axis pointing up
    """
    rotations, translations = load_kinect_poses(config_folder, kids)
    rotations_back = []
    translations_back = []
    for r, t in zip(rotations, translations):
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t

        trans_back = np.linalg.inv(trans) # now the y-axis point down

        r_back = trans_back[:3, :3]
        t_back = trans_back[:3, 3]
        if rotate:
            r_back, t_back = rotate_yaxis(r_back, t_back)

        rotations_back.append(r_back)
        translations_back.append(t_back)
    return rotations_back, translations_back


class BEHAVE(Dataset):
    def __init__(self, transform, style_transform, data_split, task=None):
        super(BEHAVE, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.style_transform = style_transform
        self.output_path = Path('data') / 'BEHAVE' / 'preprocessed_data' / data_split
        self.dataset_name = 'behave'
        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'BEHAVE')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.calibs_path = os.path.join(self.data_dir, 'calibs')
        self.split_path = os.path.join(self.data_dir, 'split.json')
        with open(self.split_path, 'r') as f:
            split_seq_names = json.load(f)[data_split]
        self.world2local = {}


        self.ground_plane_coeffs = {
                                    'Date01': {'slope_x': 0.027886679868858337, 'slope_z': -0.008861975902524766, 'intercept': 1.2010171273689858},
                                    'Date02': {'slope_x': -0.003986309215670653, 'slope_z': 0.010984028844282371, 'intercept': 1.1783578342733487},
                                    'Date03': {'slope_x': 0.03393579524309811, 'slope_z': 0.01698806891885562, 'intercept': 1.2301442773210718},
                                    'Date04': {'slope_x': 0.007791812651764115, 'slope_z': -0.009504492838200576, 'intercept': 1.19440280933608},
                                    'Date05': {'slope_x': 0.022922587915062298, 'slope_z': -0.012357171309608093, 'intercept': 1.1935510398680813},
                                    'Date06': {'slope_x': 0.01953566944407335, 'slope_z': -0.018724217446130235, 'intercept': 1.2132717332079113},
                                    'Date07': {'slope_x': 0.009182683971260567, 'slope_z': -0.00393739368066494, 'intercept': 1.2173628769412963},
                                    }
        split_sample_id_file_path = str(Path('data/BEHAVE/preprocessed_data') / data_split / 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]
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
        seq_name = aid.split('-')[0]
        date_name = seq_name.split('_')[0]
        obj_name = seq_name.split('_')[2]
        if 'chair' in obj_name:
            obj_name = 'chair'
        elif 'monitor' in obj_name:
            obj_name = 'monitor'
        elif obj_name in ['yogaball', 'basketball']:
            obj_name = 'sports ball'
        elif 'keyboard' in obj_name:
            obj_name = 'keyboard'
        frame_id = aid.split('-')[2]
        camera_id = aid.split('-')[1]
        kid = int(camera_id.split('k')[-1])
        sample_id = f'{seq_name}-{camera_id}-{frame_id}'


        orig_img_path = os.path.join(self.data_dir, date_name, seq_name, frame_id, f'{camera_id}.color.jpg')
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape
        cam_intr_path = os.path.join(self.calibs_path, 'intrinsics', str(kid), 'calibration.json')
        with open(cam_intr_path, 'r') as f:
            cam_intr = json.load(f)
            cam_param = {'focal': [cam_intr['color']['fx'], cam_intr['color']['fy']], 'princpt': [cam_intr['color']['cx'], cam_intr['color']['cy']]}


        mesh_body_path = os.path.join(self.data_dir, date_name, seq_name, frame_id, 'person', 'fit02', f'person_fit.ply')
        mesh_body = trimesh.load(mesh_body_path)
        mesh_body_world = mesh_body.copy()
        mesh_smplx_world = trimesh.Trimesh(np.matmul(self.smpl_to_smplx_mapping['matrix'], mesh_body_world.vertices), smpl_x.face, process=False)
        mesh_foot_r_world = extract_partial_mesh(mesh_smplx_world, self.smplx_rightfoot_idxs)
        mesh_obj_path = os.path.join(self.data_dir, date_name, seq_name, frame_id, obj_name, 'fit01', f'{obj_name}_fit.ply')
        mesh_obj = trimesh.load(mesh_obj_path)
        mesh_obj_world = mesh_obj.copy()
        if date_name in self.world2local:
            w2l_rot, w2l_trans = (self.world2local[date_name]['rot'], self.world2local[date_name]['trans'])
        else:
            w2l_rot, w2l_trans = load_kinect_poses_back(os.path.join(self.calibs_path, date_name, 'config'), [0, 1, 2, 3])
            self.world2local[date_name] = {}
            self.world2local[date_name]['rot'] = w2l_rot
            self.world2local[date_name]['trans'] = w2l_trans
        ground_plane_slope_x, ground_plane_slope_z, ground_plane_intercept = (self.ground_plane_coeffs[date_name]['slope_x'], self.ground_plane_coeffs[date_name]['slope_z'], self.ground_plane_coeffs[date_name]['intercept'])
        distances = compute_signed_distances_to_plane(mesh_body_world.vertices, ground_plane_slope_x, ground_plane_slope_z, ground_plane_intercept)
        ground_contact_smpl = (np.abs(distances) < cfg.MODEL.c_thres_fitted_ground).astype(np.uint8)
        ground_contact_smplx = np.matmul(self.smpl_to_smplx_mapping['matrix'], ground_contact_smpl)
        ground_contact_f = ground_contact_smplx[self.smplx_rightfoot_idxs]
        rot_target_cam, trans_target_cam = (w2l_rot[kid], w2l_trans[kid])
        verts_ho_world = np.concatenate([mesh_body_world.vertices, mesh_obj_world.vertices], axis=0)
        x_center_world = verts_ho_world[:, 0].mean()
        z_center_world = verts_ho_world[:, 2].mean()
        center_xz_world = (x_center_world, 0.0, z_center_world)
        ground_mesh_world = create_ground_plane_mesh(
            ground_plane_slope_x,
            ground_plane_slope_z,
            ground_plane_intercept,
            center=center_xz_world,
            size=(5.0, 5.0),
            resolution=50
        )
        mesh_body_cam = mesh_body_world.copy()
        mesh_body_cam.vertices = (rot_target_cam @ mesh_body_cam.vertices.T).T + trans_target_cam
        mesh_obj_cam = mesh_obj_world.copy()
        mesh_obj_cam.vertices = (rot_target_cam @ mesh_obj_cam.vertices.T).T + trans_target_cam
        ground_mesh_world_transformed = ground_mesh_world.copy()
        ground_mesh_world_transformed.vertices = (rot_target_cam @ ground_mesh_world_transformed.vertices.T).T + trans_target_cam
        ground_plane_slope_x_cam, ground_plane_slope_z_cam, ground_plane_intercept_cam = extract_ground_plane(ground_mesh_world_transformed)


        ground_normal = compute_ground_normal(ground_plane_slope_x_cam, ground_plane_slope_z_cam)
        obj_contact_f, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_foot_r_world, mesh_obj_world, cfg.MODEL.c_thres)
        mesh_smplx_cam = mesh_smplx_world.copy()
        mesh_foot_r_cam = mesh_foot_r_world.copy()
        mesh_smplx_cam.vertices = np.matmul(mesh_smplx_cam.vertices, w2l_rot[kid].T) + w2l_trans[kid]
        mesh_foot_r_cam.vertices = np.matmul(mesh_foot_r_cam.vertices, w2l_rot[kid].T) + w2l_trans[kid]
        mesh_foot_r_img = cam2pixel(mesh_foot_r_cam.vertices, cam_param['focal'], cam_param['princpt'])
        bbox_foot_r = get_bbox(mesh_foot_r_img, np.ones(len(mesh_foot_r_img)), expansion_factor=cfg.DATASET.foot_scene_bbox_expand_ratio)
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
        _, _, inv_trans_foot_r, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)
        contact_f = np.array(ground_contact_f.astype(int) | obj_contact_f.astype(int))


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
    parser.add_argument('--output-dir', type=Path, default=Path('data/BEHAVE/preprocessed_data'))
    args = parser.parse_args()
    split_root = Path('data/BEHAVE/preprocessed_data')
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
        dataset = BEHAVE(transform, style_transform, manifest.parent.name, task='annotate')
        dataset.output_path = output
        count = len(dataset)
        with torch.no_grad():
            for index in tqdm(range(count), desc='BEHAVE ' + manifest.parent.name):
                dataset[index]
        print('BEHAVE ' + manifest.parent.name + ': ' + str(count) + ' samples complete')