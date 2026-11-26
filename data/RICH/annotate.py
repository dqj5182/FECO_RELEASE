import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import json
import numpy as np
import pickle
import trimesh
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
from smplx.lbs import transform_mat
from lib.core.config import cfg
from lib.utils.transforms import cam2pixel, apply_homogeneous_transformation_np
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.preprocessing import augmentation_contact
from lib.utils.pixel_height_utils import render_foot_depth, compute_image_pixel_height_along_ground_direction
from lib.utils.mesh_utils import create_ground_plane_mesh, compute_ground_normal, extract_ground_plane
from lib.utils.vis_utils import ContactRenderer

contact_renderer = ContactRenderer()


def extract_cam_param_xml(xml_path='', dtype=torch.float32):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)

    rotation = torch.tensor([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]],
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]],
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = torch.tensor([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                    -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11],
                    -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

    cam_center = torch.tensor([cam_center], dtype=dtype)

    k1 = torch.tensor([distortion_vec[0]], dtype=dtype)
    k2 = torch.tensor([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2


class CalibratedCamera(nn.Module):

    def __init__(self, calib_path='', rotation=None, translation=None, focal_length_x=None, focal_length_y=None, batch_size=1, center=None, dtype=torch.float32, **kwargs):
        super(CalibratedCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        self.calib_path = calib_path
        self.register_buffer('zero', torch.zeros([batch_size], dtype=dtype))
        if not os.path.exists(calib_path):
            raise FileNotFoundError('Couldt find {}.'.format(calib_path))
        else:
            focal_length_x, focal_length_y, center, rotation, translation, cam_center, _, _ \
                    = extract_cam_param_xml(xml_path=calib_path, dtype=dtype)
        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full([batch_size], focal_length_x, dtype=dtype)
        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full([batch_size], focal_length_y, dtype=dtype)
        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)
        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)
        rotation = rotation.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        rotation = nn.Parameter(rotation, requires_grad=False)
        self.register_parameter('rotation', rotation)
        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)
        translation = translation.view(3, -1).repeat(batch_size, 1, 1).squeeze(dim=-1)
        translation = nn.Parameter(translation, requires_grad=False)
        self.register_parameter('translation', translation)
        cam_center = nn.Parameter(cam_center, requires_grad=False)
        self.register_parameter('cam_center', cam_center)

    def forward(self, points):
        device = points.device
        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2], dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y
        camera_transform = transform_mat(self.rotation, self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1], dtype=points.dtype, device=device)
        points_h = torch.cat([points, homog_coord], dim=-1)
        projected_points = torch.einsum('bki,bji->bjk', [camera_transform, points_h])
        img_points = torch.div(projected_points[:, :, :2], projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) + self.center.unsqueeze(dim=1)
        return img_points


class RICH(Dataset):
    def __init__(self, transform, style_transform, data_split, task=None):
        super(RICH, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.style_transform = style_transform
        self.output_path = Path('data') / 'RICH' / 'preprocessed_data' / data_split
        self.dataset_name = 'rich'
        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'RICH')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.scan_calibration_dir = os.path.join(self.data_dir, 'scan_calibration')
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
        split_sample_id_file_path = str(Path('data/RICH/preprocessed_data') / data_split / 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]


        self.ground_plane_coeffs_world = {
                                        'BBQ': {'slope_x': 0.02896752966970827, 'slope_z': -0.13739345059413977, 'intercept': 1.4216280163129},
                                        'Gym': {'slope_x': -0.0551509672433374, 'slope_z': -0.23722705094667726, 'intercept': 1.5771458362125366},
                                        'LectureHall': {'slope_x': 0.02729623340593575, 'slope_z': -0.2995441817378889, 'intercept': 1.979380422788644},
                                        'ParkingLot1': {'slope_x': -0.022766824900505665, 'slope_z': -0.1284374934974873, 'intercept': 1.3724529223783881},
                                        'ParkingLot2': {'slope_x': -0.01801075810423359, 'slope_z': -0.13162635414275703, 'intercept': 1.3752358220214005},
                                        }


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        seq_name = aid.split('-')[0]
        seq_loc_name = seq_name.split('_')[0]
        annot_name = seq_name.split('_')[1]
        cam_name = aid.split('-')[1]
        cam_id = int(cam_name.split('cam_')[-1])
        img_name = aid.split('-')[2]
        img_annot_name = img_name.split('_')[0]
        sample_id = f'{seq_name}-{cam_name}-{img_name}'

        orig_img_path = os.path.join(self.data_dir, 'images_jpg_subset', self.data_split, seq_name, cam_name, f'{img_name}.jpeg')
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        camera_path = os.path.join(self.data_dir, 'scan_calibration', seq_loc_name, 'calibration', f'{cam_id:03d}.xml')
        cam = CalibratedCamera(calib_path=camera_path)
        cam_param = {'focal': [cam.focal_length_x.item(), cam.focal_length_y.item()], 'princpt': [cam.center[0][0].item(), cam.center[0][1].item()]}
        contact_mesh_path = os.path.join(self.data_dir, 'hsc', self.data_split, seq_name, img_annot_name, f'{annot_name}.obj')
        contact_mesh_data_path = os.path.join(self.data_dir, 'hsc', self.data_split, seq_name, img_annot_name, f'{annot_name}.pkl')
        contact_mesh_world = trimesh.load(contact_mesh_path)
        contact_mesh_data = np.load(contact_mesh_data_path, allow_pickle=True)
        contact_smplx = np.matmul(self.smpl_to_smplx_mapping['matrix'], contact_mesh_data['contact'])
        contact_f = contact_smplx[self.smplx_rightfoot_idxs]
        ext_camera_transform = transform_mat(cam.rotation, cam.translation.unsqueeze(dim=-1))[0].detach().cpu().numpy()
        contact_mesh_cam = contact_mesh_world.copy()
        contact_mesh_cam.vertices = apply_homogeneous_transformation_np(contact_mesh_cam.vertices, ext_camera_transform)
        mesh_foot_r_cam = contact_mesh_cam.vertices[self.smplx_rightfoot_idxs]
        mesh_foot_r_img = cam2pixel(mesh_foot_r_cam, cam_param['focal'], cam_param['princpt'])
        bbox_foot_r = get_bbox(mesh_foot_r_img, np.ones(len(mesh_foot_r_img)), expansion_factor=cfg.DATASET.foot_scene_bbox_expand_ratio)
        rot_target_cam, trans_target_cam = (ext_camera_transform[:3, :3], ext_camera_transform[:3, 3])
        ground_plane_slope_x, ground_plane_slope_z, ground_plane_intercept = (self.ground_plane_coeffs_world[seq_loc_name]['slope_x'], self.ground_plane_coeffs_world[seq_loc_name]['slope_z'], self.ground_plane_coeffs_world[seq_loc_name]['intercept'])
        x_center_world = contact_mesh_world.vertices[:, 0].mean()
        z_center_world = contact_mesh_world.vertices[:, 2].mean()
        center_xz_world = (x_center_world, 0.0, z_center_world)
        ground_mesh_world = create_ground_plane_mesh(
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
        _, _, inv_trans_foot_r, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)

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
    parser.add_argument('--output-dir', type=Path, default=Path('data/RICH/preprocessed_data'))
    args = parser.parse_args()
    split_root = Path('data/RICH/preprocessed_data')
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
        dataset = RICH(transform, style_transform, manifest.parent.name, task='annotate')
        dataset.output_path = output
        count = len(dataset)
        with torch.no_grad():
            for index in tqdm(range(count), desc='RICH ' + manifest.parent.name):
                dataset[index]
        print('RICH ' + manifest.parent.name + ': ' + str(count) + ' samples complete')