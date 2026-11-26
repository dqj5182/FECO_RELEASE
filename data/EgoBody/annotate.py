import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import cv2
import json
import glob
import smplx
import pandas as pd
import pickle
import trimesh
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lib.core.config import cfg
from lib.utils.transforms import cam2pixel, apply_homogeneous_transformation_np
from lib.utils.func_utils import load_img, get_bbox as get_bbox_square
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.preprocessing import augmentation_contact, apply_augmentation_height, apply_augmentation_mask
from lib.utils.pixel_height_utils import render_foot_depth, compute_image_pixel_height_along_ground_direction
from lib.utils.mesh_utils import create_ground_plane_mesh, compute_ground_normal, extract_ground_plane
from lib.utils.mesh_utils import extract_partial_mesh
from lib.utils.vis_utils import ContactRenderer

contact_renderer = ContactRenderer()


def get_scene_name(cal_root, rec_name):
    files = glob.glob(f'{cal_root}/{rec_name}/cal_trans/kinect12_to_world/*.json')
    return os.path.splitext(os.path.basename(files[0]))[0] if files else None


def get_bbox(joint_img, joint_valid, expansion_factor=1.0):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = (xmax-xmin)*expansion_factor;
    xmin = x_center - 0.5*width
    xmax = x_center + 0.5*width

    y_center = (ymin+ymax)/2.; height = (ymax-ymin)*expansion_factor;
    ymin = y_center - 0.5*height
    ymax = y_center + 0.5*height

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


class EgoBody(Dataset):
    def __init__(self, transform, style_transform, data_split, task=None):
        super(EgoBody, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        self.style_transform = style_transform
        self.output_path = Path('data') / 'EgoBody' / 'preprocessed_data' / data_split
        self.legacy_bbox_ids = set(Path(__file__).with_name('splits').joinpath(data_split + '_legacy_bbox_ids.txt').read_text().splitlines())
        self.legacy_contact_recordings = set(Path(__file__).with_name('splits').joinpath('legacy_contact_recordings.txt').read_text().splitlines())
        self.body_models = {}
        self.dataset_name = 'egobody'
        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'EgoBody')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.camera_dir = os.path.join(self.data_dir, 'kinect_cam_params')
        self.scene_mesh_dir = os.path.join(self.data_dir, 'scene_mesh')
        self.calibration_dir = os.path.join(self.data_dir, 'calibrations')
        self.data_info = pd.read_csv(os.path.join(self.data_dir, 'data_info_release.csv'))
        self.data_split_info = pd.read_csv(os.path.join(self.data_dir, 'data_splits.csv'))
        split_sample_id_file_path = str(Path('data/EgoBody/preprocessed_data') / data_split / 'split_sample_ids.txt')
        with open(split_sample_id_file_path, encoding='utf-8') as f:
            self.db = [line.strip() for line in f if line.strip()]
        self.kinetics_dict = {'master': 12, 'sub_1': 11, 'sub_2': 13, 'sub_3': 14, 'sub_4': 15}
        self.camera_data = {'kinect_master': [], 'kinect_sub_1': [], 'kinect_sub_2': [], 'kinect_sub_3': [], 'kinect_sub_4': []}
        for each_camera in [*self.camera_data]:
            with open(os.path.join(self.camera_dir, each_camera, 'Color.json'), 'r') as f:
                each_camera_data = json.load(f)
            self.camera_data[each_camera] = each_camera_data

        with open('data/base_data/conversions/smplx_vert_segmentation.json', 'r') as f:
            self.smplx_part_seg = json.load(f)
            self.smplx_leftfoot_idxs = self.smplx_part_seg['leftFoot']
            self.smplx_rightfoot_idxs = self.smplx_part_seg['rightFoot']
            self.smplx_lefttoe_idxs = self.smplx_part_seg['leftToeBase']
            self.smplx_righttoe_idxs = self.smplx_part_seg['rightToeBase']
            self.smplx_leftfoot_idxs = np.array(self.smplx_leftfoot_idxs + self.smplx_lefttoe_idxs)
            self.smplx_rightfoot_idxs = np.array(self.smplx_rightfoot_idxs + self.smplx_righttoe_idxs)


        self.ground_plane_coeffs_world = {
                                        'seminar_g110': {'slope_x': -0.01243882702620634, 'slope_z': 0.002003330338846491, 'intercept': -1.6616062773698008},
                                        'seminar_h52': {'slope_x': 0.001277403308641447,'slope_z': 0.001123768201110772, 'intercept': -0.5057067908914791},
                                        'kitchen_gfloor': {'slope_x': 0.0016836271629233016, 'slope_z': 0.00013181574761274428, 'intercept': -0.8421900135549135},
                                        'cab_e': {'slope_x': 0.0007028084327446223, 'slope_z': 0.006315266140192529, 'intercept': -0.7118280414753204},
                                        'cab_h_tables': {'slope_x': -0.0007532804150646103, 'slope_z': -0.0005418926240979, 'intercept': -0.33183330989383414},
                                        'seminar_d78': {'slope_x': 0.0020367192131017144, 'slope_z': 0.000249695789946862, 'intercept': -0.8142851155650215},
                                        'seminar_d78_0318': {'slope_x': -0.00163178420689663, 'slope_z': 0.0022964335918064984, 'intercept': -1.0230844126499916},
                                        'seminar_g110_0415': {'slope_x': 0.0007352632435973441, 'slope_z': -0.0018434813147808259, 'intercept': -0.7891731097234855},
                                        'foodlab_0312': {'slope_x': -0.005830124538798786, 'slope_z': -0.000576601049087027, 'intercept': -0.7094501060276045},
                                        'cab_g_benches': {'slope_x': -0.0006035999693871452, 'slope_z': -0.0007448107697422907, 'intercept': -0.15538600158354357},
                                        'seminar_j716': {'slope_x': 0.000745377361571713, 'slope_z': -0.0010808526528580113, 'intercept': -0.8945951885066635},
                                        'seminar_h53_0218': {'slope_x': -4.12334329918273e-05, 'slope_z': -0.003414559202846256, 'intercept': -0.7865086139037313},
                                        'cnb_dlab_0215': {'slope_x': 0.0005248446309989809, 'slope_z': -0.0007400589459382387, 'intercept': -0.11932542119583164},
                                        'seminar_g110_0315': {'slope_x': 0.0036225272516694045, 'slope_z': -0.0004462810203995797, 'intercept': -0.7337753329620351},
                                        'cnb_dlab_0225': {'slope_x': 0.0007152839799067577, 'slope_z': 0.0019866348230020026, 'intercept': -0.15529019837073307},
                                        }
        self.scene_mesh_dict = {}


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        camera_type = aid.split('-')[0]
        seq_name = aid.split('-')[1]
        view_name = aid.split('-')[2]
        img_name = aid.split('-')[3]
        sample_id = f'{camera_type}-{seq_name}-{view_name}-{img_name}'

        orig_img_path = os.path.join(self.data_dir, camera_type, seq_name, view_name, f'{img_name}.jpg')
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape
        camera_data = self.camera_data[f'kinect_{view_name}']
        cam_param = {'focal': camera_data['f'], 'princpt': camera_data['c']}
        recording_name_list = list(self.data_info['recording_name'])
        body_idx_fpv_list = list(self.data_info['body_idx_fpv'])
        gender_0_list = list(self.data_info['body_idx_0'])
        gender_1_list = list(self.data_info['body_idx_1'])
        body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
        gender_0_dict = dict(zip(recording_name_list, gender_0_list))
        gender_1_dict = dict(zip(recording_name_list, gender_1_list))
        scene_name = get_scene_name(self.calibration_dir, seq_name)
        interactee_idx = int(body_idx_fpv_dict[seq_name].split(' ')[0])
        camera_wearer_idx = 1 - interactee_idx
        interactee_gender = body_idx_fpv_dict[seq_name].split(' ')[1]
        if camera_wearer_idx == 0:
            camera_wearer_gender = gender_0_dict[seq_name].split(' ')[1]
        elif camera_wearer_idx == 1:
            camera_wearer_gender = gender_1_dict[seq_name].split(' ')[1]
        calib_trans_dir = os.path.join(self.calibration_dir, seq_name)
        cam_extr_master_to_world_path = os.path.join(calib_trans_dir, 'cal_trans', 'kinect12_to_world', f'{scene_name}.json')
        with open(cam_extr_master_to_world_path, 'r') as f:
            cam_extr_master_to_world = np.array(json.load(f)['trans'])
            cam_extr_world_to_master = np.linalg.inv(cam_extr_master_to_world)
        if scene_name not in self.scene_mesh_dict:
            scene_path = os.path.join(self.scene_mesh_dir, scene_name, scene_name + '.obj')
            self.scene_mesh_dict[scene_name] = trimesh.load(scene_path)
        scene_mesh_world = self.scene_mesh_dict[scene_name]
        if view_name != 'master':
            kinetics_id = self.kinetics_dict[view_name]
            cam_extr_cam_to_master_path = os.path.join(calib_trans_dir, 'cal_trans', f'kinect_{view_name}to12_color.json')
            with open(cam_extr_cam_to_master_path, 'r') as f:
                cam_extr_cam_to_master = np.array(json.load(f)['trans'])
                cam_extr_master_to_cam = np.linalg.inv(cam_extr_cam_to_master)
        fitting_root_interactee = os.path.join(self.data_dir, 'smplx_interactee_{}'.format(self.data_split), seq_name)
        with open(os.path.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', img_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key])
        if interactee_gender not in self.body_models:
            self.body_models[interactee_gender] = smplx.create(cfg.MODEL.human_model_path, model_type='smplx',
                                        gender=interactee_gender, ext='npz', num_pca_comps=12,
                                        create_global_orient=True, create_transl=True, create_body_pose=True,
                                        create_betas=True,
                                        create_left_hand_pose=True, create_right_hand_pose=True,
                                        create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                        create_reye_pose=True)
        model_interactee = self.body_models[interactee_gender]
        output = model_interactee(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        mesh_body_master = trimesh.Trimesh(vertices, model_interactee.faces, process=False)
        mesh_body_world = mesh_body_master.copy()
        mesh_body_world.vertices = apply_homogeneous_transformation_np(mesh_body_master.vertices, cam_extr_master_to_world)
        mesh_foot_r_master = extract_partial_mesh(mesh_body_master, self.smplx_rightfoot_idxs)
        mesh_foot_r_world = mesh_foot_r_master.copy()
        mesh_foot_r_world.vertices = apply_homogeneous_transformation_np(mesh_foot_r_world.vertices, cam_extr_master_to_world)
        mesh_body_cam = mesh_body_master.copy()
        mesh_foot_r_cam = mesh_foot_r_master.copy()
        if view_name != 'master':
            mesh_body_cam.apply_transform(cam_extr_master_to_cam)
            mesh_foot_r_cam.apply_transform(cam_extr_master_to_cam)
        mesh_foot_r_img = cam2pixel(mesh_foot_r_cam.vertices, cam_param['focal'], cam_param['princpt'])
        bbox_func = get_bbox if sample_id in self.legacy_bbox_ids else get_bbox_square
        bbox_foot_r = bbox_func(mesh_foot_r_img, np.ones(len(mesh_foot_r_img)), expansion_factor=cfg.DATASET.foot_scene_distorted_bbox_expand_ratio)
        if view_name != 'master':
            cam_extr_world_to_cam = cam_extr_master_to_cam @ cam_extr_world_to_master
            w2l_rot, w2l_trans = (cam_extr_world_to_cam[:3, :3], cam_extr_world_to_cam[:3, 3])
        else:
            w2l_rot, w2l_trans = (cam_extr_world_to_master[:3, :3], cam_extr_world_to_master[:3, 3])
        rot_target_cam, trans_target_cam = (w2l_rot, w2l_trans)
        ground_plane_slope_x, ground_plane_slope_z, ground_plane_intercept = (self.ground_plane_coeffs_world[scene_name]['slope_x'], self.ground_plane_coeffs_world[scene_name]['slope_z'], self.ground_plane_coeffs_world[scene_name]['intercept'])
        x_center_world = mesh_body_world.vertices[:, 0].mean()
        z_center_world = mesh_body_world.vertices[:, 2].mean()
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
        contact_mesh = mesh_foot_r_cam if seq_name in self.legacy_contact_recordings else mesh_foot_r_world
        contact_f, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(contact_mesh, scene_mesh_world, cfg.MODEL.c_thres)

        output = Path(self.output_path)
        _, trans, inv_trans, _, _, _, _, _ = augmentation_contact(orig_img.copy(), bbox_foot_r, data_split='test', enforce_flip=False)
        height_bbox = get_bbox_square(mesh_foot_r_img, np.ones(len(mesh_foot_r_img)), expansion_factor=cfg.DATASET.foot_scene_distorted_bbox_expand_ratio)
        _, height_trans, _, _, _, _, _, _ = augmentation_contact(orig_img.copy(), height_bbox, data_split='test', enforce_flip=False)
        height = apply_augmentation_height(pixel_height_map, height_trans, False, cfg.MODEL.input_img_shape)
        valid = apply_augmentation_mask(valid_height_mask, height_trans, False, cfg.MODEL.input_img_shape)
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
    parser.add_argument('--output-dir', type=Path, default=Path('data/EgoBody/preprocessed_data'))
    args = parser.parse_args()
    split_root = Path('data/EgoBody/preprocessed_data')
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
        dataset = EgoBody(transform, style_transform, manifest.parent.name, task='annotate')
        dataset.output_path = output
        count = len(dataset)
        with torch.no_grad():
            for index in tqdm(range(count), desc='EgoBody ' + manifest.parent.name):
                dataset[index]
        print('EgoBody ' + manifest.parent.name + ': ' + str(count) + ' samples complete')