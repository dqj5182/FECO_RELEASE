import numpy as np
import torch
import os.path as osp
import pickle

from lib.core.config import cfg
from lib.utils.transforms import transform_joint_to_other_db
from lib.utils.smplx import smplx


class SMPLX(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        self.layer = smplx.create(cfg.MODEL.human_model_path, 'smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True, **self.layer_arg)
        self.vertex_num = 10475
        self.face = self.layer.faces
        self.shape_param_dim = 10
        with open(osp.join(cfg.MODEL.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(osp.join(cfg.MODEL.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))


class SMPL(object):
    def __init__(self):
        self.layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(cfg.MODEL.human_model_path, 'smpl', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(cfg.MODEL.human_model_path, 'smpl', gender='MALE', **self.layer_arg), 'female': smplx.create(cfg.MODEL.human_model_path, 'smpl', gender='FEMALE', **self.layer_arg)}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10

        # original SMPL joint set
        self.orig_joint_num = 24
        self.orig_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.orig_flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)
 
        # joint set for the supervision
        self.joint_num = 33
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Index_1', 'L_Middle_1', 'L_Ring_1', 'L_Pinky_1', 'R_Index_1', 'R_Middle_1', 'R_Ring_1', 'R_Pinky_1', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel')
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.flip_pairs = ( (1,2), (3,4), (5,6), (8,9), (10,11), (12,13), (14,18), (15,19), (16,20), (17,21), (23,24), (25,26), (27,30), (28,31), (29,32) )
        self.joint_regressor = self.make_joint_regressor()

        # joint set for PositionNet prediction
        self.pos_joint_num = 25
        self.pos_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose')

    def reduce_joint_set(self, joint):
        new_joint = []
        for name in self.pos_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:,idx,:])
        new_joint = torch.stack(new_joint,1)
        return new_joint

    def make_joint_regressor(self):
        joint_regressor = transform_joint_to_other_db(self.orig_joint_regressor, self.orig_joints_name, self.joints_name)
        joint_regressor[self.joints_name.index('L_Index_1')] = np.eye(self.vertex_num)[2274]
        joint_regressor[self.joints_name.index('L_Middle_1')] = np.eye(self.vertex_num)[2270]
        joint_regressor[self.joints_name.index('L_Ring_1')] = np.eye(self.vertex_num)[2293]
        joint_regressor[self.joints_name.index('L_Pinky_1')] = np.eye(self.vertex_num)[2173]
        joint_regressor[self.joints_name.index('R_Index_1')] = np.eye(self.vertex_num)[5595]
        joint_regressor[self.joints_name.index('R_Middle_1')] = np.eye(self.vertex_num)[5675]
        joint_regressor[self.joints_name.index('R_Ring_1')] = np.eye(self.vertex_num)[5636]
        joint_regressor[self.joints_name.index('R_Pinky_1')] = np.eye(self.vertex_num)[5655]
        joint_regressor[self.joints_name.index('Nose')] = np.eye(self.vertex_num)[331]
        joint_regressor[self.joints_name.index('L_Eye')] = np.eye(self.vertex_num)[2802]
        joint_regressor[self.joints_name.index('R_Eye')] = np.eye(self.vertex_num)[6262]
        joint_regressor[self.joints_name.index('L_Ear')] = np.eye(self.vertex_num)[3489]
        joint_regressor[self.joints_name.index('R_Ear')] = np.eye(self.vertex_num)[3990]
        joint_regressor[self.joints_name.index('L_Big_toe')] = np.eye(self.vertex_num)[3292]
        joint_regressor[self.joints_name.index('L_Small_toe')] = np.eye(self.vertex_num)[3313]
        joint_regressor[self.joints_name.index('L_Heel')] = np.eye(self.vertex_num)[3468]
        joint_regressor[self.joints_name.index('R_Big_toe')] = np.eye(self.vertex_num)[6691]
        joint_regressor[self.joints_name.index('R_Small_toe')] = np.eye(self.vertex_num)[6713]
        joint_regressor[self.joints_name.index('R_Heel')] = np.eye(self.vertex_num)[6858]
        return joint_regressor


smpl_x = SMPLX()
smpl = SMPL()