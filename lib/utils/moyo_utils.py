import os
import re
import c3d
import cv2
import copy
import smplx
import trimesh
import pickle as pkl
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESSURE_MAP_FPS = 60
CONTACT_THRESH = 0.15
MOYO_DATA_DIR = 'data/MOYO/data'
ESSENTIALS_DIR = os.path.join(MOYO_DATA_PATH, 'essentials')
SMPLX_DIR = os.path.join(ESSENTIALS_DIR, 'yogi_segments', 'smplx')
SMPLX_PART_BOUNDS = os.path.join(SMPLX_DIR, 'part_meshes_ply/smplx_segments_bounds.pkl')
FID_TO_PART = os.path.join(SMPLX_DIR, 'part_meshes_ply/fid_to_part.pkl')
PART_VID_FID = os.path.join(SMPLX_DIR, 'part_meshes_ply/smplx_part_vid_fid.pkl')
HD_SMPLX_MAP  = os.path.join(ESSENTIALS_DIR, 'hd_model/smplx/smplx_neutral_hd_sample_from_mesh_out.pkl')
MOYO_V_TEMPLATE = os.path.join(MOYO_DATA_DIR, 'v_templates/220923_yogi_03596_minimal_simple_female/mesh.ply')


def build_hd_to_orig_sparse_map_mean(hd_operator_path):
    # Load regressor
    hd_operator = np.load(hd_operator_path)
    rows, cols = hd_operator['index_row_col']
    vals = hd_operator['values']
    size = hd_operator['size']
    num_hd = size[0]
    num_orig = size[1]

    # Build: For each HD vert, track which original verts it came from
    hd_to_orig = [[] for _ in range(num_orig)]
    for r, c in zip(rows, cols):
        hd_to_orig[c].append(r)

    # Invert: for each original vert, collect all contributing HD verts
    row_idx = []
    col_idx = []
    values = []
    for orig_idx, hd_indices in enumerate(hd_to_orig):
        if len(hd_indices) == 0:
            continue
        weight = 1.0 / len(hd_indices)
        for hd_idx in hd_indices:
            row_idx.append(orig_idx)
            col_idx.append(hd_idx)
            values.append(weight)

    # Create sparse matrix
    indices = torch.tensor([row_idx, col_idx], dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)
    size = torch.Size([num_orig, num_hd])
    M = torch.sparse.FloatTensor(indices, values, size)

    return M


def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor):
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(*batch_dim, 9), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([ 1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,  1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,],dim=-1,))
    quat_by_rijk = torch.stack([torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1), torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1), torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1), torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),],dim=-2,)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None])
    bla= quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(*batch_dim, 4)
    return bla


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def ea2rm(x, y, z):
    cos_x, sin_x = torch.cos(x), torch.sin(x)
    cos_y, sin_y = torch.cos(y), torch.sin(y)
    cos_z, sin_z = torch.cos(z), torch.sin(z)

    R = torch.stack(
            [torch.cat([cos_y*cos_z, sin_x*sin_y*cos_z - cos_x*sin_z, cos_x*sin_y*cos_z + sin_x*sin_z], dim=1),
            torch.cat([cos_y*sin_z, sin_x*sin_y*sin_z + cos_x*cos_z, cos_x*sin_y*sin_z - sin_x*cos_z], dim=1),
            torch.cat([-sin_y, sin_x*cos_y, cos_x*cos_y], dim=1)], dim=1)
    return R


def get_trans_offset(pelvis, smplx_params, trans, body_model):
    bs = trans.shape[0]
    init_root_orient = smplx_params['global_orient']
    pelvis_height = pelvis[:, 2]

    new_smplx_params = copy.deepcopy(smplx_params)
    R_init = axis_angle_to_matrix(init_root_orient)
    R1 = ea2rm(torch.tensor([[np.radians(270)]]), torch.tensor([[np.radians(0)]]),
               torch.tensor([[np.radians(0)]])).float().to(R_init.device)
    R = torch.bmm(R1.expand(bs, -1, -1), R_init)
    new_smplx_params['global_orient'] = matrix_to_axis_angle(R)

    body_model_output = body_model(
        global_orient=new_smplx_params['global_orient'],
        body_pose=new_smplx_params['body_pose'])

    new_pelvis = body_model_output.joints[:, 0]
    new_ground_plane_height = new_pelvis[:, 1] - pelvis_height
    trans_offset = -new_ground_plane_height
    return trans_offset


def smplx_breakdown(bdata, device):
    num_frames = len(bdata['trans'])

    bdata['poses'] = bdata['fullpose']

    global_orient = torch.from_numpy(bdata['poses'][:, :3]).float().to(device)
    body_pose = torch.from_numpy(bdata['poses'][:, 3:66]).float().to(device)
    jaw_pose = torch.from_numpy(bdata['poses'][:, 66:69]).float().to(device)
    leye_pose = torch.from_numpy(bdata['poses'][:, 69:72]).float().to(device)
    reye_pose = torch.from_numpy(bdata['poses'][:, 72:75]).float().to(device)
    left_hand_pose = torch.from_numpy(bdata['poses'][:, 75:120]).float().to(device)
    right_hand_pose = torch.from_numpy(bdata['poses'][:, 120:]).float().to(device)

    v_template = trimesh.load(MOYO_V_TEMPLATE, process=False)

    body_params = {'global_orient': global_orient, 'body_pose': body_pose,
                   'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                   'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                   'v_template': torch.Tensor(v_template.vertices).to(device), }
    return body_params


def smplx_to_mesh(body_params, model_folder, model_type, gender='neutral'):
    with torch.no_grad():
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        num_betas = 10

        smplx_params = smplx_breakdown(body_params, device)

        trans = torch.from_numpy(body_params['trans']).float().to(device)

        betas = torch.from_numpy(body_params['betas']).float().to(device).unsqueeze(0)
        betas = betas[:, :num_betas]

        body_model_params = dict(model_path=model_folder,
                                 model_type=model_type,
                                 gender=gender,
                                 v_template=smplx_params['v_template'],
                                 batch_size=trans.shape[0],
                                 create_global_orient=True,
                                 create_body_pose=True,
                                 create_betas=True,
                                 num_betas=num_betas,
                                 create_left_hand_pose=True,
                                 create_right_hand_pose=True,
                                 create_expression=True,
                                 create_jaw_pose=True,
                                 create_leye_pose=True,
                                 create_reye_pose=True,
                                 create_transl=True,
                                 use_pca=False,
                                 flat_hand_mean=True,
                                 dtype=torch.float32)

        body_model = smplx.create(**body_model_params).to(device)
        body_model_output = body_model(transl=trans,
                                       global_orient=smplx_params['global_orient'],
                                       body_pose=smplx_params['body_pose'],
                                       left_hand_pose=smplx_params['left_hand_pose'],
                                       right_hand_pose=smplx_params['right_hand_pose'])

        pelvis = body_model_output.joints[:, 0]
        trans_offset = get_trans_offset(pelvis, smplx_params, trans, body_model)
        zero_vec = torch.zeros_like(trans_offset)
        transl = torch.stack([zero_vec, trans_offset, zero_vec], dim=-1)

    return body_model_output, smplx_params, transl, body_model.faces


class PressureMat:
    def __init__(self, csv_path, xml_path, c3d_file):
        self.csv_path = csv_path
        self.xml_path = xml_path
        self.c3d_file = c3d_file

        self.marker_positions_metric = self.extract_pressure_mat_markers()

        self.gt_cops_relative, gt_pressure_frameids = self.process_csv()
        self.gt_pressures, self.gt_heatmaps, self.mat_size, self.mat_size_metric = self.process_xml()
        self.gt_pressures = self.gt_pressures[gt_pressure_frameids]
        self.gt_heatmaps = [self.gt_heatmaps[i] for i in gt_pressure_frameids]


    def parse_pressure_crops(self, pressure_crop):
        '''
            Takes raw pressure data from xml which is a list of strings for every frame containing pressure
            pressure_crops: N x 1 list of strings
            '''
        pressure_crop = [pressure_crop[i].get_text() for i in range(len(pressure_crop))]
        pressure_crop = [re.split(r'\n\t+', pressure_crop[i])[1:-1] for i in range(len(pressure_crop))]
        for i in range(len(pressure_crop)):
            for j in range(len(pressure_crop[i])):
                row = re.split(' ', pressure_crop[i][j])
                row = [float(el) for el in row]
                pressure_crop[i][j] = row
        return pressure_crop

    def unnormalize_pressure(self, pressure_crop, cell_begin, cell_end, mat_size, frame_count):
        '''
        Unnormalize pressure data to the full size of the pressure mat
        pressure_crop: N x 1 list of lists of lists (N frames, 1 pressure crop, pressure_crop[i] is a list of lists of floats (pressure values))
        cell_begin: N x 2 list of lists (N frames, 2 coordinates of the beginning of the pressure crop)
        cell_end: N x 2 list of lists (N frames, 2 coordinates of the end of the pressure crop)
        mat_size: 2 x 1 list (2 coordinates of the full size of the pressure mat)
        '''
        pressure_unnorm = np.zeros([frame_count, mat_size[1], mat_size[0]])
        for i in range(frame_count):
            pressure_crop_i = np.flipud(pressure_crop[i])
            pressure_unnorm[i, cell_begin[i][1]:cell_end[i][1],
            cell_begin[i][0]:cell_end[i][0]] = pressure_crop_i
        return pressure_unnorm

    def process_xml(self):
        with open(self.xml_path, 'r') as f:
            xml = f.read()
        xml_soup = bs(xml, 'lxml')
        mat_size = [int(xml_soup.movements.clips.clip.cell_count.x.get_text()),
                    int(xml_soup.movements.clips.clip.cell_count.y.get_text())]
        sensel_size = [float(xml_soup.movements.clips.clip.cell_size.x.get_text()),
                       float(xml_soup.movements.clips.clip.cell_size.y.get_text())]
        mat_size_metric = [mat_size[0] * sensel_size[0] / 1000,
                           mat_size[1] * sensel_size[1] / 1000]
        fps = float(xml_soup.movements.clips.clip.frequency.get_text())
        frame_count = int(xml_soup.movements.clips.clip.count.get_text())

        cell_begin_raw = xml_soup.movements.clips.clip.data.find_all('cell_begin')
        assert len(
            cell_begin_raw) == frame_count, 'Number of frames in xml file does not match the number of frames in the pressure data'
        cell_begin = [[int(cell_begin_raw[i].x.get_text()), int(cell_begin_raw[i].y.get_text())] for i in
                      range(len(cell_begin_raw))]
        cell_extent_raw = xml_soup.movements.clips.clip.data.find_all('cell_count')
        assert len(
            cell_extent_raw) == frame_count, 'Number of frames in xml file does not match the number of frames in the pressure data'
        cell_end = [[cell_begin[i][0] + int(cell_extent_raw[i].x.get_text()),
                     cell_begin[i][1] + int(cell_extent_raw[i].y.get_text())] for i in range(len(cell_extent_raw))]
        pressure_crop_raw = xml_soup.movements.clips.clip.data.find_all('cells')
        assert len(
            pressure_crop_raw) == frame_count, 'Number of frames in xml file does not match the number of frames in the pressure data'
        pressure_crop = self.parse_pressure_crops(pressure_crop_raw)
        pressure_unnorm = self.unnormalize_pressure(pressure_crop, cell_begin, cell_end, mat_size, frame_count)
        heatmaps = vis_heatmap_seq(pressure_unnorm, normalize=True)
        return pressure_unnorm, heatmaps, mat_size, mat_size_metric

    def process_csv(self):
        pressure_md = pd.read_csv(self.csv_path, header=[0], nrows=1)
        pressure_df = pd.read_csv(self.csv_path, header=[2])

        pressure_fps = pressure_md['frequency'].values[0]
        assert pressure_fps == PRESSURE_MAP_FPS, 'Pressure fps is not equal to default'
        pressure_frame_count = pressure_md['count'].values[0]

        cop_x = pressure_df['Pressure, Raw Pressure-distribution-Pressure center-x (mm)'].values / 1000  # mm to m
        cop_y = pressure_df['Pressure, Raw Pressure-distribution-Pressure center-y (mm)'].values / 1000
        cop = np.array([cop_x, cop_y]).T

        pressure_timestamps = pressure_df['time']
        pressure_timestamps = pressure_timestamps.values
        assert pressure_timestamps.shape[
                   0] == pressure_frame_count, 'Number of frames in md does not match the number of frames in the timestamp'

        pressure_fids = np.rint(pressure_timestamps * pressure_fps).astype(int)
        assert pressure_fids.shape[0] == pressure_fids[-1] - pressure_fids[
            0] + 1, 'Some frames are missing in the timestamps'
        return cop, pressure_fids


    def extract_pressure_mat_markers(self):
        with open(self.c3d_file, 'rb') as f:
            data = c3d.Reader(f)
            for frame in data.read_frames():
                frame = frame[1][:, :3]
                break

        marker_id_one = np.where(data.point_labels == 'SensorMat:M_1                                                   ')[0][0]
        marker_id_two = np.where(data.point_labels == 'SensorMat:M_2                                                   ')[0][0]
        marker_id_three = np.where(data.point_labels == 'SensorMat:M_3                                                   ')[0][0]
        marker_id_four = np.where(data.point_labels == 'SensorMat:M_4                                                   ')[0][0]
        marker_ids_unordered = [marker_id_one, marker_id_two, marker_id_three, marker_id_four]

        # Put the markers in the correct order
        for i, marker_id in enumerate(marker_ids_unordered):
            assert marker_id is not None, f'Marker {i} not found'
            if frame[marker_id, :][0] > 0 and frame[marker_id, :][1] > 0:
                marker_id_bl = marker_id
            if frame[marker_id, :][0] > 0 and frame[marker_id, :][1] < 0:
                marker_id_tl = marker_id
            if frame[marker_id, :][0] < 0 and frame[marker_id, :][1] < 0:
                marker_id_tr = marker_id
            if frame[marker_id, :][0] < 0 and frame[marker_id, :][1] > 0:
                marker_id_br = marker_id

        marker_ids = [marker_id_bl, marker_id_tl, marker_id_tr, marker_id_br]

        marker_positions = frame[marker_ids, :]
        return marker_positions


def vis_heatmap_seq(pressure, normalize=True):
    heatmaps = []
    for i in range(len(pressure)):
        if normalize:
            pressure_i = (pressure[i] - np.min(pressure[i])) / (np.max(pressure[i]) - np.min(pressure[i]))
        else:
            pressure_i = pressure[i]
        heatmap = cv2.applyColorMap(np.uint8(255 * pressure_i), cv2.COLORMAP_JET)
        heatmaps.append(heatmap)
    return heatmaps


def vis_heatmap(pressure, normalize=True):
    if normalize:
        pressure = (pressure - np.min(pressure)) / (np.max(pressure) - np.min(pressure))
    heatmap = cv2.applyColorMap(np.uint8(255 * pressure), cv2.COLORMAP_JET)
    return heatmap


class MetricsCollector:
    def __init__(self):
        self.ious = []
        self.cop_errors = []
        self.frame_diffs = []
        self.binary_ths = []
        self.best_cop_w = []
        self.best_cop_k = []
        self.pred_heatmaps = []
        self.pred_cops_relative = []
        self.contact_smplx = []

    def assign(self, *, iou, cop_error, frame_diff, binary_th, best_cop_w, best_cop_k, pred_heatmap, pred_cop_relative, contact_smplx):
        self.ious.append(iou)
        self.cop_errors.append(cop_error)
        self.frame_diffs.append(frame_diff)
        self.binary_ths.append(binary_th)
        self.best_cop_w.append(best_cop_w)
        self.best_cop_k.append(best_cop_k)
        self.pred_heatmaps.append(pred_heatmap)
        self.pred_cops_relative.append(pred_cop_relative)
        self.contact_smplx.append(contact_smplx)


def sparse_batch_mm(m1, m2):
    batch_size = m2.shape[0]
    m2_stack = m2.transpose(0, 1).reshape(m1.shape[1], -1)
    result = m1.mm(m2_stack).reshape(m1.shape[0], batch_size, -1) \
        .transpose(1, 0)
    return result


class HDfier():
    def __init__(self, model_type='smplx'):
        hd_operator_path = os.path.join(ESSENTIALS_DIR, 'hd_model', model_type,
                                    f'{model_type}_neutral_hd_vert_regressor_sparse.npz')
        hd_operator = np.load(hd_operator_path)
        self.hd_operator = torch.sparse.FloatTensor(
            torch.tensor(hd_operator['index_row_col']),
            torch.tensor(hd_operator['values']),
            torch.Size(hd_operator['size']))

    def hdfy_mesh(self, vertices, model_type='smplx'):
        """
        Applies a regressor that maps SMPL vertices to uniformly distributed vertices
        """
        # device = body.vertices.device
        # check if vertices ndim are 3, if not , add a new axis
        if vertices.ndim != 3:
            # batchify the vertices
            vertices = vertices[None, :, :]

        # check if vertices are an ndarry, if yes, make pytorch tensor
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(self.device)

        vertices = vertices.to(torch.double)

        if self.hd_operator.device != vertices.device:
            self.hd_operator = self.hd_operator.to(vertices.device)
        hd_verts = sparse_batch_mm(self.hd_operator, vertices).to(torch.float)
        return hd_verts


class SMPLXMesh(nn.Module):
    def __init__(self, vertices, faces):
        super(SMPLXMesh, self).__init__()

        self.vertices = vertices
        self.faces = faces


class PartVolume(nn.Module):
    def __init__(self,
                 part_name,
                 vertices,
                 faces):
        super(PartVolume, self).__init__()

        self.part_name = part_name
        self.smplx_mesh = SMPLXMesh(vertices, faces)

        self.part_triangles = None
        self.device = vertices.device

        self.new_vert_ids = []
        self.new_face_ids = []

    def close_mesh(self, boundary_vids):
        # find the center of the boundary
        mean_vert = self.smplx_mesh.vertices[:, boundary_vids, :].mean(dim=1, keepdim=True)
        self.smplx_mesh.vertices = torch.cat([self.smplx_mesh.vertices, mean_vert], dim=1)
        new_vert_idx = self.smplx_mesh.vertices.shape[1]-1
        self.new_vert_ids.append(new_vert_idx)
        # add faces
        new_faces = [[boundary_vids[i + 1], boundary_vids[i], new_vert_idx] for i in range(len(boundary_vids) - 1)]
        self.smplx_mesh.faces = torch.cat([self.smplx_mesh.faces.to(self.device), torch.tensor(new_faces, dtype=torch.long, device=self.device)], dim=0)
        self.new_face_ids += list(range(self.smplx_mesh.faces.shape[0]-len(new_faces), self.smplx_mesh.faces.shape[0]))

    def extract_part_triangles(self, part_vids, part_fid):
        batch_size = self.smplx_mesh.vertices.shape[0]

        part_vertices = self.smplx_mesh.vertices[:, part_vids, :]
        part_faces = self.smplx_mesh.faces[part_fid, :]

        part_mean = part_vertices.mean(dim=1, keepdim=True)

        self.smplx_mesh.vertices = self.smplx_mesh.vertices - part_mean

        if self.part_triangles is None:
            self.part_triangles = torch.index_select(self.smplx_mesh.vertices, 1, part_faces.view(-1)).reshape(batch_size, -1, 3, 3)
        else:
            self.part_triangles = torch.cat([self.part_triangles,
                                             torch.index_select(self.smplx_mesh.vertices, 1,
                                                     part_faces.view(-1)).reshape(batch_size, -1, 3, 3)], dim=1)
        # add back vert mean
        self.smplx_mesh.vertices = self.smplx_mesh.vertices + part_mean

    def part_volume(self):
        # Note: the mesh should be enclosing the origin (mean-subtracted)
        # compute volume of triangles by drawing tetrahedrons
        # https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up
        x = self.part_triangles[:, :, :, 0]
        y = self.part_triangles[:, :, :, 1]
        z = self.part_triangles[:, :, :, 2]
        volume = (
                         -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
                         x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
                         x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
                         x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
                         x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
                         x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
                 ).sum(dim=1).abs() / 6.0
        return volume


def reduce_hd_contact_to_smplx(contact_hd, faces_vert_is_sampled_from, reduce='max', fill_value=0.0):
    """
    Convert 20k HD contact to 10475 SMPLX mesh contact.
    Handles unmapped vertices safely.

    Args:
        contact_hd: (B, 20000) tensor
        faces_vert_is_sampled_from: list of 20000 ints
        reduce: 'max' or 'mean'
        fill_value: default value to assign for unmapped verts

    Returns:
        contact_smplx: (B, 10475) tensor
    """
    device = contact_hd.device
    num_hd_verts = len(faces_vert_is_sampled_from)
    num_smplx_verts = max(faces_vert_is_sampled_from) + 1

    mapping = [[] for _ in range(num_smplx_verts)]
    for hd_vid, smplx_vid in enumerate(faces_vert_is_sampled_from):
        mapping[smplx_vid].append(hd_vid)

    contact_smplx = []
    for smplx_vids in mapping:
        if len(smplx_vids) == 0:
            contact_smplx.append(torch.full((contact_hd.shape[0],), fill_value, device=device))
        else:
            hd_vals = contact_hd[:, smplx_vids]
            if reduce == 'max':
                contact_smplx.append(torch.max(hd_vals, dim=1).values)
            elif reduce == 'mean':
                contact_smplx.append(torch.mean(hd_vals, dim=1))
            else:
                raise ValueError("Unsupported reduction type")
    contact_smplx = torch.stack(contact_smplx, dim=1)
    return contact_smplx


class StabilityLossCoS(nn.Module):
    def __init__(self,
                 faces,
                 cos_w = 10,
                 cos_k = 100,
                 contact_thresh=CONTACT_THRESH,
                 model_type='smplx',
    ):
        super().__init__()
        """
        Loss that ensures that the COM of the SMPL mesh is close to the center of support 
        """
        if model_type == 'smplx':
            num_faces = 20908
        if model_type == 'smpl':
            num_faces = 13776
        num_verts_hd = 20000


        assert faces is not None, 'Faces tensor is none'
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long)
        self.register_buffer('faces', faces)

        self.cos_w = cos_w
        self.cos_k = cos_k
        self.contact_thresh = contact_thresh

        self.hdfy_op = HDfier()

        with open(SMPLX_PART_BOUNDS, 'rb') as f:
            d = pkl.load(f)
            self.part_bounds = {k: d[k] for k in sorted(d)}
        self.part_order = sorted(self.part_bounds)

        with open(PART_VID_FID, 'rb') as f:
            self.part_vid_fid = pkl.load(f)

        # mapping between vid_hd and fid
        with open(HD_SMPLX_MAP, 'rb') as f:
            self.faces_vert_is_sampled_from = pkl.load(f)['faces_vert_is_sampled_from']
        index_row_col = torch.stack(
            [torch.LongTensor(np.arange(0, num_verts_hd)), torch.LongTensor(self.faces_vert_is_sampled_from)], dim=0)
        values = torch.ones(num_verts_hd, dtype=torch.float)
        size = torch.Size([num_verts_hd, num_faces])
        hd_vert_on_fid = torch.sparse.FloatTensor(index_row_col, values, size)

        # mapping between fid and part label
        with open(FID_TO_PART, 'rb') as f:
            fid_to_part_dict = pkl.load(f)
        fid_to_part = torch.zeros([len(fid_to_part_dict.keys()), len(self.part_order)], dtype=torch.float32)
        for fid, partname in fid_to_part_dict.items():
            part_idx = self.part_order.index(partname)
            fid_to_part[fid, part_idx] = 1.

        # mapping between vid_hd and part label
        self.hd_vid_in_part = self.vertex_id_to_part_mapping(hd_vert_on_fid, fid_to_part)

    def compute_triangle_area(self, triangles):
        vectors = torch.diff(triangles, dim=2)
        crosses = torch.cross(vectors[:, :, 0], vectors[:, :, 1])
        area = torch.norm(crosses, dim=2) / 2
        return area

    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(vertices.device)
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]['face_id']).to(vertices.device)
            pv = PartVolume(part_name, vertices, self.faces)
            for bound_name, bound_vids in part_bounds.items():
                pv.close_mesh(bound_vids)
            # add extra vids and fids to original part ids
            new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
            new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
            part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
            part_fid = torch.cat((part_fid, new_face_ids), dim=0)
            pv.extract_part_triangles(part_vid, part_fid)
            part_volume.append(pv.part_volume())
        return torch.vstack(part_volume).permute(1,0).to(vertices.device)

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol

    def vertex_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part

    def forward(self, vertices):
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)

        # pressure based center of support
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 1] - ground_plane_height)
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()
        pressure_weights = inside_mask * (1-self.cos_k*vertex_height) + outside_mask *  torch.exp(-self.cos_w * vertex_height)
        cos = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (torch.sum(pressure_weights, dim=1, keepdim=True) +eps)

        contact_confidence = torch.sum(pressure_weights, dim=1)
        contact_mask = (vertex_height < self.contact_thresh).float()

        # project com, cos to ground plane (x-z plane)
        # weight loss by number of contact vertices to zero out if zero vertices in contact
        com_xz = torch.stack([com[:, 0], torch.zeros_like(com)[:, 0], com[:, 2]], dim=1)
        contact_centroid_xz = torch.stack([cos[:, 0], torch.zeros_like(cos)[:, 0], cos[:, 2]], dim=1)
        stability_loss = (torch.norm(com_xz - contact_centroid_xz, dim=1))
        return stability_loss



class BiomechanicalEvaluator(StabilityLossCoS):
    def __init__(self,
                 faces,
                 cop_w=10,
                 cop_k=100,
                 contact_thresh=CONTACT_THRESH,
                 model_type='smplx',
                 ):
        super().__init__(faces, cop_w, cop_k, contact_thresh, model_type)

        self.iou_first = MetricsCollector()
        self.cop_error_first = MetricsCollector()
        self.frame_diff_first = MetricsCollector()

        self.cop_k_range = np.linspace(10, 200, 10).astype(np.float64)
        self.cop_w_range = np.linspace(10, 5000, 15).astype(np.float64)

    def generate_aligned_heatmap(self, vertices, mat_size, marker_positions,
                                 ground_plane_height=0.0, heatmap_res=512, cop_w=10, cop_k=100, vp=0.1,
                                 com_type='volume', cop_type='pressure', debug=False):
        """
        get vertices height along *z-axis* from ground-plane and pass it through function e(-wx) to get values for the heatmap.
        align and crop the heatmap so that it is it matches the Gt pressure map
        Args:
            mat_size: size of the pressure mat
            marker_positions: bottom left, top left, top right, bottom right
            cop_w: paramaeter for contact function
            vp: view padding empty space around contact region (in m)
        """
        # map 4 x 4 m area to a 512x512 image
        pressure_map = np.zeros((mat_size[1], mat_size[0]), dtype=np.float32)
        pressure_map_smplx = np.zeros((mat_size[1], mat_size[0]), dtype=np.float32)
        heatmap_point = np.zeros((heatmap_res, heatmap_res), dtype=np.float32)

        # uniformly sample vertices from the SMPL mesh so that the contact vertices are not biased to hands and face
        vertices = torch.tensor(vertices)[None, :, :]
        vertices_smplx = vertices[0, 0].detach().cpu().numpy()
        vertices_hd = HDfier().hdfy_mesh(vertices).cpu().numpy()[0]
        vertices = vertices.cpu().numpy()[0]

        # calculate values for heatmap
        vertex_height = vertices_hd[:, 2] - ground_plane_height
        vertex_height = vertex_height.astype(np.float64)

        vertices_smplx_height = vertices_smplx[:, 2] - ground_plane_height
        vertices_smplx_height = vertices_smplx_height.astype(np.float64)

        # Get metric range. x-z plane is ground
        mat_bl_corner = marker_positions[0, :2] / 1000
        mat_tl_corner = marker_positions[1, :2] / 1000
        mat_tr_corner = marker_positions[2, :2] / 1000
        mat_br_corner = marker_positions[3, :2] / 1000
        m_x = mat_br_corner[0], mat_bl_corner[0]
        m_y = mat_tr_corner[1], mat_br_corner[1]
        m_range_x = m_x[1] - m_x[0]
        m_range_y = m_y[1] - m_y[0]

        # filter out vertices outside of the mat
        mask = (vertices_hd[:, 0] >= m_x[0]) & (vertices_hd[:, 0] <= m_x[1]) & (vertices_hd[:, 1] >= m_y[0]) & (
                vertices_hd[:, 1] <= m_y[1])
        vertex_height = vertex_height[mask]
        vertices_hd = vertices_hd[mask]

        mask_smplx = (vertices_smplx[:, 0] >= m_x[0]) & (vertices_smplx[:, 0] <= m_x[1]) & (vertices_smplx[:, 1] >= m_y[0]) & (
                vertices_smplx[:, 1] <= m_y[1])
        vertices_smplx_height = vertices_smplx_height[mask_smplx]
        vertices_smplx = vertices_smplx[mask_smplx]

        # Normalize metric values to image range.
        max_mat_idx_x = mat_size[0] - 1
        max_mat_idx_y = mat_size[1] - 1
        v_x = np.rint((m_x[1] - vertices_hd[:, 0]) / m_range_x * (max_mat_idx_x)).astype(int) # flipped because global x axis is opposite of pressure mat x-axis
        v_y = np.rint((m_y[1] - vertices_hd[:, 1]) / m_range_y * (max_mat_idx_y)).astype(
            int) # flipped because mat goes from bottom to up while image goes from top to bottom

        v_x_smplx = np.rint((m_x[1] - vertices_smplx[:, 0]) / m_range_x * (max_mat_idx_x)).astype(int) # flipped because global x axis is opposite of pressure mat x-axis
        v_y_smplx = np.rint((m_y[1] - vertices_smplx[:, 1]) / m_range_y * (max_mat_idx_y)).astype(
            int) # flipped because mat goes from bottom to up while image goes from top to bottom

        # assymetric function for inside and outside vertices
        inside_mask = (vertex_height < 0)
        outside_mask = (vertex_height >= 0)
        v_z = inside_mask * (1 - cop_k * vertex_height) + outside_mask * np.exp(-cop_w * vertex_height)

        inside_mask_smplx = (vertices_smplx_height < 0)
        outside_mask_smplx = (vertices_smplx_height >= 0)
        v_z_smplx = inside_mask_smplx * (1 - cop_k * vertices_smplx_height) + outside_mask_smplx * np.exp(-cop_w * vertices_smplx_height)

        # normalize v_z to [0, 1]
        v_z = (v_z - np.min(v_z)) / (np.max(v_z) - np.min(v_z))
        v_z_smplx = (v_z_smplx - np.min(v_z_smplx)) / (np.max(v_z_smplx) - np.min(v_z_smplx))

        # foe each overlapping vertex, add the value to the heatmap
        for i in range(len(v_x)):
            pressure_map[v_y[i], v_x[i]] += v_z[i]
        for i in range(len(v_x_smplx)):
            pressure_map_smplx[v_y_smplx[i], v_x_smplx[i]] += v_z_smplx[i]
        # normalize heatmap to [0, 1]
        pressure_map = (pressure_map - np.min(pressure_map)) / (np.max(pressure_map) - np.min(pressure_map))
        pressure_map_smplx = (pressure_map_smplx - np.min(pressure_map_smplx)) / (np.max(pressure_map_smplx) - np.min(pressure_map_smplx))

        # flip the pressure_map because the GT is inverted
        # pressure_map = np.flipud(pressure_map)
        heatmap = vis_heatmap(pressure_map)
        heatmap_smplx = vis_heatmap(pressure_map_smplx)
        import pdb; pdb.set_trace()
        return pressure_map, heatmap

    def iou(self, gt_pressure, pred_pressure):
        """
        Intersection over union
        :return:
        """
        binary_threholds = np.linspace(0, 1, 10)
        ious = []
        for th in binary_threholds:
            pred_binary = pred_pressure > th
            gt_binary = gt_pressure > th
            intersection = np.logical_and(pred_binary, gt_binary)
            union = np.logical_or(pred_binary, gt_binary)
            eps = 1e-13
            iou_score = np.sum(intersection) / (np.sum(union) + eps)
            ious.append(iou_score)
        max_iou_score = np.max(ious)
        max_th_idx = np.argmax(ious)
        max_th = binary_threholds[max_th_idx]
        return max_iou_score, max_th

    def draw_ious_graph(self):
        from numpy import exp, arange
        from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show

        # the function that I'm going to plot
        x = arange(-3.0, 3.0, 0.1)
        y = arange(-3.0, 3.0, 0.1)
        X, Y = meshgrid(x, y) # grid of point
        Z = z_func(X, Y) # evaluation of the function on the grid

        im = imshow(Z, cmap=cm.RdBu) # drawing the function
        # adding the Contour lines with labels
        cset = contour(Z, arange(-1, 1.5, 0.2), linewidths=2, cmap=cm.Set2)
        clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        colorbar(im) # adding the colobar on the right
        # latex fashion title
        title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
        show()

    def evaluate_com(self, gt_com, vertices):
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        vertices = vertices.float()
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)
        com_error = torch.norm(com - gt_com, dim=1)
        return com, com_error

    def evaluate_pressure(self, gt_pressure, gt_cop_relative, vertices, mat_size, mat_size_global, marker_positions):
        """
        Evaluate the predicted pressure map against the ground truth pressure map.
        Args:
            gt_pressure:
            gt_cop_relative: cop relative to the pressure mat in mms
            vertices:
            mat_size: the resolution of the mat image
            mat_bbox: [tl_x, tl_y, br_x, br_y] of the mat in the image

        Returns:

        """
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        vertices = vertices.float()
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)

        # pressure based center of support
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 2] - ground_plane_height).double()
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()

        iou_first_best_iou = 0.0 # best iou score
        cop_error_first_best_cop_error = np.inf # best cop error
        frame_diff_first_best_frame_diff = np.inf
        for cop_k in self.cop_k_range:
            for cop_w in self.cop_w_range:
                pressure_weights = inside_mask * (1 - cop_k * vertex_height) + outside_mask * torch.exp(-cop_w * vertex_height)
                pred_cos_global = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (torch.sum(pressure_weights, dim=1, keepdim=True) + eps)

                ## convert pred_cos relative to the pressure mat
                # Get metric range. x-z plane is ground
                mat_bl_corner = marker_positions[0, :2] / 1000
                mat_tl_corner = marker_positions[1, :2] / 1000
                mat_tr_corner = marker_positions[2, :2] / 1000
                mat_br_corner = marker_positions[3, :2] / 1000
                m_x = mat_br_corner[0], mat_bl_corner[0]
                m_y = mat_tr_corner[1], mat_br_corner[1]
                m_range_x = m_x[1] - m_x[0]
                m_range_y = m_y[1] - m_y[0]

                # convert all vertices relative to the pressure mat
                mat_size_global_x = mat_size_global[0]
                mat_size_global_y = mat_size_global[1]
                v_x_relative = (m_x[1] - vertices_hd[:, :, 0]) / m_range_x * (mat_size_global_x) # flipped because global x axis is opposite of pressure mat x-axis
                v_y_relative = (m_y[1] - vertices_hd[:, :, 1]) / m_range_y * (mat_size_global_y) # flipped because mat goes from bottom to up while image goes from top to bottom
                vertices_hd_relative = torch.stack([v_x_relative, v_y_relative, vertices_hd[:, :, 2]], dim=2)
                pred_cos_relative = torch.sum(vertices_hd_relative * pressure_weights.unsqueeze(-1), dim=1) / (
                        torch.sum(pressure_weights, dim=1, keepdim=True) + eps)
                pred_cos_relative = pred_cos_relative[0, :2].cpu().numpy()

                # compute l2 distance between gt and pred cop
                cop_error = np.linalg.norm(pred_cos_relative - gt_cop_relative, axis=0) # in metres

                # get pressure heatmap
                pred_pressure, pred_heatmap = self.generate_aligned_heatmap(vertices, mat_size, marker_positions,
                                                                            cop_w=cop_w, cop_k=cop_k)

                # calculate intersection-over-union between gt_pressure and predicted pressure map
                iou, binary_th = self.iou(gt_pressure, pred_pressure)
                # calculate mean frame diff
                frame_diff = np.mean(np.abs(gt_pressure - pred_pressure))
                if frame_diff <= frame_diff_first_best_frame_diff:
                    frame_diff_first_best_iou = iou
                    frame_diff_first_best_th = binary_th
                    frame_diff_first_best_cop_error = cop_error
                    frame_diff_first_best_frame_diff = frame_diff
                    frame_diff_first_best_pred_heatmap = pred_heatmap
                    frame_diff_first_best_pred_cop_relative = pred_cos_relative
                    frame_diff_first_best_cop_w = cop_w
                    frame_diff_first_best_cop_k = cop_k
                if iou >= iou_first_best_iou:
                    iou_first_best_iou = iou
                    iou_first_best_cop_error = cop_error
                    iou_first_best_frame_diff = frame_diff
                    iou_first_best_th = binary_th
                    iou_first_best_pred_heatmap = pred_heatmap
                    iou_first_best_pred_cop_relative = pred_cos_relative
                    iou_first_best_cop_w = cop_w
                    iou_first_best_cop_k = cop_k
                if cop_error <= cop_error_first_best_cop_error:
                    cop_error_first_best_iou = iou
                    cop_error_first_best_cop_error = cop_error
                    cop_error_first_best_frame_diff = frame_diff
                    cop_error_first_best_th = binary_th
                    cop_error_first_best_pred_heatmap = pred_heatmap
                    cop_error_first_best_pred_cop_relative = pred_cos_relative
                    cop_error_first_best_cop_w = cop_w
                    cop_error_first_best_cop_k = cop_k

        pressure_weights_best = inside_mask * (1 - frame_diff_first_best_cop_k * vertex_height) + outside_mask * torch.exp(-frame_diff_first_best_cop_w * vertex_height)
        mapping_hd_to_smplx = build_hd_to_orig_sparse_map_mean(os.path.join(ESSENTIALS_DIR, 'hd_model', 'smplx', f'smplx_neutral_hd_vert_regressor_sparse.npz')).to(pressure_weights_best.device)
        contact_smplx = torch.matmul(mapping_hd_to_smplx, pressure_weights_best.float().transpose(0, 1)).transpose(0, 1)

        self.frame_diff_first.assign(iou=frame_diff_first_best_iou,
                                     cop_error=frame_diff_first_best_cop_error,
                                     frame_diff=frame_diff_first_best_frame_diff,
                                     binary_th=frame_diff_first_best_th,
                                     best_cop_w=frame_diff_first_best_cop_w,
                                     best_cop_k=frame_diff_first_best_cop_k,
                                     pred_heatmap=frame_diff_first_best_pred_heatmap,
                                     pred_cop_relative=frame_diff_first_best_pred_cop_relative,
                                     contact_smplx=contact_smplx)

        self.iou_first.assign(iou=iou_first_best_iou,
                                      cop_error=iou_first_best_cop_error,
                                      frame_diff = iou_first_best_frame_diff,
                                      binary_th=iou_first_best_th,
                                        best_cop_w=iou_first_best_cop_w,
                                        best_cop_k=iou_first_best_cop_k,
                                      pred_heatmap=iou_first_best_pred_heatmap,
                                      pred_cop_relative=iou_first_best_pred_cop_relative,
                                      contact_smplx=contact_smplx)
        self.cop_error_first.assign(iou=cop_error_first_best_iou,
                                            cop_error=cop_error_first_best_cop_error,
                                            frame_diff=cop_error_first_best_frame_diff,
                                            binary_th=cop_error_first_best_th,
                                            best_cop_w=cop_error_first_best_cop_w,
                                            best_cop_k=cop_error_first_best_cop_k,
                                            pred_heatmap=cop_error_first_best_pred_heatmap,
                                            pred_cop_relative=cop_error_first_best_pred_cop_relative,
                                            contact_smplx=contact_smplx)