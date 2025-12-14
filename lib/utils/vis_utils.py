import os
import cv2
import json
import torch
import trimesh
import pyrender
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"

from lib.core.config import cfg
from lib.utils.human_models import smpl_x
from lib.utils.mesh_utils import extract_partial_mesh, build_new_faces


watertight_foot_faces = np.array([[  2,   1, 107],
       [142, 105, 106],
       [108, 107, 106],
       [107,   1, 106],
       [106,   1, 142],
       [  6, 105, 142],
       [  0,   1,   2],
       [  2, 107, 109],
       [109, 107, 108],
       [  8,   7,   5],
       [  8,   5,   6],
       [  6, 142,   8],
       [  4,   2, 109],
       [  4,   3,   0],
       [  0,   2,   4],
       [108,   3,   4],
       [109, 108,   4]])


# This function is modified from the function of DECO (https://github.com/sha2nkt/deco/blob/main/inference.py)
class ContactRenderer():
    def __init__(self):
        self.default_mesh_color = [130, 130, 130, 255]
        self.contact_mesh_color = [0, 255, 0, 255]

        with torch.no_grad():
            smpl_x_rest_out = smpl_x.layer(
                betas=torch.zeros(1, 10),
                global_orient=torch.zeros(1, 3),                     # [1, 3]
                body_pose=torch.zeros(1, 21 * 3),                    # [1, 63]
                left_hand_pose=torch.zeros(1, 15 * 3),               # [1, 45]
                right_hand_pose=torch.zeros(1, 15 * 3),              # [1, 45]
                jaw_pose=torch.zeros(1, 3),
                leye_pose=torch.zeros(1, 3),
                reye_pose=torch.zeros(1, 3),
                expression=torch.zeros(1, 10),
                transl=torch.zeros(1, 3),
                return_verts=True
            )

            smpl_x_j_regressor = smpl_x.layer.J_regressor.numpy()
            right_foot_ankle_joint_idx = 8

            self.body_model_smpl_x = trimesh.Trimesh(smpl_x_rest_out.vertices[0], smpl_x.face)
            self.body_model_smpl_x.vertices = self.body_model_smpl_x.vertices - (smpl_x_j_regressor @ self.body_model_smpl_x.vertices)[right_foot_ankle_joint_idx]

            with open('data/base_data/conversions/smplx_vert_segmentation.json', 'r') as f:
                self.smplx_part_seg = json.load(f)
                self.smplx_rightfoot_idxs = self.smplx_part_seg['rightFoot']
                self.smplx_righttoe_idxs = self.smplx_part_seg['rightToeBase']
                self.smplx_rightleg_idxs = self.smplx_part_seg['rightLeg']
                self.smplx_rightfoot_full_idxs = np.array(self.smplx_rightfoot_idxs + self.smplx_righttoe_idxs)

            self.mesh_foot_r = extract_partial_mesh(self.body_model_smpl_x, self.smplx_rightfoot_full_idxs)

            # import pdb; pdb.set_trace()

            self.mesh_foot_r.vertices = self.mesh_foot_r.vertices - self.mesh_foot_r.vertices.mean(axis=0)
            self.mesh_foot_r.vertices = self.mesh_foot_r.vertices * 0.7

            self.mesh_foot_r.vertices[:, 1] = self.mesh_foot_r.vertices[:, 1] - 0.03
            self.mesh_foot_r.vertices[:, 2] = self.mesh_foot_r.vertices[:, 2] + 0.01




            # self.mesh_foot_r.vertices[:, 2] = self.mesh_foot_r.vertices[:, 2] + 10

            # Get open vertices
            # faces = build_new_faces(
            #     mesh_vertices=self.mesh_foot_r.vertices,
            #     source_indices=self.smplx_rightfoot_full_idxs,
            #     target_indices=self.smplx_rightleg_idxs
            # )
            
            self.mesh_foot_r.faces = np.concatenate((self.mesh_foot_r.faces, watertight_foot_faces), axis=0)

            # _ = self.mesh_foot_r.export('mesh_foot_r.obj')
            # import pdb; pdb.set_trace()
            # print('debug')

    def render_image(self, scene, img_res, img=None, viewer=False):
        r = pyrender.OffscreenRenderer(viewport_width=img_res, viewport_height=img_res, point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if img is not None:
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
        else:
            output_img = color
        return output_img

    def create_scene(self, mesh, img, focal_length=5000, camera_center=250, img_res=500):
        # Setup the scene
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                            ambient_light=(0.3, 0.3, 0.3))
        # add mesh for camera
        camera_pose = np.eye(4)
        camera_rotation = np.eye(3, 3)
        camera_translation = np.array([0., 0, 2.5])
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = camera_rotation @ camera_translation

        pyrencamera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center, cy=camera_center)
        scene.add(pyrencamera, pose=camera_pose)

        # create and add light
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
            light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
            # out_mesh.vertices.mean(0) + np.array(lp)
            scene.add(light, pose=light_pose)

        # add body mesh
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        mesh_images = []

        # resize input image to fit the mesh image height
        img_height = img_res
        img_width = int(img_height * img.shape[1] / img.shape[0])
        img = cv2.resize(img, (img_width, img_height))
        mesh_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for sideview_angle in [0, 90, 180, 270]:
            out_mesh = mesh.copy()
            rot = trimesh.transformations.rotation_matrix(
                np.radians(sideview_angle), [0, 1, 0])
            out_mesh.apply_transform(rot)
            out_mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material)
            mesh_pose = np.eye(4)
            scene.add(out_mesh, pose=mesh_pose, name='mesh')

            output_img = self.render_image(scene, img_res)
            output_img = (output_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2RGB)
            mesh_images.append(output_img)

            # delete the previous mesh
            prev_mesh = scene.get_nodes(name='mesh').pop()
            scene.remove_node(prev_mesh)

        # show upside down view
        for topview_angle in [90, 270]:
            out_mesh = mesh.copy()
            rot = trimesh.transformations.rotation_matrix(
                np.radians(topview_angle), [1, 0, 0])
            out_mesh.apply_transform(rot)
            out_mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material)
            mesh_pose = np.eye(4)
            scene.add(out_mesh, pose=mesh_pose, name='mesh')

            output_img = self.render_image(scene, img_res)
            output_img = (output_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2RGB)
            mesh_images.append(output_img)

            # delete the previous mesh
            prev_mesh = scene.get_nodes(name='mesh').pop()
            scene.remove_node(prev_mesh)

        # stack images
        IMG = np.hstack(mesh_images)
        return IMG

    def create_scene_demo(self, mesh, img, focal_length=5000, camera_center=250, img_res=500):
        # Setup the scene
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                            ambient_light=(0.3, 0.3, 0.3))
        
        # Camera
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0., 0, 2.5])
        pyrencamera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center, cy=camera_center)
        scene.add(pyrencamera, pose=camera_pose)

        # Lighting
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
            light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
            scene.add(light, pose=light_pose)

        # Material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        mesh_images = []

        # Resize input image
        img_height = img_res
        img_width = int(img_height * img.shape[1] / img.shape[0])
        img = cv2.resize(img, (img_width, img_height))
        mesh_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Top views only (X-axis rotations), then rotate 90° clockwise
        for topview_angle in [90, 270]:
            out_mesh = mesh.copy()

            # Rotate around X-axis
            rot = trimesh.transformations.rotation_matrix(
                np.radians(topview_angle), [1, 0, 0])
            out_mesh.apply_transform(rot)

            # Move mesh to the right (positive X-axis) and assign label
            if topview_angle == 90:
                right_shift = np.array([-0.02, 0.03, 0.0])  # Dorsal view
                label = "Dorsal"
            elif topview_angle == 270:
                right_shift = np.array([-0.02, -0.025, 0.0])  # Palmar view
                label = "Palmar"
            out_mesh.apply_translation(right_shift)

            # Create pyrender mesh and add to scene
            mesh_node = pyrender.Mesh.from_trimesh(out_mesh, material=material)
            mesh_pose = np.eye(4)
            scene.add(mesh_node, pose=mesh_pose, name='mesh')

            # Render the scene
            output_img = self.render_image(scene, img_res)
            output_img = (output_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2RGB)

            # Rotate 90 degrees clockwise
            output_img = cv2.rotate(output_img, cv2.ROTATE_90_CLOCKWISE)

            # Write label directly on the image (bottom center)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            if topview_angle == 90:
                text_x_move = 44
            elif topview_angle == 270:
                text_x_move = -34
            text_x = (output_img.shape[1] - text_size[0]) // 2 + text_x_move
            text_y = output_img.shape[0] - 25  # 10px above bottom
            cv2.putText(output_img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            mesh_images.append(output_img)

            # Remove the mesh node
            scene.remove_node(scene.get_nodes(name='mesh').pop())

        # Stack images horizontally
        IMG = np.hstack(mesh_images)
        return IMG

    def render_contact(self, img, contact, mode='test'):   
        vis_contact = contact == 1.

        for vert in range(self.mesh_foot_r.visual.vertex_colors.shape[0]):
            self.mesh_foot_r.visual.vertex_colors[vert] = self.default_mesh_color
        self.mesh_foot_r.visual.vertex_colors[vis_contact] = self.contact_mesh_color

        img = cv2.resize(img.copy(), cfg.MODEL.input_img_shape, cv2.INTER_CUBIC)

        if mode == 'demo':
            rend = self.create_scene_demo(self.mesh_foot_r, img[..., ::-1].astype(np.uint8))
        else:
            rend = self.create_scene(self.mesh_foot_r, img[..., ::-1].astype(np.uint8))
        return rend

    def render_multiview_partseg(self, part_dict, img_res=500, colormap=cv2.COLORMAP_SUMMER, annotate=False):
        """
        Render self.mesh_foot_r with per-part vertex colors in multiple views.
        Returns an RGB image (H, W, 3) concatenating [front, right, back, left, top, bottom].
        """
        import numpy as np
        import cv2
        import pyrender
        import trimesh

        # 1) Color by vertex labels so pyrender respects colors
        mesh = self.mesh_foot_r.copy()
        nv = mesh.vertices.shape[0]

        vlabels = -np.ones(nv, dtype=np.int32)
        for i, idxs in enumerate(part_dict.values()):
            idxs = np.asarray(idxs, dtype=np.int32)
            vlabels[idxs] = i

        valid = vlabels >= 0
        max_lab = int(vlabels[valid].max()) if valid.any() else 0

        norm = np.zeros(nv, dtype=np.uint8)
        if max_lab > 0:
            norm[valid] = np.round(vlabels[valid] / max_lab * 255).astype(np.uint8)

        colors_bgr = cv2.applyColorMap(norm.reshape(-1, 1), colormap).reshape(-1, 3)
        colors_rgba = np.concatenate([colors_bgr[:, ::-1], 255*np.ones((nv, 1), dtype=np.uint8)], axis=1)
        colors_rgba[~valid] = np.array([200, 200, 200, 255], dtype=np.uint8)  # unlabeled -> light gray
        mesh.visual.vertex_colors = colors_rgba

        # 2) Scene with bright ambient and stronger lights
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(1.0, 1.0, 1.0))

        cam_pose = np.eye(4)
        cam_pose[:3, 3] = np.array([0., 0., 2.5])
        pyrencamera = pyrender.camera.IntrinsicsCamera(fx=5000, fy=5000, cx=250, cy=250)
        scene.add(pyrencamera, pose=cam_pose)

        # Brighten with a directional-ish setup
        center = mesh.vertices.mean(0)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light_pose = np.eye(4)
        for lp in ([1, 1, 2.5], [-1, 1, 2.5], [1, -1, 2.5], [-1, -1, 2.5]):
            light_pose[:3, 3] = center + np.array(lp, dtype=float)
            scene.add(light, pose=light_pose)

        # Use a local offscreen renderer so we can set FLAT shading
        renderer = pyrender.OffscreenRenderer(viewport_width=img_res, viewport_height=img_res, point_size=1.0)
        render_flags = (pyrender.RenderFlags.RGBA |
                        pyrender.RenderFlags.FLAT |
                        pyrender.RenderFlags.ALL_SOLID)

        def add_and_render(rot_axis, angle_deg, extra_shift=None):
            m = mesh.copy()
            rot = trimesh.transformations.rotation_matrix(np.radians(angle_deg), rot_axis, point=center)
            m.apply_transform(rot)
            if extra_shift is not None:
                m.apply_translation(np.asarray(extra_shift, dtype=float))
            # preserve vertex colors: material=None
            m_node = pyrender.Mesh.from_trimesh(m, material=None, smooth=False)
            scene.add(m_node, name="mesh")
            color_rgba, _ = renderer.render(scene, flags=render_flags)
            scene.remove_node(scene.get_nodes(name="mesh").pop())
            # RGBA uint8 -> RGB uint8
            return cv2.cvtColor(color_rgba, cv2.COLOR_RGBA2RGB)

        panels = []
        view_names = []

        # around Y: front, right, back, left
        for ang, name in zip([0, 90, 180, 270], ["Front", "Right", "Back", "Left"]):
            panels.append(add_and_render([0, 1, 0], ang))
            view_names.append(name)

        # top and bottom around X (with your translations if desired)
        top_shift = (0.0, 0.02, 0.0)
        panels.append(add_and_render([1, 0, 0], 90, extra_shift=top_shift))
        view_names.append("Top")

        bottom_shift = (0.0, 0.03, 0.0)
        panels.append(add_and_render([1, 0, 0], 270, extra_shift=tuple((np.array(top_shift) + np.array(bottom_shift)).tolist())))
        view_names.append("Bottom")

        canvas = np.hstack(panels)

        if False:
            w = panels[0].shape[1]
            for i, name in enumerate(view_names):
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs, th = 0.7, 2
                size = cv2.getTextSize(name, font, fs, th)[0]
                x = i * w + (w - size[0]) // 2
                y = panels[i].shape[0] - 10
                cv2.putText(canvas, name, (x, y), font, fs, (0, 0, 0), th, cv2.LINE_AA)

        return canvas





# This function is for demo code with mediapipe
MARGIN = 10  # pixels

# COCO style skeleton pairs for 17 kpts
SKELETON = [
    (5, 6),      # shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (11, 12),            # hips
    (5, 11), (6, 12),    # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (0, 5), (0, 6),      # head to shoulders
]


def draw_landmarks_on_image(rgb_image, keypoints):
    """
    Args:
        rgb_image: H x W x 3 RGB numpy array
        keypoints: (K, 2) array-like, pixel coords [x, y]

    Returns:
        annotated_image, human_bbox [x, y, w, h] or None
    """
    if keypoints is None:
        return rgb_image, None

    if isinstance(keypoints, (list, tuple)):
        keypoints = np.array(keypoints)
    if hasattr(keypoints, "detach"):  # torch tensor
        keypoints = keypoints.detach().cpu().numpy()

    if keypoints.size == 0:
        return rgb_image, None

    annotated = rgb_image.copy()
    h, w, _ = annotated.shape

    xs, ys = keypoints[:, 0], keypoints[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    bw, bh = (x_max - x_min), (y_max - y_min)

    expand_ratio = getattr(cfg.DATASET, "body_big_bbox_expand_ratio",
                           getattr(cfg.DATASET, "ho_big_bbox_expand_ratio", 1.5))
    bw_exp, bh_exp = bw * expand_ratio, bh * expand_ratio
    x_min_exp, y_min_exp = cx - 0.5 * bw_exp, cy - 0.5 * bh_exp
    human_bbox = [x_min_exp, y_min_exp, bw_exp, bh_exp]

    # === draw keypoints (larger circles + index numbers) ===
    for idx, (x, y) in enumerate(keypoints):
        if 0 <= x < w and 0 <= y < h:
            # Larger circle
            cv2.circle(annotated, (int(x), int(y)), 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            # Draw index number
            cv2.putText(
                annotated,
                str(idx),
                (int(x) + 8, int(y) - 8),  # slight offset
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,                       # larger text
                (0, 255, 0),
                2,                         # thicker text
                cv2.LINE_AA
            )

    # === draw skeleton (thicker lines) ===
    K = len(keypoints)
    for i1, i2 in SKELETON:
        if i1 >= K or i2 >= K:
            continue
        x1, y1 = keypoints[i1]
        x2, y2 = keypoints[i2]
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue
        cv2.line(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            4,               # increased from 2 → 4
            lineType=cv2.LINE_AA
        )

    # === bbox around keypoints ===
    cv2.rectangle(
        annotated,
        (int(x_min), int(y_min)),
        (int(x_max), int(y_max)),
        (0, 255, 0),
        3  # thicker bbox
    )
    cv2.putText(
        annotated,
        "Person",
        (int(x_min), int(y_min) - MARGIN),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,      # larger font
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return annotated, human_bbox