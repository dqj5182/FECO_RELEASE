import os
import cv2
import json
import torch
import trimesh
import pyrender
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

os.environ["PYOPENGL_PLATFORM"] = "egl"

from lib.core.config import cfg
from lib.utils.human_models import smpl_x
from lib.utils.mesh_utils import extract_partial_mesh


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
                global_orient=torch.zeros(1, 3),
                body_pose=torch.zeros(1, 21 * 3),
                left_hand_pose=torch.zeros(1, 15 * 3),
                right_hand_pose=torch.zeros(1, 15 * 3),
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

            self.mesh_foot_r.vertices = self.mesh_foot_r.vertices - self.mesh_foot_r.vertices.mean(axis=0)
            self.mesh_foot_r.vertices = self.mesh_foot_r.vertices * 0.7

            self.mesh_foot_r.vertices[:, 1] = self.mesh_foot_r.vertices[:, 1] - 0.03
            self.mesh_foot_r.vertices[:, 2] = self.mesh_foot_r.vertices[:, 2] + 0.01
            
            self.mesh_foot_r.faces = np.concatenate((self.mesh_foot_r.faces, watertight_foot_faces), axis=0)

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
        # Add mesh for camera
        camera_pose = np.eye(4)
        camera_rotation = np.eye(3, 3)
        camera_translation = np.array([0., 0, 2.5])
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = camera_rotation @ camera_translation

        pyrencamera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center, cy=camera_center)
        scene.add(pyrencamera, pose=camera_pose)

        # Create and add light
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
            light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
            scene.add(light, pose=light_pose)

        # Add body mesh
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        mesh_images = []

        # Resize input image to fit the mesh image height
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

            # Delete the previous mesh
            prev_mesh = scene.get_nodes(name='mesh').pop()
            scene.remove_node(prev_mesh)

        # Show upside down view
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

            # Delete the previous mesh
            prev_mesh = scene.get_nodes(name='mesh').pop()
            scene.remove_node(prev_mesh)

        # Stack images
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
            # Preserve vertex colors: material=None
            m_node = pyrender.Mesh.from_trimesh(m, material=None, smooth=False)
            scene.add(m_node, name="mesh")
            color_rgba, _ = renderer.render(scene, flags=render_flags)
            scene.remove_node(scene.get_nodes(name="mesh").pop())
            # RGBA uint8 -> RGB uint8
            return cv2.cvtColor(color_rgba, cv2.COLOR_RGBA2RGB)

        panels = []
        view_names = []

        # Around Y: front, right, back, left
        for ang, name in zip([0, 90, 180, 270], ["Front", "Right", "Back", "Left"]):
            panels.append(add_and_render([0, 1, 0], ang))
            view_names.append(name)

        # Top and bottom around X (with your translations if desired)
        top_shift = (0.0, 0.02, 0.0)
        panels.append(add_and_render([1, 0, 0], 90, extra_shift=top_shift))
        view_names.append("Top")

        bottom_shift = (0.0, 0.03, 0.0)
        panels.append(add_and_render([1, 0, 0], 270, extra_shift=tuple((np.array(top_shift) + np.array(bottom_shift)).tolist())))
        view_names.append("Bottom")

        canvas = np.hstack(panels)

        return canvas


# This function is for demo code with mediapipe
MARGIN = 10  # pixels


# COCO style skeleton pairs for 17 kpts
SKELETON = [
    (5, 6),              # shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (11, 12),            # hips
    (5, 11), (6, 12),    # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (0, 5), (0, 6),      # head to shoulders
]


def draw_landmarks_on_image(rgb_image, keypoints):
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


def render_mesh(img, mesh, face, cam_param, cam_pose=None):
    # Mesh
    orig_mesh = mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
	np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    if cam_pose is not None:
        scene.add(camera, pose=cam_pose)
    else:
        scene.add(camera)
 
    # Renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # Light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # Render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # Save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    return img, valid_mask


def vis_keypoints(img, kps, alpha=1, size=2):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    if len(kps) == 1: colors = [(255,255,255)]

    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps),1))], axis=1)

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    kp_mask = np.ascontiguousarray(kp_mask, dtype=np.uint8)

    # Draw the keypoints.
    for i in range(len(kps)):
        if kps[i][-1] > 0:
            p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
            cv2.circle(kp_mask, p, radius=size, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
            
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints_openpose(
    img, kps, contact=None,
    alpha=1.0, size=5, inner_ratio=0.55,
    cmap_name='rainbow',
    upscale=2,
    return_original_size=True,
    edges=((0,1),(0,2),(0,3)),
):
    palette_bgr = [
        (0, 165, 255),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    kps = np.asarray(kps)
    N = kps.shape[0]

    # Visibility
    vis = np.ones((N,), dtype=np.uint8) if kps.shape[1] == 2 else (kps[:,2] > 0).astype(np.uint8)

    # Contact
    if contact is None:
        contact_mask = np.zeros((N,), dtype=np.uint8)
    else:
        c = np.asarray(contact).reshape(-1)
        contact_mask = (c if c.dtype == bool else (c >= 0.5)).astype(np.uint8)

    # Colors
    cmap = plt.get_cmap(cmap_name)
    colors_rgba = [cmap(i) for i in np.linspace(0, 1, N + 2)]
    # Build per-joint colors (cycle if N > palette length)
    colors_bgr = [palette_bgr[i % len(palette_bgr)] for i in range(N)]
    if N == 1:
        colors_bgr = [(255, 255, 255)]

    img = np.ascontiguousarray(img, dtype=np.uint8)
    H, W = img.shape[:2]

    # Upsample image & scale params
    if upscale > 1:
        upW, upH = W * upscale, H * upscale
        big = cv2.resize(img, (upW, upH), interpolation=cv2.INTER_CUBIC)
        overlay = big.copy()
        kps_up = kps.astype(np.float32).copy()
        kps_up[:, :2] *= float(upscale)
        size_up = int(round(size * upscale))
        inner_r = max(2, int(round(size_up * inner_ratio)))
    else:
        big = img
        overlay = img.copy()
        kps_up = kps.astype(np.float32)
        size_up = size
        inner_r = max(2, int(round(size * inner_ratio)))

    upH, upW = big.shape[:2]

    edge_thickness = max(1, size_up // 2)
    for (a, b) in edges:
        if a >= N or b >= N: 
            continue
        if vis[a] <= 0 or vis[b] <= 0:
            continue
        xa, ya = kps_up[a, 0], kps_up[a, 1]
        xb, yb = kps_up[b, 0], kps_up[b, 1]
        if not (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
            continue
        pa = (int(round(xa)), int(round(ya)))
        pb = (int(round(xb)), int(round(yb)))
        if not (0 <= pa[0] < upW and 0 <= pa[1] < upH and 0 <= pb[0] < upW and 0 <= pb[1] < upH):
            continue

        # Pick the color of the non-0 endpoint if 0 is one of the vertices
        if a == 0:
            edge_color = colors_bgr[b]
        elif b == 0:
            edge_color = colors_bgr[a]
        else:
            edge_color = colors_bgr[a]

        cv2.line(overlay, pa, pb, edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)

    for i in range(N):
        if vis[i] <= 0:
            continue
        x, y = kps_up[i, 0], kps_up[i, 1]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        px, py = int(round(x)), int(round(y))
        if px < 0 or py < 0 or px >= upW or py >= upH:
            continue

        # Outer colored circle
        cv2.circle(overlay, (px, py), radius=size_up, color=colors_bgr[i],
                   thickness=-1, lineType=cv2.LINE_AA)
        # Inner contact circle
        inner_color = (0, 0, 0) if contact_mask[i] == 1 else (255, 255, 255)
        cv2.circle(overlay, (px, py), radius=inner_r, color=inner_color,
                   thickness=-1, lineType=cv2.LINE_AA)
        # Thin outline
        cv2.circle(overlay, (px, py), radius=size_up, color=(0, 0, 0),
                   thickness=1, lineType=cv2.LINE_AA)

    blended_big = cv2.addWeighted(big, 1.0 - alpha, overlay, alpha, 0.0)

    # Downsample back
    if upscale > 1 and return_original_size:
        return cv2.resize(blended_big, (W, H), interpolation=cv2.INTER_AREA)
    else:
        return blended_big


def vis_keypoints_with_text(img, kps, alpha=1, size=2, text_size=0.3):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    if len(kps) == 1: 
        colors = [(255, 255, 255)]

    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps), 1))], axis=1)

    kp_mask = np.copy(img)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    kp_mask = np.ascontiguousarray(kp_mask, dtype=np.uint8)

    for i in range(len(kps)):
        if kps[i][-1] > 0:
            p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
            cv2.circle(kp_mask, p, radius=size, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(kp_mask, str(i), (p[0] + 5, p[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_ground_normal(img, ground_normal, length_frac=0.25, color=(0, 0, 255), thickness=4, tipLength=0.2):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    n = np.asarray(ground_normal, dtype=np.float32)
    if n.shape != (3,):
        raise ValueError("ground_normal must be shape (3,)")

    v2d = np.array([n[0], -n[1]], dtype=np.float32)

    norm_v = np.linalg.norm(v2d)
    if norm_v < 1e-6:
        v2d = np.array([0.0, -1.0], dtype=np.float32)
    else:
        v2d /= norm_v

    length_px = int(min(h, w) * float(length_frac))
    dx, dy = (v2d * length_px).astype(int)

    p1 = (int(cx), int(cy))
    p2 = (int(cx + dx), int(cy + dy))

    vis = img.copy()
    cv2.arrowedLine(vis, p1, p2, color=color, thickness=thickness, tipLength=tipLength)
    return vis


def vis_bbox(img, bbox, thickness=1):
    img = img.copy()
    color = (0, 255, 0)

    if len(bbox) == 4:
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        pos1 = (x_min, y_min)
        pos2 = (x_min, y_max)
        pos3 = (x_max, y_min)
        pos4 = (x_max, y_max)

        img = cv2.line(img, pos1, pos2, color, thickness) 
        img = cv2.line(img, pos1, pos3, color, thickness) 
        img = cv2.line(img, pos2, pos4, color, thickness) 
        img = cv2.line(img, pos3, pos4, color, thickness) 

    return img


def color_mesh_parts(mesh, part_dict, colormap=cv2.COLORMAP_SUMMER):
    num_verts = mesh.vertices.shape[0]
    vert_labels = -1 * np.ones(num_verts, int)

    for i, idx in enumerate(part_dict.values()):
        vert_labels[idx] = i

    face_labels = vert_labels[mesh.faces].max(axis=1)
    norm_labels = np.clip((face_labels / face_labels.max() * 255), 0, 255).astype(np.uint8)

    colors = cv2.applyColorMap(norm_labels, colormap).reshape(-1, 3)
    colors = np.concatenate((colors[:, ::-1], np.full((len(colors), 1), 255, np.uint8)), axis=-1)

    mesh.visual.face_colors = colors

    return mesh, colors


def render_part_seg(img, mesh, face_color, cam_param, cam_pose=None):
    rot = trimesh.transformations.rotation_matrix(
	            np.radians(180), [1, 0, 0]
            )
    mesh.apply_transform(rot)
    mesh.visual.face_colors = face_color

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=(1.0, 1.0, 1.0))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    if cam_pose is not None:
        scene.add(camera, pose=cam_pose)
    else:
        scene.add(camera)
 
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    rgb, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

    return rgb


def render_part_seg_img(img, mesh, face_color, cam_param, cam_pose=None):
    rot = trimesh.transformations.rotation_matrix(
	            np.radians(180), [1, 0, 0]
            )
    mesh.apply_transform(rot)
    mesh.visual.face_colors = face_color

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=(1.0, 1.0, 1.0))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    if cam_pose is not None:
        scene.add(camera, pose=cam_pose)
    else:
        scene.add(camera)
 
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    rgb, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    valid_mask = (depth > 0)[:, :, None]
    img = rgb[:, :, :3] * valid_mask + img * (1 - valid_mask)

    return img


class ContactHeatmapRenderer():
    def __init__(self):
        self.default_mesh_color = [130, 130, 130, 255]
        self.contact_mesh_color = [0, 255, 0, 255]

        with torch.no_grad():
            smpl_x_rest_out = smpl_x.layer(
                betas=torch.zeros(1, 10),
                global_orient=torch.zeros(1, 3),
                body_pose=torch.zeros(1, 21 * 3),
                left_hand_pose=torch.zeros(1, 15 * 3),
                right_hand_pose=torch.zeros(1, 15 * 3),
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

            self.mesh_foot_r.vertices = self.mesh_foot_r.vertices - self.mesh_foot_r.vertices.mean(axis=0)
            self.mesh_foot_r.vertices = self.mesh_foot_r.vertices * 0.7

            self.mesh_foot_r.vertices[:, 1] = self.mesh_foot_r.vertices[:, 1] - 0.03
            self.mesh_foot_r.vertices[:, 2] = self.mesh_foot_r.vertices[:, 2] + 0.01

            self.mesh_foot_r.faces = np.concatenate((self.mesh_foot_r.faces, watertight_foot_faces), axis=0)

    def normalize_contact(self, contact):
        contact_min = contact.min()
        contact_max = contact.max()
        if contact_max - contact_min > 0:
            return (contact - contact_min) / (contact_max - contact_min)
        return contact

    def contact_to_color(self, contact_value):
        cmap = cm.get_cmap('jet')
        return cmap(1.0 - contact_value)[:3]

    def render_image(self, scene, img_res, img=None, viewer=False):
        r = pyrender.OffscreenRenderer(viewport_width=img_res, viewport_height=img_res, point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        color[color[:, :, 3] > 0, 3] = 1.0  

        if img is not None:
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
        else:
            output_img = color
        return output_img

    def create_scene(self, mesh, focal_length=5000, camera_center=250, img_res=500):
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        camera_pose = np.eye(4)
        camera_rotation = np.eye(3, 3)
        camera_translation = np.array([0., 0, 2.5])
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = camera_rotation @ camera_translation

        pyrencamera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center, cy=camera_center)
        scene.add(pyrencamera, pose=camera_pose)

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
            light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
            scene.add(light, pose=light_pose)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='MASK',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0)
        )

        mesh_images = []

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
            mesh_images.append(output_img)

            prev_mesh = scene.get_nodes(name='mesh').pop()
            scene.remove_node(prev_mesh)

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
            mesh_images.append(output_img)

            prev_mesh = scene.get_nodes(name='mesh').pop()
            scene.remove_node(prev_mesh)

        IMG = np.hstack(mesh_images)
        return IMG, mesh_images


    def render_contact(self, contact):
        contact = self.normalize_contact(contact)

        for vert in range(self.mesh_foot_r.visual.vertex_colors.shape[0]):
            color = self.contact_to_color(contact[vert])
            self.mesh_foot_r.visual.vertex_colors[vert] = [
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
                255
            ]

        rend = self.create_scene(self.mesh_foot_r)
        return rend