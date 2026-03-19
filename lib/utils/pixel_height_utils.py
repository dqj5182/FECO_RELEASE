import cv2
import trimesh
import pyrender
import numpy as np


def render_foot_depth(img, mesh, face, cam_param, cam_pose=None):
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

    # Scene
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh_node, 'mesh')

    # Camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    if cam_pose is not None:
        scene.add(camera, pose=cam_pose)
    else:
        scene.add(camera)

    # Renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # Lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    for pos in [[0, -1, 1], [0, 1, 1], [1, 1, 2]]:
        light_pose[:3, 3] = pos
        scene.add(light, pose=light_pose)

    # Render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    img = rgb * valid_mask + img * (1 - valid_mask)
    return img, depth, valid_mask


def compute_image_pixel_height_along_ground_direction(
    depth, cam_param, ground_plane_slope_x, ground_plane_slope_z, ground_plane_intercept, positive_y_up=False
):
    H, W = depth.shape
    fx, fy = cam_param['focal']
    cx, cy = cam_param['princpt']

    valid_mask = (depth > 0).astype(np.uint8)
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    x_norm = (xv - cx) / fx
    y_norm = (yv - cy) / fy

    Z = depth
    Z_safe = np.where(valid_mask, Z, 1.0)

    X = x_norm * Z_safe
    Y = y_norm * Z_safe

    ground_Y = ground_plane_slope_x * X + ground_plane_slope_z * Z_safe + ground_plane_intercept

    image_x = (X * fx) / Z_safe + cx
    image_y = (Y * fy) / Z_safe + cy
    ground_image_y = (ground_Y * fy) / Z_safe + cy

    disp_vec_x = image_x - xv
    disp_vec_y = image_y - yv
    disp_vec_y_ground = ground_image_y - yv

    delta_x = disp_vec_x
    delta_y = disp_vec_y - disp_vec_y_ground

    plane_normal = np.array([ground_plane_slope_x, -1.0, ground_plane_slope_z])
    plane_normal /= np.linalg.norm(plane_normal)

    ground_dir_2d_x = (fx * plane_normal[0]) / Z_safe
    ground_dir_2d_y = (fy * plane_normal[1]) / Z_safe

    ground_dir_norm = np.sqrt(ground_dir_2d_x**2 + ground_dir_2d_y**2) + 1e-6
    ground_dir_2d_x /= ground_dir_norm
    ground_dir_2d_y /= ground_dir_norm

    pixel_height_map = delta_x * ground_dir_2d_x + delta_y * ground_dir_2d_y
    pixel_height_map *= valid_mask

    pixel_height_map[pixel_height_map < 0] = 0

    return pixel_height_map, valid_mask


def compute_image_pixel_height_along_ground_direction_xy(
    depth, cam_param, ground_plane_slope_x, ground_plane_slope_y, ground_plane_intercept
):
    H, W = depth.shape
    fx, fy = cam_param['focal']
    cx, cy = cam_param['princpt']

    valid_mask = (depth > 0).astype(np.uint8)
    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    x_norm = (xv - cx) / fx
    y_norm = (yv - cy) / fy

    Z = depth
    Z_safe = np.where(valid_mask, Z, 1.0)

    X = x_norm * Z_safe
    Y = y_norm * Z_safe

    ground_Z = ground_plane_slope_x * X + ground_plane_slope_y * Y + ground_plane_intercept

    image_x = (X * fx) / Z_safe + cx
    image_y = (Y * fy) / Z_safe + cy
    ground_image_z = ground_Z

    delta_X = X
    delta_Y = Y
    delta_Z = Z - ground_Z

    plane_normal = np.array([ground_plane_slope_x, ground_plane_slope_y, -1.0])
    plane_normal /= np.linalg.norm(plane_normal)

    ground_dir_2d_x = (fx * plane_normal[0]) / Z_safe
    ground_dir_2d_y = (fy * plane_normal[1]) / Z_safe

    ground_dir_norm = np.sqrt(ground_dir_2d_x**2 + ground_dir_2d_y**2) + 1e-6
    ground_dir_2d_x /= ground_dir_norm
    ground_dir_2d_y /= ground_dir_norm

    image_x_disp = (X * fx) / Z_safe + cx - xv
    image_y_disp = (Y * fy) / Z_safe + cy - yv
    delta_z = delta_Z

    pixel_height_map = image_x_disp * ground_dir_2d_x + image_y_disp * ground_dir_2d_y
    pixel_height_map *= valid_mask

    pixel_height_map[pixel_height_map < 0] = 0

    return pixel_height_map, valid_mask


def save_height_map_cv2(height_map, valid_mask, save_path, img_size, colormap=cv2.COLORMAP_VIRIDIS, fixed_min=None, fixed_max=None):
    H, W = img_size
    assert height_map.shape == valid_mask.shape == (H, W)

    pixel_height_map = np.tile(np.arange(H-1, -1, -1).reshape(H, 1), (1, W)).astype(float)

    combined_height = np.where(valid_mask.astype(bool), height_map, pixel_height_map)

    valid_heights = height_map[valid_mask.astype(bool)]
    background_heights = pixel_height_map[valid_mask == 0]

    all_heights = np.concatenate([valid_heights, background_heights]) if valid_heights.size + background_heights.size > 0 else np.array([0.0])

    min_h = fixed_min if fixed_min is not None else all_heights.min()
    max_h = fixed_max if fixed_max is not None else all_heights.max()

    if max_h - min_h > 1e-6:
        norm_height = (combined_height - min_h) / (max_h - min_h)
        norm_height = np.clip(norm_height, 0, 1)
    else:
        norm_height = np.zeros_like(combined_height)

    norm_height = (norm_height * 255).astype(np.uint8)
    color_height = cv2.applyColorMap(norm_height, colormap)

    cv2.imwrite(save_path, color_height)

    return color_height


def save_foot_height_map_cv2(height_map, valid_mask, save_path, img_size, colormap=cv2.COLORMAP_VIRIDIS, fixed_min=None, fixed_max=None):
    H, W = img_size
    assert height_map.shape == valid_mask.shape == (H, W)

    valid_heights = height_map[valid_mask.astype(bool)]
    if valid_heights.size > 0:
        min_h = fixed_min if fixed_min is not None else valid_heights.min()
        max_h = fixed_max if fixed_max is not None else valid_heights.max()
    else:
        min_h, max_h = 0.0, 1.0

    range_h = max_h - min_h

    clipped_height = np.copy(height_map)
    clipped_height[valid_mask == 0] = min_h
    if range_h > 1e-6:
        norm_height = (clipped_height - min_h) / range_h
        norm_height = np.clip(norm_height, 0, 1)
    else:
        norm_height = np.zeros_like(height_map)

    norm_height_uint8 = (norm_height * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(norm_height_uint8, colormap)

    cv2.imwrite(save_path, color_map)
    return color_map