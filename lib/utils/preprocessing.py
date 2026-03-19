import cv2
import torch
import random
import numpy as np
import torchvision.transforms as T

from lib.core.config import cfg


def get_aug_config_contact():
    # Augmentation factors
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    trans_factor = 0.1
    noise_std = 0.02
    motion_blur_prob = 0.15
    extreme_crop_prob = 0.1
    extreme_crop_lvl = 0.3
    low_res_prob = 0.05
    low_res_scale_range = (0.15, 0.5)

    # Scaling augmentation
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0

    # Rotation augmentation
    rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0

    # Color augmentation
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([
        random.uniform(c_low, c_up),
        random.uniform(c_low, c_up),
        random.uniform(c_low, c_up)
    ])

    # Flipping augmentation
    do_flip = random.random() <= 0.5

    # Translation augmentation
    tx = np.clip(np.random.randn(), -1.0, 1.0) * trans_factor
    ty = np.clip(np.random.randn(), -1.0, 1.0) * trans_factor

    # Extreme cropping augmentation
    do_extreme_crop = random.random() <= extreme_crop_prob

    # Noise augmentation (returns standard deviation for Gaussian noise injection)
    add_noise = random.random() <= 0.3  # 30% chance of adding noise
    noise_std = noise_std if add_noise else 0.0

    # Motion blur augmentation
    apply_motion_blur = random.random() <= motion_blur_prob
    motion_blur_kernel_size = random.choice([3, 5, 7]) if apply_motion_blur else 0

    # Low-resolution augmentation
    apply_low_res = random.random() <= low_res_prob
    low_res_scale = random.uniform(*low_res_scale_range) if apply_low_res else 1.0

    return {
        'scale': scale,
        'rot': rot,
        'color_scale': color_scale,
        'do_flip': do_flip,
        'tx': tx,
        'ty': ty,
        'do_extreme_crop': do_extreme_crop,
        'extreme_crop_lvl': extreme_crop_lvl if do_extreme_crop else 0,
        'noise_std': noise_std,
        'motion_blur_kernel_size': motion_blur_kernel_size,
        'low_res_scale': low_res_scale # Added low-res scale parameter
    }


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def generate_patch_image_contact(cvimg, bbox, scale, rot, do_flip, out_shape, tx=0.0, ty=0.0, bkg_color='black'):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if bkg_color == 'white':
        borderMode=cv2.BORDER_CONSTANT
        borderValue=(255, 255, 255)
    else:
        borderMode=cv2.BORDER_CONSTANT
        borderValue=(0, 0, 0)

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    # Add translation offset
    bb_c_x += tx * img_width
    bb_c_y += ty * img_height

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, 
                                    out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=borderValue)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, 
                                        out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans


def augmentation_shoe(img, bbox, data_split, enforce_flip=None, bkg_color='black'):
    """
    Augmentation for shoe images with RandAugment, ColorJitter, RandomResizedCrop.
    """
    if data_split == 'train':
        scale = np.clip(np.random.randn(), -1.0, 1.0) * 0.25 + 1.0
        rot = np.clip(np.random.randn(), -2.0, 2.0) * 30 if random.random() <= 0.6 else 0
        do_flip = random.random() <= 0.5
        tx = np.clip(np.random.randn(), -1.0, 1.0) * 0.1
        ty = np.clip(np.random.randn(), -1.0, 1.0) * 0.1
    else:
        scale, rot, do_flip, tx, ty = 1.0, 0.0, False, 0.0, 0.0

    if enforce_flip is not None:
        do_flip = enforce_flip

    img_patch, trans, inv_trans = generate_patch_image_shoe(
        img, bbox, scale, rot, do_flip, cfg.MODEL.input_img_shape, tx, ty, bkg_color
    )

    if data_split == 'train':
        img_patch = apply_randaugment_pipeline(img_patch)

    return img_patch, trans, inv_trans, rot, do_flip


def generate_patch_image_shoe(img, bbox, scale, rot, do_flip, out_shape, tx=0.0, ty=0.0, bkg_color='black'):
    """
    Generate cropped and transformed image patch.
    """
    img = img.copy()
    h, w = img.shape[:2]

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    borderValue = (255, 255, 255) if bkg_color == 'white' else (0, 0, 0)

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = w - bb_c_x - 1

    bb_c_x += tx * w
    bb_c_y += ty * h

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (out_shape[1], out_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch.astype(np.float32), trans, inv_trans


randaug_pipeline = T.Compose([
    T.ToPILImage(),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor()
])


def apply_randaugment_pipeline(img_patch):
    """
    Apply RandAugment, ColorJitter, and RandomResizedCrop.
    Expects input as NumPy image in uint8 or float32 [0-255] range.
    Returns float32 image scaled [0,255].
    """
    img_patch = np.clip(img_patch, 0, 255).astype(np.uint8)
    img_patch = randaug_pipeline(img_patch) * 255.0
    return img_patch.permute(1, 2, 0).numpy().astype(np.float32)


def mixup_data(x, y, num_classes_per_group, alpha=0.2):
    """
    MixUp for images and multi-group class labels.
    y['class_label']: (B, G) tensor of class indices (-1 for invalid)
    y['class_valid']: (B, G) tensor (0/1) validity mask
    num_classes_per_group: list of class counts per group
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_labels = []

    for g, num_cls in enumerate(num_classes_per_group):
        labels_a = y['class_label'][:, g]
        labels_b = y['class_label'][index, g]
        valid_a = y['class_valid'][:, g]
        valid_b = y['class_valid'][index, g]

        one_hot_a = torch.zeros(batch_size, num_cls, device=x.device)
        one_hot_b = torch.zeros(batch_size, num_cls, device=x.device)

        mask_a = labels_a >= 0
        mask_b = labels_b >= 0

        one_hot_a[mask_a, labels_a[mask_a]] = 1
        one_hot_b[mask_b, labels_b[mask_b]] = 1

        mixed_label = lam * one_hot_a + (1 - lam) * one_hot_b
        mixed_labels.append(mixed_label.to(x.device))

    return mixed_x, mixed_labels, lam


def apply_augmentation_height(height_map, img2bb_trans, do_flip, out_shape, apply_extreme_crop=False, extreme_crop_lvl=0.0):
    """
    Applies the same geometric augmentation to the pixel height map.

    Args:
        height_map (H, W): Original dense height map
        img2bb_trans: Transformation matrix from generate_patch_image_contact
        do_flip: Whether horizontal flip was applied
        out_shape: Output image shape (height, width)
        apply_extreme_crop: Whether to apply extreme cropping
        extreme_crop_lvl: Crop level for extreme cropping

    Returns:
        height_map_aug: Augmented height map (out_shape)
    """
    height_map_aug = height_map.copy()

    # 1) Flip if necessary
    if do_flip:
        height_map_aug = height_map_aug[:, ::-1]

    # 2) Warp (crop + scale + rotate) to the same out_shape
    height, width = out_shape
    height_map_aug = cv2.warpAffine(height_map_aug, img2bb_trans, (width, height), flags=cv2.INTER_NEAREST)

    # 3) Apply extreme cropping (if requested)
    if apply_extreme_crop and extreme_crop_lvl > 0:
        h, w = height_map_aug.shape
        crop_size = max(1, int(min(h, w) * (1 - extreme_crop_lvl)))
        start_x = np.random.randint(0, max(0, w - crop_size))
        start_y = np.random.randint(0, max(0, h - crop_size))
        cropped = height_map_aug[start_y:start_y + crop_size, start_x:start_x + crop_size]
        height_map_aug = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    return height_map_aug


def apply_augmentation_mask(mask, img2bb_trans, do_flip, out_shape, apply_extreme_crop=False, extreme_crop_lvl=0.0):
    """
    Applies the same geometric augmentation to valid mask or binary maps.
    """
    mask_aug = mask.copy()

    # Flip if needed
    if do_flip:
        mask_aug = mask_aug[:, ::-1]

    # Affine transform
    height, width = out_shape
    mask_aug = cv2.warpAffine(mask_aug, img2bb_trans, (width, height), flags=cv2.INTER_NEAREST)

    # # Extreme cropping if applicable
    # if apply_extreme_crop and extreme_crop_lvl > 0:
    #     h, w = mask_aug.shape
    #     crop_size = max(1, int(min(h, w) * (1 - extreme_crop_lvl)))
    #     start_x = np.random.randint(0, max(0, w - crop_size))
    #     start_y = np.random.randint(0, max(0, h - crop_size))
    #     cropped = mask_aug[start_y:start_y + crop_size, start_x:start_x + crop_size]
    #     mask_aug = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask_aug


def compose_affine(A, B):
    """Compose two 2x3 affines: returns A ∘ B (apply B first, then A)."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    C3 = A3 @ B3
    return C3[:2, :]


def flip_affine_full(do_flip, full_w):
    if not do_flip:
        return np.array([[1., 0., 0.],
                         [0., 1., 0.]], dtype=np.float32)
    return np.array([[-1.,  0., full_w - 1.],
                     [ 0.,  1., 0.        ]], dtype=np.float32)


def apply_augmentation_mask_direct_from_crop(
    mask_cropped,
    inv_trans_cropped_to_full,
    img2bb_trans_full_to_out,
    do_flip,
    full_image_shape,
    out_shape,
    apply_extreme_crop=False,
    extreme_crop_lvl=0.0
):
    H_out, W_out = out_shape
    H_full, W_full = full_image_shape

    # Build flip in full-image coordinates
    F_full = flip_affine_full(do_flip, W_full)

    # Compose: cropped → out
    # Order matters: inv_trans first, then flip, then img2bb
    T_direct = compose_affine(img2bb_trans_full_to_out,
                 compose_affine(F_full, inv_trans_cropped_to_full)).astype(np.float32)

    # Single warp from cropped to output
    flags = cv2.INTER_NEAREST
    if mask_cropped.ndim == 2:
        out = cv2.warpAffine(mask_cropped, T_direct, (W_out, H_out),
                             flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        out = cv2.warpAffine(mask_cropped, T_direct, (W_out, H_out),
                             flags=flags, borderMode=cv2.BORDER_CONSTANT)

    # # Preserve your existing extreme crop step exactly
    # if apply_extreme_crop and extreme_crop_lvl > 0:
    #     h, w = out.shape[:2]
    #     crop_size = max(1, int(min(h, w) * (1 - extreme_crop_lvl)))
    #     start_x = np.random.randint(0, max(0, w - crop_size))
    #     start_y = np.random.randint(0, max(0, h - crop_size))
    #     cropped = out[start_y:start_y + crop_size, start_x:start_x + crop_size]
    #     out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    return out


def apply_augmentation_height_direct_from_crop(
    height_map_cropped,
    inv_trans_cropped_to_full,
    img2bb_trans_full_to_out,
    do_flip,
    full_image_shape,
    out_shape,
    apply_extreme_crop=False,
    extreme_crop_lvl=0.0
):
    """
    Warp pixel height map directly from cropped space to final output by composing
    inv_trans, optional flip in full coordinates, and img2bb_trans into one affine.
    """
    H_out, W_out = out_shape
    H_full, W_full = full_image_shape

    # Preserve original dtype, but use float32 for warping to keep exact values with INTER_NEAREST
    orig_dtype = height_map_cropped.dtype
    src = height_map_cropped.astype(np.float32, copy=False)

    # Build flip in full-image coordinates
    F_full = flip_affine_full(do_flip, W_full)

    # Compose: cropped → full → flip → out
    T_direct = compose_affine(
        img2bb_trans_full_to_out,
        compose_affine(F_full, inv_trans_cropped_to_full)
    ).astype(np.float32)

    # Single warp
    height_map_out = cv2.warpAffine(
        src,
        T_direct,
        (W_out, H_out),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # # Optional extreme crop on the output grid for equivalence with your current pipeline
    # if apply_extreme_crop and extreme_crop_lvl > 0:
    #     h, w = height_map_out.shape
    #     crop_size = max(1, int(min(h, w) * (1 - extreme_crop_lvl)))
    #     start_x = np.random.randint(0, max(0, w - crop_size))
    #     start_y = np.random.randint(0, max(0, h - crop_size))
    #     cropped = height_map_out[start_y:start_y + crop_size, start_x:start_x + crop_size]
    #     height_map_out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    return height_map_out.astype(orig_dtype, copy=False)


def apply_augmentation_ground_normal(normal, rot_deg, do_flip):
    """
    Apply image-space rotation and horizontal flip to a normal vector defined in camera space.
    Args:
        normal: np.ndarray of shape (3,), unit ground normal in camera space
        rot_deg: float, rotation in degrees (applied in image-space, around camera z-axis)
        do_flip: bool, whether horizontal flip was applied

    Returns:
        transformed_normal: np.ndarray of shape (3,), transformed normal vector
    """
    normal = normal.copy()
    
    # 1. Flip horizontally in image-space → flip x-axis in camera space
    if do_flip:
        normal[0] = -normal[0]  # flip x

    # 2. Rotate around z-axis in camera space (rotation in image space)
    rot_rad = np.deg2rad(rot_deg)
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    R = np.array([
        [cs, -sn, 0],
        [sn,  cs, 0],
        [ 0,   0, 1],
    ], dtype=np.float32)
    normal = R @ normal

    # 3. Re-normalize in case of numerical instability
    normal /= np.linalg.norm(normal)

    return normal


def augmentation_contact(img, bbox, data_split, enforce_flip=None, bkg_color='black'):
    if data_split == 'train':
        aug_params = get_aug_config_contact()
    else:
        aug_params = {
            'scale': 1.0,
            'rot': 0.0,
            'color_scale': np.array([1, 1, 1]),
            'do_flip': False,
            'tx': 0.0,
            'ty': 0.0,
            'do_extreme_crop': False,
            'extreme_crop_lvl': 0.0,
            'noise_std': 0.0,
            'motion_blur_kernel_size': 0,
            'low_res_scale': 1.0
        }
    
    # Enforce flip if specified
    if enforce_flip is not None:
        aug_params['do_flip'] = enforce_flip

    # Apply geometric augmentations (scaling, rotation, flipping)
    img, trans, inv_trans = generate_patch_image_contact(
        img, bbox, aug_params['scale'], aug_params['rot'], 
        aug_params['do_flip'], cfg.MODEL.input_img_shape, 
        aug_params['tx'], aug_params['ty'], bkg_color
    )

    # Apply low-resolution augmentation
    if aug_params['low_res_scale'] < 1.0:  # Only apply if scaling down
        img = apply_low_res(img, aug_params['low_res_scale'])

    # Apply color augmentation
    img = np.clip(img * aug_params['color_scale'][None, None, :], 0, 255)

    # # Apply extreme cropping
    # if aug_params['do_extreme_crop']:
    #     img = apply_extreme_crop(img, aug_params['extreme_crop_lvl'])

    # Apply noise augmentation
    if aug_params['noise_std'] > 0:
        img = add_gaussian_noise(img, aug_params['noise_std'])

    # Apply motion blur augmentation
    if aug_params['motion_blur_kernel_size'] > 0:
        img = apply_motion_blur(img, aug_params['motion_blur_kernel_size'])

    return img, trans, inv_trans, aug_params['rot'], aug_params['do_flip'], aug_params['color_scale'], aug_params['do_extreme_crop'], aug_params['extreme_crop_lvl']


def apply_extreme_crop(img, crop_lvl):
    """Extreme cropping: Aggressively crop the image."""
    h, w = img.shape[:2]
    crop_size = max(1, int(min(h, w) * (1 - crop_lvl)))
    start_x = random.randint(0, max(0, w - crop_size))
    start_y = random.randint(0, max(0, h - crop_size))
    cropped_img = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    # Preserve aspect ratio during resizing
    return cv2.resize(cropped_img, (w, h), interpolation=cv2.INTER_LINEAR)


def add_gaussian_noise(img, noise_std):
    """Add Gaussian noise to the image with proper scaling for data type."""
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    
    if img.dtype == np.uint8:
        noisy_img = np.clip(img + noise * 255, 0, 255).astype(np.uint8)
    elif img.dtype == np.float32:
        noisy_img = np.clip(img + noise, 0.0, 1.0).astype(np.float32)
    elif img.dtype == np.float64:
        noisy_img = np.clip(img + noise, 0.0, 1.0).astype(np.float64)
    else:
        raise TypeError("Unsupported image dtype. Expected uint8 or float32.")
        
    return noisy_img


def apply_motion_blur(img, kernel_size):
    """Apply motion blur to the image with a random direction."""
    kernel = np.zeros((kernel_size, kernel_size))
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])

    if direction == 'horizontal':
        kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
    elif direction == 'vertical':
        kernel[:, (kernel_size - 1) // 2] = np.ones(kernel_size)
    elif direction == 'diagonal':
        np.fill_diagonal(kernel, 1)
    
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def apply_low_res(img, scale_factor=0.25):
    """Simulate low-resolution effect by downsampling and upsampling."""
    if not (0 < scale_factor < 1):
        raise ValueError("scale_factor should be between 0 and 1.")

    h, w = img.shape[:2]

    # Calculate target dimensions for downsampling
    downsampled_size = (max(1, int(w * scale_factor)), max(1, int(h * scale_factor)))

    # Downsample using INTER_AREA for better quality in aggressive downsampling
    low_res_img = cv2.resize(img, downsampled_size, interpolation=cv2.INTER_AREA)

    # Upsample using INTER_NEAREST for strong pixelation effect
    return cv2.resize(low_res_img, (w, h), interpolation=cv2.INTER_NEAREST).astype(img.dtype)


def mask2bbox(mask, expansion_factor=1.0):
    # Find non-zero elements (object pixels)
    coords = np.argwhere(mask)
    
    # Extract bounding box coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Compute width and height
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Expand bounding box
    if expansion_factor > 0:
        x_min = max(0, int(x_min - width * expansion_factor / 2))
        y_min = max(0, int(y_min - height * expansion_factor / 2))
        x_max = min(mask.shape[1] - 1, int(x_max + width * expansion_factor / 2))
        y_max = min(mask.shape[0] - 1, int(y_max + height * expansion_factor / 2))

        # Recalculate width and height after expansion
        width = x_max - x_min + 1
        height = y_max - y_min + 1

    return (x_min, y_min, width, height)


def uncrop_pixel_height_map(cropped_map, full_img_shape, inv_trans):
    # Ensure type is preserved (e.g., int32 or int16)
    dtype = cropped_map.dtype

    # Convert to float32 for warpAffine
    cropped_map_f32 = cropped_map.astype(np.float32)

    # Warp back using inverse transform to the full image space
    full_map = cv2.warpAffine(
        cropped_map_f32,
        inv_trans,
        (full_img_shape[1], full_img_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return full_map.astype(dtype)


def uncrop_valid_mask(mask_cropped, full_image_shape, inv_trans):
    if mask_cropped.ndim == 3:
        full_mask = cv2.warpAffine(mask_cropped, inv_trans, (full_image_shape[1], full_image_shape[0]),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    else:
        full_mask = cv2.warpAffine(mask_cropped, inv_trans, (full_image_shape[1], full_image_shape[0]),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    return full_mask


def affine_scale_from_img2bb(img2bb):
    if isinstance(img2bb, np.ndarray):
        img2bb = torch.from_numpy(img2bb)
    if img2bb.ndim == 2:
        img2bb = img2bb.unsqueeze(0)

    A = img2bb[..., :2]
    sx = torch.sqrt(A[..., 0, 0]**2 + A[..., 1, 0]**2)
    sy = torch.sqrt(A[..., 0, 1]**2 + A[..., 1, 1]**2)
    s = 0.5 * (sx + sy)

    return s.squeeze().item()


def normalize_height_to_maxdim(
    gt, valid=None, crop_h=256, crop_w=256, keep_invalid_zero=True, to_uint8=False
):
    """
    Global translation:
      1) clamp negatives to 0
      2) compute max over valid pixels
      3) if max > cap: subtract offset = max - cap from ALL pixels
         so relative differences are preserved everywhere
         then clamp to [0, cap]
      4) if max <= cap: return unchanged
    This keeps zeros at 0 after clamping and preserves relative differences.
    """
    cap = float(max(crop_h, crop_w) - 1)  # e.g. 255 for 256x256

    # 1) clamp negatives first
    x = np.clip(gt, 0.0, None).astype(np.float32)

    # 2) valid mask and max
    if valid is not None:
        m = valid.astype(bool)
        has_valid = m.any()
        max_val = x[m].max() if has_valid else 0.0
    else:
        m = None
        has_valid = True
        max_val = float(x.max())

    # 3) decide offset
    if not has_valid or max_val <= cap:
        out = x.copy()
        if keep_invalid_zero and m is not None:
            out[~m] = 0.0
        if to_uint8:
            out = np.rint(out).astype(np.uint8)
        return out, 0.0

    offset = max_val - cap  # positive

    # Apply the same downward shift to all pixels
    out = x - offset

    # 4) clamp to [0, cap]
    out = np.clip(out, 0.0, cap)

    if keep_invalid_zero and m is not None:
        out[~m] = 0.0

    if to_uint8:
        out = np.rint(out).astype(np.uint8)

    return out, float(offset)