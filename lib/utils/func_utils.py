import os
import re
import cv2
import yaml
import numpy as np
try:
    from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR
    _TJ = TurboJPEG()
except Exception:
    _TJ = None


def load_img(path: str, order: str = 'RGB', ignore_orientation: bool = True):
    ext = os.path.splitext(path)[1].lower()

    if _TJ is not None and ext in ('.jpg', '.jpeg'):
        with open(path, 'rb') as f:
            data = f.read()

        pixel_format = TJPF_RGB if order.upper() == 'RGB' else TJPF_BGR
        img = _TJ.decode(data, pixel_format=pixel_format)  # uint8

        if img is None:
            raise IOError(f"Fail to decode JPEG via TurboJPEG: {path}")
        return img

    flags = cv2.IMREAD_COLOR
    if ignore_orientation:
        flags |= cv2.IMREAD_IGNORE_ORIENTATION

    img = cv2.imread(path, flags)
    if img is None:
        raise IOError(f"Fail to read {path}")

    if order.upper() == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_bbox(joint_img, joint_valid, expansion_factor=1.0):
    # Filter valid joints
    valid_coords = joint_img[joint_valid == 1]
    if valid_coords.shape[0] == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)

    # Compute center
    min_xy = valid_coords.min(axis=0)
    max_xy = valid_coords.max(axis=0)
    center = (min_xy + max_xy) / 2.0

    # Compute square scale
    size = max(max_xy - min_xy) * expansion_factor

    # Create bbox: (xmin, ymin, width, height)
    top_left = center - size / 2.0
    bbox = np.array([top_left[0], top_left[1], size, size], dtype=np.float32)

    return bbox


def get_bbox_pennaction(
    joint_img, joint_valid,
    r_ankle_idx=11, l_ankle_idx=12,
    foot_frac=0.12,
    ankle_in_box=0.375,
    image_size=None
):
    joint_img = np.asarray(joint_img, dtype=np.float32)
    joint_valid = np.asarray(joint_valid).astype(bool)

    valid_pts = joint_img[joint_valid]
    if valid_pts.size == 0:
        z = np.array([0, 0, 0, 0], dtype=np.float32)
        return z.copy(), z.copy()

    body_min = valid_pts.min(axis=0)
    body_max = valid_pts.max(axis=0)
    body_size = float(max(body_max - body_min))
    foot_size = float(max(1.0, foot_frac * body_size))

    def _ankle(idx):
        ok = 0 <= idx < len(joint_img) and joint_valid[idx]
        return joint_img[idx] if ok else None

    def _make_box(ankle):
        if ankle is None:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        w = h = foot_size
        x = ankle[0] - 0.5 * w
        y = ankle[1] - ankle_in_box * h
        box = np.array([x, y, w, h], dtype=np.float32)
        if image_size is not None:
            H, W = int(image_size[0]), int(image_size[1])
            x, y, w, h = box
            x = float(np.clip(x, 0, max(W - 1, 0)))
            y = float(np.clip(y, 0, max(H - 1, 0)))
            w = float(max(0.0, min(w, W - x)))
            h = float(max(0.0, min(h, H - y)))
            box = np.array([x, y, w, h], dtype=np.float32)
        return box

    left_box  = _make_box(_ankle(l_ankle_idx))
    right_box = _make_box(_ankle(r_ankle_idx))
    return left_box, right_box


def get_bbox_body(
    joint_img, joint_valid,
    r_ankle_idx=11, l_ankle_idx=12,
    foot_frac=0.12,
    ankle_in_box=0.375,
    image_size=None
):
    joint_img = np.asarray(joint_img, dtype=np.float32)
    joint_valid = np.asarray(joint_valid).astype(bool)

    valid_pts = joint_img[joint_valid]
    if valid_pts.size == 0:
        z = np.array([0, 0, 0, 0], dtype=np.float32)
        return z.copy(), z.copy()

    body_min = valid_pts.min(axis=0)
    body_max = valid_pts.max(axis=0)
    body_size = float(max(body_max - body_min))
    foot_size = float(max(1.0, foot_frac * body_size))

    def _ankle(idx):
        ok = 0 <= idx < len(joint_img) and joint_valid[idx]
        return joint_img[idx] if ok else None

    def _make_box(ankle):
        if ankle is None:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        w = h = foot_size
        x = ankle[0] - 0.5 * w
        y = ankle[1] - ankle_in_box * h
        box = np.array([x, y, w, h], dtype=np.float32)
        if image_size is not None:
            H, W = int(image_size[0]), int(image_size[1])
            x, y, w, h = box
            x = float(np.clip(x, 0, max(W - 1, 0)))
            y = float(np.clip(y, 0, max(H - 1, 0)))
            w = float(max(0.0, min(w, W - x)))
            h = float(max(0.0, min(h, H - y)))
            box = np.array([x, y, w, h], dtype=np.float32)
        return box

    left_box  = _make_box(_ankle(l_ankle_idx))
    right_box = _make_box(_ankle(r_ankle_idx))
    return left_box, right_box


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def path_natural_key(path_str):
    parts = path_str.split(os.sep)
    key = []
    for part in parts:
        key += natural_keys(part)
    return key


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg