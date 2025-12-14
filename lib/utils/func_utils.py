import numpy as np


def get_bbox_body(
    joint_img, joint_valid,
    r_ankle_idx=11, l_ankle_idx=12,
    foot_frac=0.12,
    ankle_in_box=0.375,   # midpoint of 0.15 and 0.60
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