import numpy as np


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / (cam_coord[:,2] + 1e-5) * f[0] + c[0]
    y = cam_coord[:,1] / (cam_coord[:,2] + 1e-5) * f[1] + c[1]
    z = cam_coord[:,2] + 1e-5
    return np.stack((x,y,z),1)


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint