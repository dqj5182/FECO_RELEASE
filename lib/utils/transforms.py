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

def apply_homogeneous_transformation_np(vertices, transform_matrix):
    # Convert vertices to homogeneous coordinates (add a column of ones)
    num_verts = vertices.shape[0]
    verts_homogeneous = np.concatenate([vertices, np.ones((num_verts, 1), dtype=vertices.dtype)], axis=1)  # Shape (num_verts, 4)

    # Apply the homogeneous transformation
    transformed_homogeneous = np.dot(transform_matrix, verts_homogeneous.T).T  # Shape (num_verts, 4)

    # Convert back to Cartesian coordinates (divide by the homogeneous component)
    transformed_vertices = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3][:, None]  # Shape (num_verts, 3)

    return transformed_vertices