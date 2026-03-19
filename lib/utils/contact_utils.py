import gc
import numpy as np
from trimesh.proximity import ProximityQuery


foot_vertex_num = 265


def get_ho_contact_and_offset(mesh_hand, mesh_obj, c_thres):
    pq = ProximityQuery(mesh_obj)
    obj_coord_c, dist, obj_coord_c_idx = pq.on_surface(mesh_hand.vertices.astype(np.float32))

    is_contact_h = (dist < c_thres)
    contact_h = (1. * is_contact_h).astype(np.float32)

    contact_valid = np.ones((foot_vertex_num, 1))
    inter_coord_valid = np.ones((foot_vertex_num))

    del pq
    gc.collect()

    return np.array(contact_h), np.array(obj_coord_c), contact_valid, inter_coord_valid


def get_contact_thres(backbone_type='vit-h-14'):
    if backbone_type == 'vit-h-14':
        return 0.5
    elif backbone_type == 'vit-l-16':
        return 0.45
    elif backbone_type == 'vit-b-16':
        return 0.5
    elif backbone_type == 'vit-s-16':
        return 0.65
    elif backbone_type == 'resnet-152':
        return 0.5
    elif backbone_type == 'resnet-101':
        return 0.6
    elif backbone_type == 'resnet-50':
        return 0.4
    elif backbone_type == 'resnet-34':
        return 0.7
    elif backbone_type == 'resnet-18':
        return 0.45
    else:
        raise NotImplementedError