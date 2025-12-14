import torch
import trimesh
import numpy as np
from scipy.spatial import Delaunay
from plyfile import PlyData, PlyElement
from sklearn.linear_model import RANSACRegressor


def extract_partial_mesh(mesh: trimesh.Trimesh, vertex_indices: np.ndarray) -> trimesh.Trimesh:
    vertex_indices = np.asarray(vertex_indices)
    vertex_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}

    # Keep only faces where all 3 vertices are in the subset
    face_mask = np.all(np.isin(mesh.faces, vertex_indices), axis=1)
    selected_faces = mesh.faces[face_mask]

    # Remap face indices to the new vertex ordering
    remapped_faces = np.vectorize(vertex_index_map.get)(selected_faces)

    # Get vertex positions in the same order as vertex_indices
    selected_vertices = mesh.vertices[vertex_indices]

    return trimesh.Trimesh(vertices=selected_vertices, faces=remapped_faces, process=False)


def build_new_faces(mesh_vertices: np.ndarray,
                                 source_indices: np.ndarray,
                                 target_indices: np.ndarray,
                                 projection_dims=(0, 2)) -> np.ndarray:
    """
    Triangulates a 2D projection of the subset of vertices shared between two index lists.
    
    Parameters:
        mesh_vertices (np.ndarray): (N, 3) array of vertex positions.
        source_indices (np.ndarray): Full set of vertex indices (e.g., right foot).
        target_indices (np.ndarray): Subset to match (e.g., right leg).
        projection_dims (tuple): Dimensions to project to 2D (default: (0, 2) → XZ plane).
        
    Returns:
        np.ndarray: (M, 3) triangle face indices (global vertex indices).
    """
    # Find positions in source that are also in target
    mask = np.isin(source_indices, target_indices)
    positions = np.nonzero(mask)[0]               # indices in source_indices
    global_indices = source_indices[positions]    # actual vertex indices
    
    # Project to 2D
    points_2d = mesh_vertices[global_indices][:, projection_dims]
    
    # Triangulate
    tri = Delaunay(points_2d)
    
    # Map local indices back to global mesh vertex indices, and fix winding
    faces = global_indices[tri.simplices][:, [0, 2, 1]]
    
    return faces