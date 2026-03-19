import trimesh
import numpy as np
from scipy.spatial import Delaunay


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