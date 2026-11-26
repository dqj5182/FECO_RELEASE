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

def create_ground_plane_mesh(slope_x, slope_z, intercept, center=(0,0,0), size=(1.0, 1.0), resolution=10):
    """
    Create a ground plane mesh aligned with fitted plane: y = slope_x * x + slope_z * z + intercept
    """
    width, depth = size
    cx, _, cz = center

    # Generate grid of x, z
    x_vals = np.linspace(cx - width / 2, cx + width / 2, resolution)
    z_vals = np.linspace(cz - depth / 2, cz + depth / 2, resolution)
    x_grid, z_grid = np.meshgrid(x_vals, z_vals)

    # Compute corresponding y values
    y_grid = slope_x * x_grid + slope_z * z_grid + intercept

    # Flatten to vertices (N x 3)
    vertices = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)

    # Create faces (two triangles per quad)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])
    faces = np.array(faces)

    # Create and return the mesh
    ground_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return ground_mesh


def create_ground_plane_mesh_xy(slope_x, slope_y, intercept, center=(0, 0, 0), size=(1.0, 1.0), resolution=10):
    """Create a plane z = slope_x * x + slope_y * y + intercept."""
    width, height = size
    cx, cy, _ = center
    x_vals = np.linspace(cx - width / 2, cx + width / 2, resolution)
    y_vals = np.linspace(cy - height / 2, cy + height / 2, resolution)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_grid = slope_x * x_grid + slope_y * y_grid + intercept
    vertices = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)


def compute_ground_normal(slope_x, slope_z):
    """
    Compute the unit normal vector of the ground plane in camera space.
    Plane is defined as: y = slope_x * x + slope_z * z + intercept
    """
    n = np.array([slope_x, -1.0, slope_z])
    n_unit = n / np.linalg.norm(n)
    return n_unit


def create_ground_plane_mesh_xz(slope_x, slope_z, intercept,
                                center=(0.0, 0.0, 0.0),
                                size=(1.0, 1.0),
                                resolution=10):
    """
    Create a ground plane mesh parameterized over (x, z):
        y(x, z) = slope_x * x + slope_z * z + intercept

    Args:
        slope_x, slope_z, intercept: plane coefficients in y = a*x + b*z + c
        center: (cx, cy, cz) to center the grid in world space
        size: (width_x, depth_z) extents along x and z
        resolution: grid resolution per axis

    Returns:
        trimesh.Trimesh
    """
    width_x, depth_z = size
    cx, cy, cz = center

    # Grid over x and z around the provided center
    x_vals = np.linspace(cx - width_x / 2.0, cx + width_x / 2.0, resolution)
    z_vals = np.linspace(cz - depth_z / 2.0, cz + depth_z / 2.0, resolution)
    x_grid, z_grid = np.meshgrid(x_vals, z_vals, indexing='ij')  # shape (R, R)

    # Compute y from the plane equation
    y_grid = slope_x * x_grid + slope_z * z_grid + intercept

    # Pack vertices as (x, y, z)
    vertices = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)

    # Build faces: two triangles per quad
    faces = []
    R = resolution
    for i in range(R - 1):
        for j in range(R - 1):
            idx = i * R + j
            # quad corners in row-major with indexing='ij':
            # (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            a = idx
            b = idx + R
            c = idx + 1
            d = idx + R + 1
            # choose winding to keep normals consistent (right-hand rule)
            faces.append([a, b, c])
            faces.append([c, b, d])
    faces = np.asarray(faces, dtype=np.int32)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def compute_signed_distances_to_plane(vertices, slope_x, slope_z, intercept):
    """
    Compute signed perpendicular distances from 3D vertices to a plane defined as y = slope_x * x + slope_z * z + intercept.

    Parameters:
        vertices (np.ndarray): Nx3 array of 3D points (x, y, z).
        slope_x (float): Plane coefficient for x.
        slope_z (float): Plane coefficient for z.
        intercept (float): Plane intercept term.

    Returns:
        np.ndarray: Signed distances from each point to the plane. Shape: (N,)
    """
    # Plane: ax - y + bz + c = 0  <=>  a = slope_x, b = slope_z, c = intercept
    plane_normal = np.array([slope_x, -1.0, slope_z])
    normal_norm = np.linalg.norm(plane_normal)
    plane_normal_unit = plane_normal / normal_norm

    # Plane offset term d (in normalized space)
    plane_d = intercept / normal_norm

    # Signed distances
    signed_distances = (vertices @ plane_normal_unit) + plane_d
    return signed_distances


def extract_ground_plane(ground_mesh):
    """
    Fit y = slope_x * x + slope_z * z + intercept using pure NumPy.

    Returns:
        slope_x, slope_z, intercept
    """
    verts = ground_mesh.vertices
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    A = np.column_stack([x, z, np.ones_like(x)])
    params = np.linalg.lstsq(A, y, rcond=None)[0]

    slope_x, slope_z, intercept = params
    return slope_x, slope_z, intercept