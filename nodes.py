"""
GaussianSplatToMesh — Convert PLY_DATA (Gaussian Splat point cloud) to TRIMESH.

Uses scipy + trimesh + sklearn (no Open3D dependency) for surface reconstruction.
Compatible with Python 3.13 and ComfyUI v0.22.0.

Supported methods:
  - alpha_shape: Alpha shape triangulation via scipy Delaunay
  - ball_pivoting: Approximate ball-pivoting via local Delaunay patches
  - marching_cubes: Volumetric marching cubes via scipy/skimage
"""

import numpy as np
import torch
import trimesh as Trimesh
from scipy.spatial import Delaunay, cKDTree


# ── SH coefficient C0 for degree-0 spherical harmonics ──────────────────────
SH_C0 = 0.28209479177387814


def _extract_points_and_colors(ply_data):
    """
    Extract 3D point positions and RGB colors from PLY_DATA dict.

    PLY_DATA structure (from VNCCS_WorldMirrorV2_3D):
      - pts3d: [B, S, H, W, 3] or None
      - pts3d_filtered: [N, 3] tensor or None
      - splats: dict with keys {means, quats, scales, opacities, sh, weights, ...}
      - images: [B, S, H, W, 3] tensor (input images)
      - filter_mask: [N] boolean tensor or None

    Returns:
      points: np.ndarray [N, 3] float64
      colors: np.ndarray [N, 3] float64 in [0, 1]
    """
    points = None
    colors = None

    # ── Strategy 1: Use splats (Gaussian Splat parameters) ────────────────
    splats = ply_data.get("splats")
    if splats is not None and "means" in splats:
        means = splats["means"]
        if isinstance(means, torch.Tensor):
            # means shape: [B, N, 3] or [N, 3]
            if means.dim() == 3:
                means = means[0]  # take first batch
            points = means.detach().cpu().float().numpy()

        # Extract colors from SH coefficients
        sh = splats.get("sh")
        if sh is not None and isinstance(sh, torch.Tensor):
            if sh.dim() == 4:
                sh = sh[0]  # [N, C, 3] or [N, 1, 3]
            elif sh.dim() == 3:
                pass  # already [N, C, 3] or [N, SH_degree, 3]

            # Take DC component (degree 0)
            if sh.dim() == 3:
                sh_dc = sh[:, 0, :]  # [N, 3]
            elif sh.dim() == 2:
                sh_dc = sh  # [N, 3]
            else:
                sh_dc = sh.reshape(-1, 3)

            # Convert SH DC to RGB: color = sigmoid(sh * C0)
            # or color = 0.5 + C0 * sh (linear approximation used in save_utils)
            rgb = 0.5 + SH_C0 * sh_dc.detach().cpu().float().numpy()
            colors = np.clip(rgb, 0.0, 1.0)

        # Fallback: use colors key directly if available
        if colors is None:
            raw_colors = splats.get("colors")
            if raw_colors is not None and isinstance(raw_colors, torch.Tensor):
                if raw_colors.dim() == 3:
                    raw_colors = raw_colors[0]
                colors = raw_colors.detach().cpu().float().numpy()
                if colors.max() > 1.0:
                    colors = colors / 255.0
                colors = np.clip(colors, 0.0, 1.0)

    # ── Strategy 2: Use pts3d_filtered ────────────────────────────────────
    if points is None:
        pts_filtered = ply_data.get("pts3d_filtered")
        if pts_filtered is not None and isinstance(pts_filtered, torch.Tensor):
            points = pts_filtered.detach().cpu().float().numpy().reshape(-1, 3)

    # ── Strategy 3: Use pts3d (full point map) ────────────────────────────
    if points is None:
        pts3d = ply_data.get("pts3d")
        if pts3d is not None and isinstance(pts3d, torch.Tensor):
            # Shape: [B, S, H, W, 3]
            pts = pts3d[0]  # first batch -> [S, H, W, 3]
            points = pts.detach().cpu().float().numpy().reshape(-1, 3)

            # Try to get colors from images
            images = ply_data.get("images")
            if images is not None and isinstance(images, torch.Tensor):
                imgs = images[0]  # [S, H, W, 3] or [S, 3, H, W]
                if imgs.shape[-1] != 3 and imgs.shape[1] == 3:
                    imgs = imgs.permute(0, 2, 3, 1)  # [S, H, W, 3]
                colors = imgs.detach().cpu().float().numpy().reshape(-1, 3)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                colors = np.clip(colors, 0.0, 1.0)

            # Apply filter mask if available
            fmask = ply_data.get("filter_mask")
            if fmask is not None and isinstance(fmask, torch.Tensor):
                mask_np = fmask.detach().cpu().numpy().astype(bool).reshape(-1)
                if mask_np.shape[0] == points.shape[0]:
                    points = points[mask_np]
                    if colors is not None and colors.shape[0] == mask_np.shape[0]:
                        colors = colors[mask_np]

    if points is None:
        raise ValueError(
            "PLY_DATA does not contain valid point data. "
            "Expected keys: 'splats.means', 'pts3d_filtered', or 'pts3d'"
        )

    # Filter out NaN/Inf points
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is not None:
        colors = colors[valid] if colors.shape[0] > valid.sum() or colors.shape[0] == valid.shape[0] else colors[:valid.sum()]

    # Ensure colors array matches points
    if colors is None or colors.shape[0] != points.shape[0]:
        colors = np.ones((points.shape[0], 3), dtype=np.float64) * 0.7  # default gray

    print(f"[GaussianSplatToMesh] Extracted {points.shape[0]} points")
    return points.astype(np.float64), colors.astype(np.float64)


def _estimate_normals(points, k=30):
    """
    Estimate point normals using PCA on k-nearest neighbors.

    Args:
        points: [N, 3] numpy array
        k: number of neighbors for normal estimation

    Returns:
        normals: [N, 3] numpy array (unit normals)
    """
    tree = cKDTree(points)
    k = min(k, points.shape[0])
    _, indices = tree.query(points, k=k)

    normals = np.zeros_like(points)
    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        centroid = neighbors.mean(axis=0)
        cov = (neighbors - centroid).T @ (neighbors - centroid)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]  # smallest eigenvalue = normal direction
        except np.linalg.LinAlgError:
            normals[i] = [0, 0, 1]

    # Orient normals consistently (toward centroid)
    centroid = points.mean(axis=0)
    for i in range(normals.shape[0]):
        if np.dot(normals[i], points[i] - centroid) < 0:
            normals[i] = -normals[i]

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normals = normals / norms

    return normals


def _downsample_points(points, colors, max_points=200000, voxel_size=None):
    """
    Downsample point cloud if too large.

    Args:
        points: [N, 3]
        colors: [N, 3]
        max_points: maximum number of points
        voxel_size: if set, use voxel grid downsampling

    Returns:
        downsampled points and colors
    """
    if points.shape[0] <= max_points:
        return points, colors

    if voxel_size is not None:
        # Voxel grid downsampling
        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_idx], colors[unique_idx]
    else:
        # Random downsampling
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        indices.sort()
        return points[indices], colors[indices]


def _remove_outliers(points, colors, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from point cloud.

    Args:
        points: [N, 3]
        colors: [N, 3]
        nb_neighbors: number of neighbors for statistics
        std_ratio: standard deviation multiplier threshold

    Returns:
        filtered points and colors
    """
    if points.shape[0] < nb_neighbors + 1:
        return points, colors

    tree = cKDTree(points)
    k = min(nb_neighbors + 1, points.shape[0])
    distances, _ = tree.query(points, k=k)
    mean_distances = distances[:, 1:].mean(axis=1)  # exclude self

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_distances < threshold
    print(f"[GaussianSplatToMesh] Outlier removal: {points.shape[0]} -> {mask.sum()} points")
    return points[mask], colors[mask]


def _alpha_shape_mesh(points, colors, alpha=0.0):
    """
    Create mesh using Delaunay triangulation with alpha shape filtering.

    Args:
        points: [N, 3] numpy array
        colors: [N, 3] numpy array in [0, 1]
        alpha: alpha value for filtering (0 = convex hull, larger = more detail)

    Returns:
        trimesh.Trimesh object
    """
    print(f"[GaussianSplatToMesh] Running alpha shape (alpha={alpha})...")

    # Compute Delaunay triangulation
    tri = Delaunay(points)
    tetrahedra = tri.simplices  # [M, 4]

    # Extract surface triangles from tetrahedra
    # Each tetrahedron has 4 triangular faces
    faces_set = set()
    face_count = {}

    for tet in tetrahedra:
        # 4 faces of a tetrahedron
        for face in [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]:
            face_count[face] = face_count.get(face, 0) + 1

    # Surface faces appear in exactly one tetrahedron (boundary faces)
    surface_faces = []
    for face, count in face_count.items():
        if count == 1:
            surface_faces.append(face)

    if alpha > 0:
        # Filter by circumradius
        filtered_faces = []
        for face in surface_faces:
            pts = points[list(face)]
            # Compute circumradius of triangle
            a = np.linalg.norm(pts[1] - pts[0])
            b = np.linalg.norm(pts[2] - pts[1])
            c = np.linalg.norm(pts[0] - pts[2])
            s = (a + b + c) / 2
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
            if area > 1e-10:
                circumradius = (a * b * c) / (4 * area)
                if circumradius < 1.0 / alpha:
                    filtered_faces.append(face)
        surface_faces = filtered_faces

    if len(surface_faces) == 0:
        raise ValueError("Alpha shape produced no faces. Try adjusting alpha parameter.")

    faces_array = np.array(surface_faces, dtype=np.int64)

    # Create vertex colors (0-255 uint8)
    vertex_colors_uint8 = (colors * 255).astype(np.uint8)
    # Add alpha channel
    vertex_colors_rgba = np.hstack([
        vertex_colors_uint8,
        np.full((vertex_colors_uint8.shape[0], 1), 255, dtype=np.uint8)
    ])

    mesh = Trimesh.Trimesh(
        vertices=points,
        faces=faces_array,
        vertex_colors=vertex_colors_rgba,
        process=True,
    )

    print(f"[GaussianSplatToMesh] Alpha shape: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def _marching_cubes_mesh(points, colors, resolution=128, padding=0.1):
    """
    Create mesh using marching cubes on a voxelized density field.

    Args:
        points: [N, 3] numpy array
        colors: [N, 3] numpy array in [0, 1]
        resolution: voxel grid resolution
        padding: padding ratio around point cloud bounds

    Returns:
        trimesh.Trimesh object
    """
    print(f"[GaussianSplatToMesh] Running marching cubes (resolution={resolution})...")

    from scipy.ndimage import gaussian_filter

    # Compute bounding box
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = pmax - pmin
    pad = extent * padding
    pmin -= pad
    pmax += pad
    extent = pmax - pmin

    # Create voxel grid
    voxel_size = extent / resolution
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    # Voxelize points (accumulate density)
    indices = ((points - pmin) / voxel_size).astype(np.int32)
    indices = np.clip(indices, 0, resolution - 1)

    for idx in indices:
        grid[idx[0], idx[1], idx[2]] += 1.0

    # Smooth the density field
    grid = gaussian_filter(grid, sigma=1.5)

    # Determine iso-level
    nonzero = grid[grid > 0]
    if len(nonzero) == 0:
        raise ValueError("No points in voxel grid")
    iso_level = np.percentile(nonzero, 20)

    # Marching cubes
    try:
        from skimage.measure import marching_cubes
        verts, faces, normals_mc, _ = marching_cubes(grid, level=iso_level)
    except ImportError:
        # Fallback: use scipy if skimage not available
        raise ImportError(
            "scikit-image is required for marching_cubes method. "
            "Please install it: pip install scikit-image"
        )

    # Transform vertices back to world coordinates
    verts = verts * voxel_size + pmin

    # Transfer colors from nearest points
    tree = cKDTree(points)
    _, nearest_idx = tree.query(verts, k=1)
    vert_colors = colors[nearest_idx]
    vert_colors_uint8 = (vert_colors * 255).astype(np.uint8)
    vert_colors_rgba = np.hstack([
        vert_colors_uint8,
        np.full((vert_colors_uint8.shape[0], 1), 255, dtype=np.uint8)
    ])

    mesh = Trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=vert_colors_rgba,
        process=True,
    )

    print(f"[GaussianSplatToMesh] Marching cubes: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def _ball_pivoting_mesh(points, colors, radius_factor=1.5):
    """
    Approximate ball-pivoting using local Delaunay triangulation patches.

    This is a simplified version that creates a mesh by:
    1. Computing local Delaunay triangulations in overlapping patches
    2. Filtering triangles by edge length (simulating ball radius)
    3. Merging patches

    Args:
        points: [N, 3] numpy array
        colors: [N, 3] numpy array in [0, 1]
        radius_factor: multiplier for average nearest-neighbor distance to set max edge length

    Returns:
        trimesh.Trimesh object
    """
    print(f"[GaussianSplatToMesh] Running ball pivoting approximation (radius_factor={radius_factor})...")

    # Compute average nearest-neighbor distance for radius estimation
    tree = cKDTree(points)
    k = min(6, points.shape[0])
    distances, _ = tree.query(points, k=k)
    avg_nn_dist = distances[:, 1:].mean()
    max_edge_length = avg_nn_dist * radius_factor * 3

    # Use Delaunay triangulation
    tri = Delaunay(points)
    tetrahedra = tri.simplices

    # Extract all triangular faces
    face_count = {}
    for tet in tetrahedra:
        for face in [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]:
            face_count[face] = face_count.get(face, 0) + 1

    # Keep boundary faces (appear once) and filter by edge length
    surface_faces = []
    for face, count in face_count.items():
        if count == 1:
            pts = points[list(face)]
            edges = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[0] - pts[2]),
            ]
            if max(edges) < max_edge_length:
                surface_faces.append(face)

    if len(surface_faces) == 0:
        raise ValueError("Ball pivoting produced no faces. Try adjusting radius_factor.")

    faces_array = np.array(surface_faces, dtype=np.int64)

    vertex_colors_uint8 = (colors * 255).astype(np.uint8)
    vertex_colors_rgba = np.hstack([
        vertex_colors_uint8,
        np.full((vertex_colors_uint8.shape[0], 1), 255, dtype=np.uint8)
    ])

    mesh = Trimesh.Trimesh(
        vertices=points,
        faces=faces_array,
        vertex_colors=vertex_colors_rgba,
        process=True,
    )

    print(f"[GaussianSplatToMesh] Ball pivoting: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Node: GaussianSplatToMesh
# ─────────────────────────────────────────────────────────────────────────────
class GaussianSplatToMesh:
    """
    Convert PLY_DATA (Gaussian Splat point cloud from HY-World 2.0) to TRIMESH.

    Extracts 3D points and colors from the Gaussian Splat data, performs
    surface reconstruction, and outputs a trimesh.Trimesh object compatible
    with Hy3DExportMesh for GLB/OBJ/PLY/STL export.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
            },
            "optional": {
                "method": (["alpha_shape", "ball_pivoting", "marching_cubes"], {
                    "default": "alpha_shape",
                    "tooltip": (
                        "Surface reconstruction method:\n"
                        "- alpha_shape: Delaunay-based alpha shape (fast, good for dense clouds)\n"
                        "- ball_pivoting: Approximate ball pivoting (good edge filtering)\n"
                        "- marching_cubes: Volumetric reconstruction (smooth, requires scikit-image)"
                    ),
                }),
                "alpha": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": (
                        "Alpha parameter for alpha_shape method. "
                        "0 = convex hull, larger values = more detail/holes. "
                        "Typical range: 0.5-10.0"
                    ),
                }),
                "resolution": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 512,
                    "step": 16,
                    "tooltip": "Voxel grid resolution for marching_cubes method",
                }),
                "radius_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Radius multiplier for ball_pivoting method",
                }),
                "max_points": ("INT", {
                    "default": 100000,
                    "min": 10000,
                    "max": 500000,
                    "step": 10000,
                    "tooltip": "Maximum number of points (downsampled if exceeded)",
                }),
                "remove_outliers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove statistical outlier points before reconstruction",
                }),
                "outlier_std_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Standard deviation ratio for outlier removal (lower = more aggressive)",
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "3D/mesh"
    DESCRIPTION = (
        "Convert PLY_DATA (Gaussian Splat point cloud from HY-World 2.0) to TRIMESH mesh. "
        "The output can be connected to Hy3DExportMesh for GLB/OBJ/PLY/STL export."
    )

    def convert(
        self,
        ply_data,
        method="alpha_shape",
        alpha=0.0,
        resolution=128,
        radius_factor=1.5,
        max_points=100000,
        remove_outliers=True,
        outlier_std_ratio=2.0,
    ):
        print(f"[GaussianSplatToMesh] Starting conversion (method={method})...")

        # ── 1. Extract points and colors ──────────────────────────────────
        points, colors = _extract_points_and_colors(ply_data)

        if points.shape[0] < 4:
            raise ValueError(
                f"Not enough points for mesh reconstruction: {points.shape[0]} points. "
                "Need at least 4 points."
            )

        # ── 2. Remove outliers ────────────────────────────────────────────
        if remove_outliers and points.shape[0] > 50:
            points, colors = _remove_outliers(
                points, colors,
                nb_neighbors=min(20, points.shape[0] - 1),
                std_ratio=outlier_std_ratio,
            )

        # ── 3. Downsample if needed ───────────────────────────────────────
        if points.shape[0] > max_points:
            print(f"[GaussianSplatToMesh] Downsampling: {points.shape[0]} -> {max_points}")
            points, colors = _downsample_points(points, colors, max_points=max_points)

        # ── 4. Surface reconstruction ─────────────────────────────────────
        if method == "alpha_shape":
            mesh = _alpha_shape_mesh(points, colors, alpha=alpha)
        elif method == "ball_pivoting":
            mesh = _ball_pivoting_mesh(points, colors, radius_factor=radius_factor)
        elif method == "marching_cubes":
            mesh = _marching_cubes_mesh(points, colors, resolution=resolution)
        else:
            raise ValueError(f"Unknown method: {method}")

        print(
            f"[GaussianSplatToMesh] Done: {len(mesh.vertices)} vertices, "
            f"{len(mesh.faces)} faces"
        )

        return (mesh,)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "GaussianSplatToMesh": GaussianSplatToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GaussianSplatToMesh": "🔷 Gaussian Splat to Mesh",
}
