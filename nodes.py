"""
GaussianSplatToMesh — Convert PLY_DATA (Gaussian Splat point cloud) to TRIMESH.

Uses scipy + trimesh + sklearn (no Open3D dependency) for surface reconstruction.
Compatible with Python 3.13 and ComfyUI v0.22.0.

Supported methods:
  - marching_cubes: Volumetric marching cubes via scipy/skimage (recommended)
  - poisson_like: Screened-Poisson-like surface reconstruction
  - alpha_shape: Alpha shape triangulation via scipy Delaunay
  - ball_pivoting: Approximate ball-pivoting via local Delaunay patches

v2.0 — Major quality improvements:
  - Proper vertex color transfer with KNN interpolation
  - Improved marching cubes with KDE-based density field
  - Poisson-like surface reconstruction method
  - Voxel grid downsampling (preserves spatial structure)
  - Density-based outlier removal
  - Higher default max_points (200K)

v2.1 — Color extraction fix:
  - Fixed bug where Strategy 2 (pts3d_filtered) left colors as None
  - Added multiple color source fallbacks: images, features_dc, sh_coeffs/shs_0
  - Proper filter_mask application for color data
  - Added mesh normal computation after surface reconstruction
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
    """
    points = None
    colors = None

    # ── Strategy 1: Use splats (Gaussian Splat parameters) ────────────────
    splats = ply_data.get("splats")
    if splats is not None and "means" in splats:
        means = splats["means"]
        if isinstance(means, torch.Tensor):
            if means.dim() == 3:
                means = means[0]
            points = means.detach().cpu().float().numpy()

        # Extract colors from SH coefficients
        sh = splats.get("sh")
        if sh is not None and isinstance(sh, torch.Tensor):
            if sh.dim() == 4:
                sh = sh[0]
            if sh.dim() == 3:
                sh_dc = sh[:, 0, :]
            elif sh.dim() == 2:
                sh_dc = sh
            else:
                sh_dc = sh.reshape(-1, 3)
            rgb = 0.5 + SH_C0 * sh_dc.detach().cpu().float().numpy()
            colors = np.clip(rgb, 0.0, 1.0)

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
            print(f"[GaussianSplatToMesh] Strategy 2: pts3d_filtered -> {points.shape[0]} points")

    # ── Strategy 2.5: Color recovery when points found but colors missing ─
    # This handles the case where Strategy 2 found points via pts3d_filtered
    # but colors were not extracted (the original bug).
    if points is not None and colors is None:
        # Color source 1: images key
        images = ply_data.get("images")
        if images is not None and isinstance(images, torch.Tensor):
            imgs = images[0] if images.dim() == 5 else images
            if imgs.dim() == 4 and imgs.shape[1] == 3:  # [B, 3, H, W]
                imgs = imgs.permute(0, 2, 3, 1)  # -> [B, H, W, 3]
            all_colors = imgs.detach().cpu().float().numpy().reshape(-1, 3)
            if all_colors.max() > 1.0:
                all_colors = all_colors / 255.0

            # Apply filter_mask if available
            fmask = ply_data.get("filter_mask")
            if fmask is not None and isinstance(fmask, torch.Tensor):
                mask_np = fmask.detach().cpu().numpy().astype(bool).reshape(-1)
                if mask_np.shape[0] == all_colors.shape[0]:
                    all_colors = all_colors[mask_np]

            if all_colors.shape[0] == points.shape[0]:
                colors = np.clip(all_colors, 0.0, 1.0)
                print(f"[GaussianSplatToMesh] Colors from images: {colors.shape}, "
                      f"range [{colors.min():.3f}, {colors.max():.3f}]")

        # Color source 2: features_dc (SH degree-0 coefficients)
        if colors is None:
            features_dc = ply_data.get("features_dc")
            if features_dc is not None and isinstance(features_dc, torch.Tensor):
                sh0 = features_dc.detach().cpu().float().numpy()
                if sh0.ndim == 3:  # [N, 1, 3]
                    sh0 = sh0.squeeze(1)
                if sh0.ndim == 2 and sh0.shape[1] >= 3:
                    rgb = sh0[:, :3] * SH_C0 + 0.5

                    # Apply filter_mask if available
                    fmask = ply_data.get("filter_mask")
                    if fmask is not None and isinstance(fmask, torch.Tensor):
                        mask_np = fmask.detach().cpu().numpy().astype(bool).reshape(-1)
                        if mask_np.shape[0] == rgb.shape[0]:
                            rgb = rgb[mask_np]

                    if rgb.shape[0] == points.shape[0]:
                        colors = np.clip(rgb, 0.0, 1.0)
                        print(f"[GaussianSplatToMesh] Colors from features_dc (SH0): {colors.shape}, "
                              f"range [{colors.min():.3f}, {colors.max():.3f}]")

        # Color source 3: sh_coeffs or shs_0 (alternative SH keys)
        if colors is None:
            for key in ["sh_coeffs", "shs_0"]:
                sh_data = ply_data.get(key)
                if sh_data is not None and isinstance(sh_data, torch.Tensor):
                    sh_np = sh_data.detach().cpu().float().numpy()
                    if sh_np.ndim == 3:
                        sh_np = sh_np.squeeze(1)
                    if sh_np.ndim == 2 and sh_np.shape[1] >= 3:
                        rgb = sh_np[:, :3] * SH_C0 + 0.5

                        fmask = ply_data.get("filter_mask")
                        if fmask is not None and isinstance(fmask, torch.Tensor):
                            mask_np = fmask.detach().cpu().numpy().astype(bool).reshape(-1)
                            if mask_np.shape[0] == rgb.shape[0]:
                                rgb = rgb[mask_np]

                        if rgb.shape[0] == points.shape[0]:
                            colors = np.clip(rgb, 0.0, 1.0)
                            print(f"[GaussianSplatToMesh] Colors from {key} (SH0): {colors.shape}")
                            break

    # ── Strategy 3: Use pts3d (full point map) ────────────────────────────
    if points is None:
        pts3d = ply_data.get("pts3d")
        if pts3d is not None and isinstance(pts3d, torch.Tensor):
            pts = pts3d[0]
            points = pts.detach().cpu().float().numpy().reshape(-1, 3)

            images = ply_data.get("images")
            if images is not None and isinstance(images, torch.Tensor):
                imgs = images[0]
                if imgs.shape[-1] != 3 and imgs.shape[1] == 3:
                    imgs = imgs.permute(0, 2, 3, 1)
                colors = imgs.detach().cpu().float().numpy().reshape(-1, 3)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                colors = np.clip(colors, 0.0, 1.0)

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

    # Log available keys for debugging when colors are missing
    if colors is None:
        available_keys = [k for k in ply_data.keys() if ply_data[k] is not None]
        print(f"[GaussianSplatToMesh] DEBUG: Available PLY_DATA keys: {available_keys}")
        for k in available_keys:
            v = ply_data[k]
            if isinstance(v, torch.Tensor):
                print(f"[GaussianSplatToMesh] DEBUG:   {k}: Tensor shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"[GaussianSplatToMesh] DEBUG:   {k}: dict with keys={list(v.keys())}")
            else:
                print(f"[GaussianSplatToMesh] DEBUG:   {k}: {type(v).__name__}")

    # Filter out NaN/Inf points
    valid = np.isfinite(points).all(axis=1)
    if colors is not None and colors.shape[0] == points.shape[0]:
        colors = colors[valid]
    points = points[valid]

    if colors is None or colors.shape[0] != points.shape[0]:
        print("[GaussianSplatToMesh] WARNING: Color data missing or mismatched, using default gray")
        colors = np.ones((points.shape[0], 3), dtype=np.float64) * 0.7

    print(f"[GaussianSplatToMesh] Extracted {points.shape[0]} points, "
          f"color range: [{colors.min():.3f}, {colors.max():.3f}]")
    return points.astype(np.float64), colors.astype(np.float64)


def _estimate_normals_fast(points, k=20):
    """Fast vectorized normal estimation using PCA on k-nearest neighbors."""
    N = points.shape[0]
    tree = cKDTree(points)
    k = min(k, N)
    _, indices = tree.query(points, k=k)

    normals = np.zeros((N, 3), dtype=np.float64)
    batch_size = 10000
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_indices = indices[start:end]
        neighbors = points[batch_indices]
        centroids = neighbors.mean(axis=1)
        centered = neighbors - centroids[:, np.newaxis, :]
        covs = np.einsum('bki,bkj->bij', centered, centered)
        for i in range(end - start):
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(covs[i])
                normals[start + i] = eigenvectors[:, 0]
            except np.linalg.LinAlgError:
                normals[start + i] = [0, 0, 1]

    # Orient normals outward
    centroid = points.mean(axis=0)
    directions = points - centroid
    dot_products = np.sum(normals * directions, axis=1)
    flip_mask = dot_products < 0
    normals[flip_mask] = -normals[flip_mask]

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normals = normals / norms
    return normals


def _downsample_voxel(points, colors, voxel_size):
    """Voxel grid downsampling — preserves spatial structure."""
    if np.isscalar(voxel_size):
        voxel_size_arr = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size_arr = np.asarray(voxel_size)

    voxel_indices = np.floor(points / voxel_size_arr).astype(np.int64)
    voxel_centers = (voxel_indices + 0.5) * voxel_size_arr

    voxel_dict = {}
    for i in range(points.shape[0]):
        key = tuple(voxel_indices[i])
        dist_to_center = np.linalg.norm(points[i] - voxel_centers[i])
        if key not in voxel_dict or dist_to_center < voxel_dict[key][1]:
            voxel_dict[key] = (i, dist_to_center)

    selected_indices = np.array([v[0] for v in voxel_dict.values()])
    selected_indices.sort()
    return points[selected_indices], colors[selected_indices]


def _downsample_points(points, colors, max_points=200000):
    """Downsample point cloud using voxel grid if too large."""
    if points.shape[0] <= max_points:
        return points, colors

    bbox_extent = points.max(axis=0) - points.min(axis=0)
    volume = np.prod(np.maximum(bbox_extent, 1e-6))
    target_voxel_size = (volume / max_points) ** (1.0 / 3.0)

    ds_points, ds_colors = _downsample_voxel(points, colors, target_voxel_size)

    if ds_points.shape[0] > max_points * 1.2:
        ratio = (ds_points.shape[0] / max_points) ** (1.0 / 3.0)
        ds_points, ds_colors = _downsample_voxel(points, colors, target_voxel_size * ratio)

    if ds_points.shape[0] > max_points:
        idx = np.random.choice(ds_points.shape[0], max_points, replace=False)
        idx.sort()
        ds_points, ds_colors = ds_points[idx], ds_colors[idx]

    print(f"[GaussianSplatToMesh] Voxel downsampled: {points.shape[0]} -> {ds_points.shape[0]} points")
    return ds_points, ds_colors


def _remove_outliers(points, colors, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers from point cloud."""
    if points.shape[0] < nb_neighbors + 1:
        return points, colors

    tree = cKDTree(points)
    k = min(nb_neighbors + 1, points.shape[0])
    distances, _ = tree.query(points, k=k)
    mean_distances = distances[:, 1:].mean(axis=1)

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_distances < threshold
    removed = points.shape[0] - mask.sum()
    print(f"[GaussianSplatToMesh] Outlier removal: {points.shape[0]} -> {mask.sum()} (removed {removed})")
    return points[mask], colors[mask]


def _remove_outliers_density(points, colors, percentile=5):
    """Remove outliers based on local density."""
    if points.shape[0] < 50:
        return points, colors

    tree = cKDTree(points)
    k = min(10, points.shape[0])
    distances, _ = tree.query(points, k=k)
    local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)

    threshold = np.percentile(local_density, percentile)
    mask = local_density >= threshold

    removed = points.shape[0] - mask.sum()
    print(f"[GaussianSplatToMesh] Density filter: {points.shape[0]} -> {mask.sum()} (removed {removed})")
    return points[mask], colors[mask]


def _transfer_colors_to_vertices(mesh_vertices, source_points, source_colors, k=5):
    """Transfer colors from source point cloud to mesh vertices using KNN interpolation."""
    tree = cKDTree(source_points)
    k = min(k, source_points.shape[0])
    distances, indices = tree.query(mesh_vertices, k=k)

    if k == 1:
        vert_colors = source_colors[indices.ravel()]
    else:
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / distances
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_normalized = weights / weights_sum
        neighbor_colors = source_colors[indices]
        vert_colors = np.sum(neighbor_colors * weights_normalized[:, :, np.newaxis], axis=1)

    vert_colors = np.clip(vert_colors, 0.0, 1.0)
    vert_colors_uint8 = (vert_colors * 255).astype(np.uint8)
    vert_colors_rgba = np.hstack([
        vert_colors_uint8,
        np.full((vert_colors_uint8.shape[0], 1), 255, dtype=np.uint8)
    ])
    return vert_colors_rgba


def _alpha_shape_mesh(points, colors, alpha=2.0):
    """Create mesh using Delaunay triangulation with alpha shape filtering."""
    print(f"[GaussianSplatToMesh] Running alpha shape (alpha={alpha})...")

    tri = Delaunay(points)
    tetrahedra = tri.simplices

    face_count = {}
    for tet in tetrahedra:
        for face in [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]:
            face_count[face] = face_count.get(face, 0) + 1

    surface_faces = [f for f, c in face_count.items() if c == 1]

    if alpha > 0:
        filtered_faces = []
        for face in surface_faces:
            pts = points[list(face)]
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
    vertex_colors_rgba = _transfer_colors_to_vertices(points, points, colors, k=1)

    mesh = Trimesh.Trimesh(
        vertices=points,
        faces=faces_array,
        vertex_colors=vertex_colors_rgba,
        process=False,
    )
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    print(f"[GaussianSplatToMesh] Alpha shape: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def _marching_cubes_mesh(points, colors, resolution=256, padding=0.1, sigma=1.5):
    """Create mesh using marching cubes on a KDE-based density field."""
    print(f"[GaussianSplatToMesh] Running marching cubes (res={resolution}, sigma={sigma})...")

    from scipy.ndimage import gaussian_filter

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = pmax - pmin
    pad = extent * padding
    pmin -= pad
    pmax += pad
    extent = pmax - pmin

    voxel_size = extent / resolution
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    indices = ((points - pmin) / voxel_size).astype(np.int32)
    indices = np.clip(indices, 0, resolution - 1)

    flat_indices = (indices[:, 0] * resolution * resolution +
                    indices[:, 1] * resolution + indices[:, 2])
    np.add.at(grid.ravel(), flat_indices, 1.0)

    # Splat to neighbors for smoother density
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                shifted = indices + np.array([dx, dy, dz])
                shifted = np.clip(shifted, 0, resolution - 1)
                flat_s = (shifted[:, 0] * resolution * resolution +
                          shifted[:, 1] * resolution + shifted[:, 2])
                w = 0.5 ** (abs(dx) + abs(dy) + abs(dz))
                np.add.at(grid.ravel(), flat_s, w)

    grid = gaussian_filter(grid, sigma=sigma)

    nonzero = grid[grid > 0]
    if len(nonzero) == 0:
        raise ValueError("No points in voxel grid")

    iso_level = np.percentile(nonzero, 30)
    print(f"[GaussianSplatToMesh] Density: min={grid.min():.3f}, max={grid.max():.3f}, iso={iso_level:.3f}")

    try:
        from skimage.measure import marching_cubes
        verts, faces, normals_mc, _ = marching_cubes(grid, level=iso_level)
    except ImportError:
        raise ImportError("scikit-image required for marching_cubes. pip install scikit-image")

    verts = verts * voxel_size + pmin
    vertex_colors_rgba = _transfer_colors_to_vertices(verts, points, colors, k=5)

    mesh = Trimesh.Trimesh(
        vertices=verts, faces=faces,
        vertex_colors=vertex_colors_rgba, process=False,
    )
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    print(f"[GaussianSplatToMesh] Marching cubes: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def _poisson_like_mesh(points, colors, depth=8, scale=1.1):
    """Screened-Poisson-like surface reconstruction without Open3D."""
    resolution = min(2 ** depth, 256)
    print(f"[GaussianSplatToMesh] Running Poisson-like (depth={depth}, res={resolution})...")

    from scipy.ndimage import gaussian_filter

    print("[GaussianSplatToMesh] Estimating normals...")
    normals = _estimate_normals_fast(points, k=min(20, points.shape[0]))

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = (pmin + pmax) / 2
    extent = (pmax - pmin) * scale
    pmin = center - extent / 2
    pmax = center + extent / 2
    voxel_size = extent / resolution

    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    weight_grid = np.zeros_like(grid)

    voxel_coords = ((points - pmin) / voxel_size).astype(np.int32)
    voxel_coords = np.clip(voxel_coords, 0, resolution - 1)

    splat_range = 2
    bandwidth = np.mean(voxel_size) * 2.0
    for i in range(points.shape[0]):
        ix, iy, iz = voxel_coords[i]
        for dx in range(-splat_range, splat_range + 1):
            for dy in range(-splat_range, splat_range + 1):
                for dz in range(-splat_range, splat_range + 1):
                    nx, ny, nz = ix + dx, iy + dy, iz + dz
                    if 0 <= nx < resolution and 0 <= ny < resolution and 0 <= nz < resolution:
                        vc = pmin + (np.array([nx, ny, nz]) + 0.5) * voxel_size
                        diff = vc - points[i]
                        signed_dist = np.dot(diff, normals[i])
                        dist_sq = np.sum(diff ** 2)
                        dist_weight = np.exp(-dist_sq / (2 * bandwidth ** 2))
                        grid[nx, ny, nz] += signed_dist * dist_weight
                        weight_grid[nx, ny, nz] += dist_weight

    valid_mask = weight_grid > 0
    grid[valid_mask] /= weight_grid[valid_mask]
    grid = gaussian_filter(grid, sigma=1.0)

    try:
        from skimage.measure import marching_cubes
        verts, faces, normals_mc, _ = marching_cubes(grid, level=0.0)
    except ImportError:
        raise ImportError("scikit-image required for poisson_like. pip install scikit-image")

    verts = verts * voxel_size + pmin
    vertex_colors_rgba = _transfer_colors_to_vertices(verts, points, colors, k=5)

    mesh = Trimesh.Trimesh(
        vertices=verts, faces=faces,
        vertex_colors=vertex_colors_rgba, process=False,
    )
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    print(f"[GaussianSplatToMesh] Poisson-like: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def _ball_pivoting_mesh(points, colors, radius_factor=1.5):
    """Approximate ball-pivoting using Delaunay with multi-scale edge filtering."""
    print(f"[GaussianSplatToMesh] Running ball pivoting (radius_factor={radius_factor})...")

    tree = cKDTree(points)
    k = min(6, points.shape[0])
    distances, _ = tree.query(points, k=k)
    avg_nn_dist = distances[:, 1:].mean()

    radii = [
        avg_nn_dist * radius_factor * 2,
        avg_nn_dist * radius_factor * 3,
        avg_nn_dist * radius_factor * 5,
    ]

    tri = Delaunay(points)
    tetrahedra = tri.simplices

    face_count = {}
    for tet in tetrahedra:
        for face in [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]:
            face_count[face] = face_count.get(face, 0) + 1

    # Multi-scale: try each radius and collect faces
    all_faces = set()
    for max_edge_length in radii:
        for face, count in face_count.items():
            if count == 1 and face not in all_faces:
                pts = points[list(face)]
                edges = [
                    np.linalg.norm(pts[1] - pts[0]),
                    np.linalg.norm(pts[2] - pts[1]),
                    np.linalg.norm(pts[0] - pts[2]),
                ]
                if max(edges) < max_edge_length:
                    all_faces.add(face)

    if len(all_faces) == 0:
        raise ValueError("Ball pivoting produced no faces. Try adjusting radius_factor.")

    faces_array = np.array(list(all_faces), dtype=np.int64)
    vertex_colors_rgba = _transfer_colors_to_vertices(points, points, colors, k=1)

    mesh = Trimesh.Trimesh(
        vertices=points, faces=faces_array,
        vertex_colors=vertex_colors_rgba, process=False,
    )
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    print(f"[GaussianSplatToMesh] Ball pivoting: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Node: GaussianSplatToMesh
# ─────────────────────────────────────────────────────────────────────────────
class GaussianSplatToMesh:
    """
    Convert PLY_DATA (Gaussian Splat point cloud from HY-World 2.0) to TRIMESH.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
            },
            "optional": {
                "method": (["marching_cubes", "poisson_like", "alpha_shape", "ball_pivoting"], {
                    "default": "marching_cubes",
                    "tooltip": (
                        "Surface reconstruction method:\n"
                        "- marching_cubes: Volumetric reconstruction (recommended, smooth)\n"
                        "- poisson_like: Normal-based surface reconstruction (best quality, slower)\n"
                        "- alpha_shape: Delaunay-based alpha shape (fast)\n"
                        "- ball_pivoting: Approximate ball pivoting (good edge filtering)"
                    ),
                }),
                "alpha": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Alpha for alpha_shape (0=convex hull, larger=more detail). Typical: 1.0-5.0",
                }),
                "resolution": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 512,
                    "step": 16,
                    "tooltip": "Voxel grid resolution for marching_cubes/poisson_like",
                }),
                "radius_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Radius multiplier for ball_pivoting method",
                }),
                "max_points": ("INT", {
                    "default": 200000,
                    "min": 10000,
                    "max": 1000000,
                    "step": 10000,
                    "tooltip": "Maximum number of points (voxel-downsampled if exceeded)",
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
                    "tooltip": "Std dev ratio for outlier removal (lower = more aggressive)",
                }),
                "density_filter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove sparse/isolated points based on local density",
                }),
                "density_percentile": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 1.0,
                    "tooltip": "Percentile threshold for density filter (higher = more aggressive)",
                }),
                "color_knn": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of nearest neighbors for vertex color interpolation",
                }),
                "smooth_sigma": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Gaussian smoothing sigma for marching_cubes density field",
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "3D/mesh"
    DESCRIPTION = (
        "Convert PLY_DATA (Gaussian Splat point cloud from HY-World 2.0) to TRIMESH mesh. "
        "The output can be connected to Hy3DExportMesh for GLB/OBJ/PLY/STL export. "
        "v2.1: Fixed color extraction for pts3d_filtered, added normal computation."
    )

    def convert(
        self,
        ply_data,
        method="marching_cubes",
        alpha=2.0,
        resolution=256,
        radius_factor=2.0,
        max_points=200000,
        remove_outliers=True,
        outlier_std_ratio=2.0,
        density_filter=True,
        density_percentile=5.0,
        color_knn=5,
        smooth_sigma=1.5,
    ):
        print(f"[GaussianSplatToMesh] Starting conversion (method={method})...")

        # ── 1. Extract points and colors ──────────────────────────────────
        points, colors = _extract_points_and_colors(ply_data)

        if points.shape[0] < 4:
            raise ValueError(
                f"Not enough points for mesh reconstruction: {points.shape[0]} points. "
                "Need at least 4 points."
            )

        # ── 2. Remove outliers (statistical) ──────────────────────────────
        if remove_outliers and points.shape[0] > 50:
            points, colors = _remove_outliers(
                points, colors,
                nb_neighbors=min(20, points.shape[0] - 1),
                std_ratio=outlier_std_ratio,
            )

        # ── 3. Remove outliers (density-based) ───────────────────────────
        if density_filter and points.shape[0] > 100:
            points, colors = _remove_outliers_density(
                points, colors, percentile=density_percentile,
            )

        # ── 4. Downsample if needed (voxel grid) ─────────────────────────
        if points.shape[0] > max_points:
            print(f"[GaussianSplatToMesh] Downsampling: {points.shape[0]} -> {max_points}")
            points, colors = _downsample_points(points, colors, max_points=max_points)

        print(f"[GaussianSplatToMesh] Final point cloud: {points.shape[0]} points")

        # ── 5. Surface reconstruction ─────────────────────────────────────
        if method == "marching_cubes":
            mesh = _marching_cubes_mesh(
                points, colors, resolution=resolution, sigma=smooth_sigma,
            )
        elif method == "poisson_like":
            # Compute depth from resolution
            depth = max(6, min(8, int(np.log2(resolution))))
            mesh = _poisson_like_mesh(points, colors, depth=depth)
        elif method == "alpha_shape":
            mesh = _alpha_shape_mesh(points, colors, alpha=alpha)
        elif method == "ball_pivoting":
            mesh = _ball_pivoting_mesh(points, colors, radius_factor=radius_factor)
        else:
            raise ValueError(f"Unknown method: {method}")

        # ── 6. Fix normals ────────────────────────────────────────────────
        try:
            if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
                mesh.fix_normals()
                print("[GaussianSplatToMesh] Computed mesh normals via fix_normals()")
            else:
                # Ensure normals are consistent
                mesh.fix_normals()
        except Exception as e:
            print(f"[GaussianSplatToMesh] WARNING: Could not fix normals: {e}")

        # ── 7. Verify vertex colors are present ──────────────────────────
        if mesh.visual is None or not hasattr(mesh.visual, 'vertex_colors'):
            print("[GaussianSplatToMesh] WARNING: Re-applying vertex colors...")
            vertex_colors_rgba = _transfer_colors_to_vertices(
                np.array(mesh.vertices), points, colors, k=color_knn,
            )
            mesh.visual = Trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=vertex_colors_rgba,
            )
        else:
            vc = mesh.visual.vertex_colors
            if vc is None or len(vc) == 0:
                print("[GaussianSplatToMesh] WARNING: Empty vertex colors, re-applying...")
                vertex_colors_rgba = _transfer_colors_to_vertices(
                    np.array(mesh.vertices), points, colors, k=color_knn,
                )
                mesh.visual = Trimesh.visual.ColorVisuals(
                    mesh=mesh, vertex_colors=vertex_colors_rgba,
                )

        print(
            f"[GaussianSplatToMesh] Done: {len(mesh.vertices)} vertices, "
            f"{len(mesh.faces)} faces, has_colors={mesh.visual.vertex_colors is not None}"
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