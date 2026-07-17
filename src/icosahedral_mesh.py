"""
================================================================================
Icosahedral Mesh Construction for GraphCast-style GNN
================================================================================

Builds a multi-resolution icosahedral mesh and the bipartite graphs needed
for the Grid ↔ Mesh encoder/decoder.

Three graph structures:
  1. grid2mesh  - bipartite edges from lat/lon grid nodes to nearby mesh nodes
  2. mesh2mesh  - multi-scale edges within the icosahedral mesh hierarchy
  3. mesh2grid  - bipartite edges from mesh nodes back to lat/lon grid nodes

References:
  - Lam et al. (2023) "GraphCast: Learning skillful medium-range global
    weather forecasting" (Science)
  - Keisler (2022) "Forecasting Global Weather with Graph Neural Networks"

Usage:
  mesh = IcosahedralMesh(
      refinement_level=5,        # ~0.5° resolution (~40k nodes globally)
      lat_range=(25.0, 50.0),    # CONUS bounding box
      lon_range=(-130.0, -60.0),
      grid_lat=np.linspace(25, 50, 621),
      grid_lon=np.linspace(-130, -60, 1405),
      land_mask=mask_2d,         # (621, 1405) binary
  )
  # mesh.grid2mesh_edge_index, mesh.mesh_edge_index, mesh.mesh2grid_edge_index
================================================================================
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from scipy.spatial import cKDTree
from collections import defaultdict


# =============================================================================
# ICOSAHEDRON BASE GEOMETRY
# =============================================================================

def icosahedron_vertices():
    """Return the 12 vertices of a unit icosahedron."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)

    # Project onto unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def icosahedron_faces():
    """Return the 20 triangular faces of an icosahedron (vertex index triples)."""
    return np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)


# =============================================================================
# RECURSIVE SUBDIVISION
# =============================================================================

def subdivide_mesh(vertices, faces, levels):
    """
    Recursively subdivide icosahedral faces and project onto the unit sphere.

    Returns:
        all_vertices: list of vertex arrays at each level [level_0, ..., level_n]
        all_faces: list of face arrays at each level
        all_edges: list of edge arrays at each level (undirected)
        hierarchy_edges: edges connecting level i to level i+1 (parent-child)
    """
    all_vertices = [vertices.copy()]
    all_faces = [faces.copy()]
    all_edges = [_faces_to_edges(faces)]
    hierarchy_edges = []

    current_verts = vertices.copy()
    current_faces = faces.copy()

    for level in range(levels):
        new_verts_list = list(current_verts)
        midpoint_cache = {}
        new_faces = []

        n_existing = len(new_verts_list)

        for tri in current_faces:
            mids = []
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                if edge not in midpoint_cache:
                    mid = (new_verts_list[edge[0]] + new_verts_list[edge[1]]) / 2.0
                    mid = mid / np.linalg.norm(mid)  # project to sphere
                    midpoint_cache[edge] = len(new_verts_list)
                    new_verts_list.append(mid)
                mids.append(midpoint_cache[edge])

            a, b, c = tri
            m_ab, m_bc, m_ca = mids
            new_faces.extend([
                [a, m_ab, m_ca],
                [b, m_bc, m_ab],
                [c, m_ca, m_bc],
                [m_ab, m_bc, m_ca],
            ])

        current_verts = np.array(new_verts_list)
        current_faces = np.array(new_faces, dtype=np.int64)

        all_vertices.append(current_verts.copy())
        all_faces.append(current_faces.copy())
        all_edges.append(_faces_to_edges(current_faces))

        # Hierarchy: connect old vertices to new midpoints at this level
        h_edges = []
        for (v1, v2), mid_idx in midpoint_cache.items():
            h_edges.append([v1, mid_idx])
            h_edges.append([v2, mid_idx])
        hierarchy_edges.append(np.array(h_edges, dtype=np.int64) if h_edges else np.zeros((0, 2), dtype=np.int64))

    return all_vertices, all_faces, all_edges, hierarchy_edges


def _faces_to_edges(faces):
    """Extract unique undirected edges from a face array."""
    edge_set = set()
    for tri in faces:
        for i in range(3):
            e = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            edge_set.add(e)
    edges = np.array(sorted(edge_set), dtype=np.int64)
    return edges


# =============================================================================
# COORDINATE CONVERSIONS
# =============================================================================

def xyz_to_latlon(xyz):
    """Convert unit-sphere XYZ to (lat, lon) in degrees."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def latlon_to_xyz(lat_deg, lon_deg):
    """Convert (lat, lon) in degrees to unit-sphere XYZ."""
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.stack([x, y, z], axis=-1)


def shortest_lon_delta(lon_to, lon_from):
    """Shortest signed longitude delta in degrees, wrapped to [-180, 180)."""
    return (lon_to - lon_from + 180.0) % 360.0 - 180.0


# =============================================================================
# REGIONAL MESH EXTRACTION
# =============================================================================

def extract_regional_mesh(vertices, edges, lat_range, lon_range, buffer_deg=5.0):
    """
    Keep only mesh nodes within the lat/lon bounding box (plus buffer).
    Remap edge indices to the new compressed node numbering.

    Args:
        vertices: (N, 3) XYZ on unit sphere
        edges: (E, 2) undirected edge list
        lat_range: (lat_min, lat_max) in degrees
        lon_range: (lon_min, lon_max) in degrees
        buffer_deg: extra degrees around the bounding box to keep

    Returns:
        regional_verts: (M, 3) subset of vertices
        regional_edges: (E', 2) remapped edges
        regional_lat: (M,) latitudes
        regional_lon: (M,) longitudes
        keep_mask: (N,) boolean mask of kept nodes
        old_to_new: dict mapping old index -> new index
    """
    lat, lon = xyz_to_latlon(vertices)

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    keep = (
        (lat >= lat_min - buffer_deg) & (lat <= lat_max + buffer_deg) &
        (lon >= lon_min - buffer_deg) & (lon <= lon_max + buffer_deg)
    )

    keep_indices = np.where(keep)[0]
    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    regional_verts = vertices[keep]
    regional_lat = lat[keep]
    regional_lon = lon[keep]

    # Filter edges: both endpoints must be in the region
    valid_edges = []
    for e in edges:
        if e[0] in old_to_new and e[1] in old_to_new:
            valid_edges.append([old_to_new[e[0]], old_to_new[e[1]]])

    regional_edges = np.array(valid_edges, dtype=np.int64) if valid_edges else np.zeros((0, 2), dtype=np.int64)

    return regional_verts, regional_edges, regional_lat, regional_lon, keep, old_to_new


# =============================================================================
# BIPARTITE GRAPH CONSTRUCTION
# =============================================================================

def build_grid2mesh_graph(grid_lat, grid_lon, mesh_lat, mesh_lon,
                          land_mask=None, k_neighbors=3):
    """
    Build bipartite edges from grid nodes to their k nearest mesh nodes.

    For each LAND grid point, find the k closest mesh nodes and create edges.

    Args:
        grid_lat: (H,) or (H, W) grid latitudes in degrees
        grid_lon: (W,) or (H, W) grid longitudes in degrees
        mesh_lat: (M,) mesh node latitudes
        mesh_lon: (M,) mesh node longitudes
        land_mask: (H, W) binary mask, or None to include all points
        k_neighbors: number of mesh neighbors per grid point

    Returns:
        edge_index: (2, num_edges) [grid_node_idx, mesh_node_idx]
        edge_attr: (num_edges, 3) relative position features [dlat, dlon, dist]
        grid_node_indices: (num_grid_nodes,) flat indices of included grid points
    """
    # Build grid coordinate arrays
    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
    else:
        lat_grid, lon_grid = grid_lat, grid_lon

    H, W = lat_grid.shape

    # Flatten and filter by land mask
    flat_lat = lat_grid.ravel()
    flat_lon = lon_grid.ravel()

    if land_mask is not None:
        land_flat = land_mask.ravel().astype(bool)
        grid_indices = np.where(land_flat)[0]
    else:
        grid_indices = np.arange(len(flat_lat))

    grid_xyz = latlon_to_xyz(flat_lat[grid_indices], flat_lon[grid_indices])
    mesh_xyz = latlon_to_xyz(mesh_lat, mesh_lon)

    # KD-tree query
    tree = cKDTree(mesh_xyz)
    dists, mesh_neighbors = tree.query(grid_xyz, k=k_neighbors)

    # Build edge list
    num_grid = len(grid_indices)
    src = np.repeat(np.arange(num_grid), k_neighbors)  # grid node indices (local)
    dst = mesh_neighbors.ravel()  # mesh node indices

    # Edge features: relative position
    dlat = mesh_lat[dst] - flat_lat[grid_indices[src]]
    dlon = shortest_lon_delta(mesh_lon[dst], flat_lon[grid_indices[src]])
    dist = dists.ravel()

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr = np.stack([dlat, dlon, dist], axis=-1).astype(np.float32)

    return edge_index, edge_attr, grid_indices


def build_mesh2grid_graph(grid_lat, grid_lon, mesh_lat, mesh_lon,
                          land_mask=None, k_neighbors=3):
    """
    Build bipartite edges from mesh nodes to grid nodes.

    For each LAND grid point, find k nearest mesh nodes. Edges go FROM mesh TO grid.

    Returns:
        edge_index: (2, num_edges) [mesh_node_idx, grid_node_idx]
        edge_attr: (num_edges, 3) relative position features
        grid_node_indices: same as grid2mesh
    """
    # Reuse grid2mesh but flip edge direction
    g2m_edges, g2m_attr, grid_indices = build_grid2mesh_graph(
        grid_lat, grid_lon, mesh_lat, mesh_lon, land_mask, k_neighbors
    )

    # Flip: (grid, mesh) -> (mesh, grid)
    edge_index = np.stack([g2m_edges[1], g2m_edges[0]], axis=0)
    # Negate relative positions since direction is reversed
    edge_attr = g2m_attr.copy()
    edge_attr[:, :2] *= -1

    return edge_index, edge_attr, grid_indices


def build_mesh2mesh_edges(edges_array):
    """
    Convert undirected edge array to directed edge_index for message passing.

    Args:
        edges_array: (E, 2) undirected edges

    Returns:
        edge_index: (2, 2E) directed edges (both directions)
    """
    if len(edges_array) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    fwd = edges_array.T  # (2, E)
    bwd = edges_array[:, ::-1].T  # (2, E) reversed
    edge_index = np.concatenate([fwd, bwd], axis=1)
    return edge_index.astype(np.int64)


# =============================================================================
# MESH EDGE FEATURES
# =============================================================================

def compute_mesh_edge_features(vertices, edge_index):
    """
    Compute edge features for mesh-to-mesh edges.

    Features per edge:
      - relative XYZ displacement (3)
      - Euclidean distance on sphere (1)

    Args:
        vertices: (N, 3) XYZ on unit sphere
        edge_index: (2, E) directed edges

    Returns:
        edge_attr: (E, 4)
    """
    src = edge_index[0]
    dst = edge_index[1]

    displacement = vertices[dst] - vertices[src]
    distance = np.linalg.norm(displacement, axis=1, keepdims=True)

    return np.concatenate([displacement, distance], axis=-1).astype(np.float32)


def compute_mesh_node_features(vertices):
    """
    Node positional features for mesh nodes.

    Features: [x, y, z, lat_normalized, lon_normalized, sin_lat, cos_lat, sin_lon, cos_lon]

    Args:
        vertices: (N, 3) XYZ on unit sphere

    Returns:
        node_features: (N, 9)
    """
    lat, lon = xyz_to_latlon(vertices)

    lat_norm = lat / 90.0  # [-1, 1]
    lon_norm = lon / 180.0  # [-1, 1]

    sin_lat = np.sin(np.radians(lat))
    cos_lat = np.cos(np.radians(lat))
    sin_lon = np.sin(np.radians(lon))
    cos_lon = np.cos(np.radians(lon))

    return np.stack([
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        lat_norm, lon_norm, sin_lat, cos_lat, sin_lon, cos_lon,
    ], axis=-1).astype(np.float32)


# =============================================================================
# MAIN MESH CLASS
# =============================================================================

class IcosahedralMesh:
    """
    Complete icosahedral mesh for GraphCast-style GNN over a regional domain.

    Builds all three graph structures:
      1. grid2mesh (encoder)
      2. mesh2mesh (processor, multi-resolution)
      3. mesh2grid (decoder)

    Args:
        refinement_level: int, number of icosahedral subdivisions (5 -> ~10k, 6 -> ~40k nodes)
        lat_range: (min, max) latitude in degrees
        lon_range: (min, max) longitude in degrees
        grid_lat: 1D array of grid latitudes
        grid_lon: 1D array of grid longitudes
        land_mask: (H, W) binary mask, None = use all grid points
        buffer_deg: extra degrees around bounding box for mesh nodes
        k_grid2mesh: neighbors per grid point in encoder graph
        k_mesh2grid: neighbors per grid point in decoder graph
        multimesh_levels: which refinement levels to include in processor
                          (default: all levels from 0 to refinement_level)
    """
    def __init__(
        self,
        refinement_level=5,
        lat_range=(25.0, 50.0),
        lon_range=(-130.0, -60.0),
        grid_lat=None,
        grid_lon=None,
        land_mask=None,
        buffer_deg=5.0,
        k_grid2mesh=3,
        k_mesh2grid=3,
        multimesh_levels=None,
    ):
        self.refinement_level = refinement_level
        self.lat_range = lat_range
        self.lon_range = lon_range

        print(f"\n{'='*70}")
        print(f"Building Icosahedral Mesh (refinement level {refinement_level})")
        print(f"{'='*70}")

        # Step 1: Build full icosahedral mesh hierarchy
        base_verts = icosahedron_vertices()
        base_faces = icosahedron_faces()
        all_verts, all_faces, all_edges, hierarchy_edges = subdivide_mesh(
            base_verts, base_faces, refinement_level
        )

        for lvl, v in enumerate(all_verts):
            print(f"  Level {lvl}: {len(v)} vertices, {len(all_edges[lvl])} edges")

        # Step 2: Extract regional mesh at finest level
        finest_verts = all_verts[-1]
        finest_edges = all_edges[-1]

        (self.mesh_vertices, self.mesh_edges,
         self.mesh_lat, self.mesh_lon,
         self.keep_mask, self.old_to_new) = extract_regional_mesh(
            finest_verts, finest_edges, lat_range, lon_range, buffer_deg
        )

        self.num_mesh_nodes = len(self.mesh_vertices)
        self.num_mesh_edges_undirected = len(self.mesh_edges)
        print(f"\n  Regional mesh: {self.num_mesh_nodes} nodes, "
              f"{self.num_mesh_edges_undirected} undirected edges")

        # Step 3: Multi-resolution mesh edges for the processor
        # Include edges from coarser levels that connect nodes in the region
        if multimesh_levels is None:
            # Use levels from max(0, refinement_level-3) to refinement_level
            # Coarser levels have longer-range connections
            multimesh_levels = list(range(max(0, refinement_level - 3), refinement_level + 1))

        print(f"  Multi-mesh levels: {multimesh_levels}")

        multimesh_edge_sets = set()
        for lvl in multimesh_levels:
            lvl_verts = all_verts[lvl]
            lvl_edges = all_edges[lvl]

            # These are indices into the level-specific vertex array.
            # At the finest level, all vertices from coarser levels are a subset
            # (they share the same indices 0..N_coarse-1 in the finest array).
            # So coarse-level edges directly index into the finest vertex array.
            for e in lvl_edges:
                v0, v1 = int(e[0]), int(e[1])
                # Check both endpoints are in our region
                if v0 in self.old_to_new and v1 in self.old_to_new:
                    new_v0 = self.old_to_new[v0]
                    new_v1 = self.old_to_new[v1]
                    multimesh_edge_sets.add((min(new_v0, new_v1), max(new_v0, new_v1)))

        self.multimesh_edges = np.array(sorted(multimesh_edge_sets), dtype=np.int64) \
            if multimesh_edge_sets else np.zeros((0, 2), dtype=np.int64)
        print(f"  Multi-mesh total: {len(self.multimesh_edges)} undirected edges")

        # Step 4: Mesh node positional features
        self.mesh_node_features = compute_mesh_node_features(self.mesh_vertices)

        # Step 5: Mesh-to-mesh directed edges + features
        self.mesh_edge_index = build_mesh2mesh_edges(self.multimesh_edges)
        self.mesh_edge_attr = compute_mesh_edge_features(
            self.mesh_vertices, self.mesh_edge_index
        )

        # Step 6: Grid-to-mesh bipartite graph
        if grid_lat is not None and grid_lon is not None:
            mask_np = None
            if land_mask is not None:
                if torch is not None and isinstance(land_mask, torch.Tensor):
                    mask_np = land_mask.cpu().numpy()
                else:
                    mask_np = np.array(land_mask)

            self.g2m_edge_index, self.g2m_edge_attr, self.grid_node_indices = \
                build_grid2mesh_graph(
                    grid_lat, grid_lon, self.mesh_lat, self.mesh_lon,
                    mask_np, k_grid2mesh
                )

            self.m2g_edge_index, self.m2g_edge_attr, _ = \
                build_mesh2grid_graph(
                    grid_lat, grid_lon, self.mesh_lat, self.mesh_lon,
                    mask_np, k_mesh2grid
                )

            self.num_grid_nodes = len(self.grid_node_indices)
            print(f"\n  Grid nodes (land): {self.num_grid_nodes}")
            print(f"  Grid2Mesh edges: {self.g2m_edge_index.shape[1]}")
            print(f"  Mesh2Grid edges: {self.m2g_edge_index.shape[1]}")
        else:
            self.g2m_edge_index = None
            self.m2g_edge_index = None
            self.num_grid_nodes = 0

        print(f"{'='*70}\n")

    def to_torch(self, device='cpu'):
        """Convert all arrays to torch tensors on the given device."""
        assert torch is not None, "PyTorch required for to_torch()"
        self.mesh_node_features_t = torch.from_numpy(self.mesh_node_features).to(device)
        self.mesh_edge_index_t = torch.from_numpy(self.mesh_edge_index).long().to(device)
        self.mesh_edge_attr_t = torch.from_numpy(self.mesh_edge_attr).to(device)

        if self.g2m_edge_index is not None:
            self.g2m_edge_index_t = torch.from_numpy(self.g2m_edge_index).long().to(device)
            self.g2m_edge_attr_t = torch.from_numpy(self.g2m_edge_attr).to(device)
            self.m2g_edge_index_t = torch.from_numpy(self.m2g_edge_index).long().to(device)
            self.m2g_edge_attr_t = torch.from_numpy(self.m2g_edge_attr).to(device)

        return self

    def summary(self):
        return {
            'refinement_level': self.refinement_level,
            'num_mesh_nodes': self.num_mesh_nodes,
            'num_mesh_edges': self.mesh_edge_index.shape[1],
            'num_multimesh_edges_undirected': len(self.multimesh_edges),
            'num_grid_nodes': self.num_grid_nodes,
            'num_g2m_edges': self.g2m_edge_index.shape[1] if self.g2m_edge_index is not None else 0,
            'num_m2g_edges': self.m2g_edge_index.shape[1] if self.m2g_edge_index is not None else 0,
        }


# =============================================================================
# UTILITY: Scatter grid data onto mesh or reverse
# =============================================================================

def grid_to_flat(grid_data, grid_node_indices, image_size):
    """
    Extract land-only values from a (B, C, H, W) grid tensor.

    Args:
        grid_data: (B, C, H, W) tensor
        grid_node_indices: (N_land,) flat indices into H*W
        image_size: (H, W)

    Returns:
        flat_data: (B, N_land, C)
    """
    B, C, H, W = grid_data.shape
    flat = grid_data.reshape(B, C, H * W)  # (B, C, H*W)
    idx = torch.from_numpy(grid_node_indices).long().to(grid_data.device)
    selected = flat[:, :, idx]  # (B, C, N_land)
    return selected.permute(0, 2, 1)  # (B, N_land, C)


def flat_to_grid(flat_data, grid_node_indices, image_size, fill_value=0.0):
    """
    Scatter land-only values back to a (B, C, H, W) grid tensor.

    Args:
        flat_data: (B, N_land, C)
        grid_node_indices: (N_land,) flat indices into H*W
        image_size: (H, W)
        fill_value: value for ocean pixels

    Returns:
        grid_data: (B, C, H, W)
    """
    B, N, C = flat_data.shape
    H, W = image_size
    device = flat_data.device

    grid = torch.full((B, C, H * W), fill_value, dtype=flat_data.dtype, device=device)
    idx = torch.from_numpy(grid_node_indices).long().to(device)
    flat_t = flat_data.permute(0, 2, 1)  # (B, C, N)
    grid[:, :, idx] = flat_t
    return grid.reshape(B, C, H, W)


if __name__ == "__main__":
    # Quick test
    grid_lat = np.linspace(25.0, 50.0, 621)
    grid_lon = np.linspace(-130.0, -60.0, 1405)

    # Dummy land mask (all land for testing)
    mask = np.ones((621, 1405), dtype=np.float32)

    mesh = IcosahedralMesh(
        refinement_level=5,
        lat_range=(25.0, 50.0),
        lon_range=(-130.0, -60.0),
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        land_mask=mask,
        buffer_deg=5.0,
        k_grid2mesh=3,
        k_mesh2grid=3,
    )

    print("\nMesh summary:")
    for k, v in mesh.summary().items():
        print(f"  {k}: {v}")
