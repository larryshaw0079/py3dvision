from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import einops
import trimesh
import networkx as nx
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors


def read_shape(file, as_cloud=False):
    """
    Read mesh from file.

    Args:
        file (str): file name
        as_cloud (bool, optional): read shape as point cloud. Default False
    Returns:
        verts (np.ndarray): vertices [V, 3]
        faces (np.ndarray): faces [F, 3] or None
    """
    if as_cloud:
        verts = np.asarray(o3d.io.read_point_cloud(file).points)
        faces = None
    else:
        mesh = o3d.io.read_triangle_mesh(file)
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    return verts, faces


def write_off(file, verts, faces):
    with open(file, 'w') as f:
        f.write("OFF\n")
        f.write(f"{verts.shape[0]} {faces.shape[0]} {0}\n")
        for x in verts:
            f.write(f"{' '.join(map(str, x))}\n")
        for x in faces:
            f.write(f"{len(x)} {' '.join(map(str, x))}\n")


def compute_geodesic_distmat(verts, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm

    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]

    Returns:
        geo_dist: geodesic distance matrix [n, n]
    """
    NN = 500

    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph
    assert nx.is_connected(vertex_adjacency), 'Graph not connected'
    vertex_adjacency_matrix = nx.adjacency_matrix(vertex_adjacency, range(verts.shape[0]))
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(verts, n_neighbors=NN, mode='distance', include_self=False)
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[vertex_adjacency_matrix != 0]
    # compute geodesic matrix
    geodesic_x = shortest_path(distance_adj, directed=False)
    if np.any(np.isinf(geodesic_x)):
        print('Inf number in geodesic distance. Increase NN.')
    return geodesic_x


def compute_udf_from_mesh(
        mesh_o3d: o3d.geometry.TriangleMesh,
        num_surface_points: int = 100_000,
        num_queries_on_surface: int = 10_000,
        queries_stds: List[float] = [0.003, 0.01, 0.1],
        num_queries_per_std: List[int] = [5_000, 4_000, 500, 500],
        coords_range: Tuple[float, float] = (-1.0, 1.0),
        max_dist: float = 0.1,
        convert_to_bce_labels: bool = False,
        use_cuda: bool = True,
        input_queries=None
) -> Tuple[Tensor, Tensor, Tensor]:
    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=num_surface_points)
    pcd = torch.tensor(np.asarray(pcd_o3d.points), dtype=torch.float)

    device = "cuda" if use_cuda else "cpu"
    if input_queries is not None:
        queries = input_queries
    else:
        queries = sample_points_around_pcd(
            pcd,
            queries_stds,
            num_queries_per_std,
            coords_range,
            device,
        )
    queries = queries.cpu()

    udf, gradients = compute_udf_and_gradients(mesh_o3d, queries)
    values = torch.clip(udf, min=0, max=max_dist)

    # q_on_surf_o3d = mesh_o3d.sample_points_uniformly(
    #     number_of_points=num_queries_on_surface
    # )
    # queries_on_surface = get_tensor_pcd_from_o3d(q_on_surf_o3d)[:, :3]
    # values_on_surface = torch.zeros(num_queries_on_surface)
    # gradients_on_surface = torch.zeros(num_queries_on_surface, 3)

    # queries = torch.cat([queries_on_surface, queries], dim=0)
    # values = torch.cat([values_on_surface, values], dim=0)
    # gradients = torch.cat([gradients_on_surface, gradients], dim=0)

    # if convert_to_bce_labels:
    #     values /= max_dist
    #     values = 1 - values

    return queries, values, gradients


def compute_sdf_from_mesh(
        mesh_o3d: o3d.geometry.TriangleMesh,
        num_surface_points: int = 100_000,
        num_queries_on_surface: int = 10_000,
        queries_stds: List[float] = [0.003, 0.01, 0.1],
        num_queries_per_std: List[int] = [5_000, 4_000, 500, 500],
        coords_range: Tuple[float, float] = (-1.0, 1.0),
        max_dist: float = 0.1,
        convert_to_bce_labels: bool = False,
        use_cuda: bool = True,
        input_queries=None
) -> Tuple[Tensor, Tensor, Tensor]:
    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=num_surface_points)
    pcd = torch.tensor(np.asarray(pcd_o3d.points), dtype=torch.float)

    device = "cuda" if use_cuda else "cpu"
    if input_queries is not None:
        queries = input_queries
    else:
        queries = sample_points_around_pcd(
            pcd,
            queries_stds,
            num_queries_per_std,
            coords_range,
            device,
        )
    queries = queries.cpu()

    sdf, gradients = compute_sdf_and_gradients(mesh_o3d, queries)
    values = torch.clip(sdf, min=-max_dist, max=max_dist)

    q_on_surf_o3d = mesh_o3d.sample_points_uniformly(
        number_of_points=num_queries_on_surface
    )

    queries_on_surface = torch.tensor(np.asarray(q_on_surf_o3d.points), dtype=torch.float)
    values_on_surface = torch.zeros(num_queries_on_surface)
    gradients_on_surface = torch.zeros(num_queries_on_surface, 3)

    queries = torch.cat([queries_on_surface, queries], dim=0)
    values = torch.cat([values_on_surface, values], dim=0)
    gradients = torch.cat([gradients_on_surface, gradients], dim=0)

    if convert_to_bce_labels:
        values /= max_dist
        values = 1 - values

    return queries, values, gradients


def sample_points_around_pcd(
        pcd: Tensor,
        stds: List[float],
        num_points_per_std: List[int],
        coords_range: Tuple[float, float],
        device: str = "cpu",
) -> Tensor:
    """Sample points around the given point cloud.
    Points are sampled by adding gaussian noise to the input cloud,
    according to the given standard deviations. Additionally, points
    are also sampled uniformly in the given range.
    Args:
        pcd: The point cloud tensor with shape (N, 3).
        stds: A list of standard deviations to compute the gaussian noise
            to obtain the points.
        num_points_per_std: A list with the number of points to sample for each
            standard deviation. The last number refers to points sampled uniformly
            in the given range (i.e., len(num_points_per_std) = len(stds) + 1).
        coords_range: The range for the points coordinates.
        device: The device for the sampled points. Defaults to "cpu".
    Returns:
        The sampled points with shape (M, 3).
    """
    coords = torch.empty(0, 3).to(device)
    num_points_pcd = pcd.shape[0]

    for sigma, num_points in zip(stds, num_points_per_std[:-1]):
        mul = num_points // num_points_pcd

        if mul > 0:
            coords_for_sampling = einops.repeat(pcd, "n d -> (n r) d", r=mul).to(device)
        else:
            coords_for_sampling = torch.empty(0, 3).to(device)

        still_needed = num_points % num_points_pcd
        if still_needed > 0:
            weights = torch.ones(num_points_pcd, dtype=torch.float).to(device)
            indices_random = torch.multinomial(weights, still_needed, replacement=False)
            pcd_random = pcd[indices_random].to(device)
            coords_for_sampling = torch.cat((coords_for_sampling, pcd_random), dim=0)

        offsets = torch.randn(num_points, 3).to(device) * sigma
        coords_i = coords_for_sampling + offsets

        coords = torch.cat((coords, coords_i), dim=0)

    random_coords = torch.rand(num_points_per_std[-1], 3).to(device)
    random_coords *= coords_range[1] - coords_range[0]
    random_coords += coords_range[0]
    coords = torch.cat((coords, random_coords), dim=0)

    coords = torch.clip(coords, min=coords_range[0], max=coords_range[1])

    return coords


def compute_udf_and_gradients(
        mesh_o3d: o3d.geometry.TriangleMesh,
        queries: Tensor,
) -> Tuple[Tensor, Tensor]:
    scene = o3d.t.geometry.RaycastingScene()
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float32)
    triangles = np.asarray(mesh_o3d.triangles, dtype=np.uint32)
    _ = scene.add_triangles(vertices, triangles)

    # signed_distance = scene.compute_signed_distance(query_point)
    closest_points = scene.compute_closest_points(queries.numpy())["points"]
    closest_points = torch.tensor(closest_points.numpy())

    q2p = queries - closest_points
    udf = torch.linalg.vector_norm(q2p, dim=-1)
    gradients = F.normalize(q2p, dim=-1)

    return udf, gradients


def compute_sdf_and_gradients(
        mesh_o3d: o3d.geometry.TriangleMesh,
        queries: Tensor,
) -> Tuple[Tensor, Tensor]:
    scene = o3d.t.geometry.RaycastingScene()
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float32)
    triangles = np.asarray(mesh_o3d.triangles, dtype=np.uint32)
    _ = scene.add_triangles(vertices, triangles)

    signed_distance = scene.compute_signed_distance(queries.numpy())
    closest_points = scene.compute_closest_points(queries.numpy())["points"]
    closest_points = torch.tensor(closest_points.numpy())
    signed_distance = torch.tensor(signed_distance.numpy())

    # print(signed_distance.shape)

    # gradients = np.zeros((signed_distance.shape[0], 3))
    q2p = queries - closest_points
    gradients = F.normalize(q2p, dim=-1)
    gradients = np.sign(signed_distance)[:, None] * gradients
    # print(gradients.shape)

    return signed_distance, gradients
