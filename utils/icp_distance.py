from tqdm.auto import tqdm
import numpy as np

import torch
from pytorch3d.transforms import random_rotations
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import iterative_closest_point

import warnings
from collections import namedtuple
from typing import Optional, Union

Device = Union[str, torch.device]


def get_reflections(
    n: int,
    type: int = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
):
    ref_mats = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(8, 1, 1)
    ref_mats[1, 0, 0] = -1
    ref_mats[2, 1, 1] = -1
    ref_mats[3, 2, 2] = -1
    ref_mats[4, 0, 0] = -1
    ref_mats[4, 1, 1] = -1
    ref_mats[5, 0, 0] = -1
    ref_mats[5, 2, 2] = -1
    ref_mats[6, 1, 1] = -1
    ref_mats[6, 2, 2] = -1
    ref_mats[7, 0, 0] = -1
    ref_mats[7, 1, 1] = -1
    ref_mats[7, 2, 2] = -1

    if type is None:
        type = np.random.randint(0, 8)
    return ref_mats[type % 8].tile(n, 1, 1)


def get_R_matrix(
    n: int,
    iter: int = 0,
    only_reflections_first: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
):
    if only_reflections_first and iter < 8:
        return get_reflections(n, device=device, type=iter % 8)
    return random_rotations(n, dtype=dtype, device=device) @ get_reflections(
        n, device=device, type=iter % 8
    )


def icp_distance_single(points_x, points_y, init_RTs=None, n_iter=30, return_transforms=False):
    """Computes the chamfer distance between two point clouds using ICP.
    Args:
        points_x: torch.tensor of shape (batch, num_points, 3)
        points_y: torch.tensor of shape (batch, num_points, 3)
        init_RTs: Optional initial transformation to use as starting point
        n_iter: Number of random initializations to try
        return_transforms: If True, also return the best transformation

    Returns:
        chamfer_distance: torch.tensor of shape (1,)
        best_RTs: (optional) Best transformation found (if return_transforms=True)
    """

    dev = "cuda"
    x = points_x.to(dev)
    y = points_y.to(dev)

    # perform ICP for each sample individually
    SimilarityTransform = namedtuple("SimilarityTransform", "R T s")

    chamfer_distances = []
    best_RTs_list = []

    for i in range(n_iter):
        if init_RTs is not None and i == 0:
            random_transforms = init_RTs
        else:
            random_transforms = SimilarityTransform(
                R=get_R_matrix(
                    x.shape[0], device=dev, only_reflections_first=True, iter=i
                ),
                T=torch.zeros(x.shape[0], 3, device=dev),
                s=torch.ones(x.shape[0], device=dev),
            )

        converged, rmse, Xt, RTs, t_history = iterative_closest_point(
            x,
            y,
            estimate_scale=False,
            allow_reflection=True,
            init_transform=random_transforms,
            max_iterations=100,
        )

        dist = chamfer_distance(Xt, y, batch_reduction=None)[0]
        chamfer_distances.append(dist.cpu())
        best_RTs_list.append(RTs)

    # Find best transformation (minimum distance)
    chamfer_distances_tensor = torch.stack(chamfer_distances)
    min_indices = torch.argmin(chamfer_distances_tensor, dim=0)
    best_distances = torch.min(chamfer_distances_tensor, dim=0)[0]

    if return_transforms:
        # Extract the best RT for each point cloud pair
        best_RTs = SimilarityTransform(
            R=torch.stack([best_RTs_list[idx].R[i] for i, idx in enumerate(min_indices)]),
            T=torch.stack([best_RTs_list[idx].T[i] for i, idx in enumerate(min_indices)]),
            s=torch.stack([best_RTs_list[idx].s[i] for i, idx in enumerate(min_indices)])
        )
        return best_distances.detach().clone(), best_RTs

    return best_distances.detach().clone()


def icp_distance(points, silent=False, n_iter=30, return_transforms=False):
    """Computes the chamfer distance matrix between every two point cloud combinations using ICP.
    Args:
        points: torch.tensor of shape (batch_size, num_points, 3)
        silent: If True, disable progress bar
        n_iter: Number of random initializations per pair
        return_transforms: If True, also return cached transformations

    Returns:
        distance_matrix: torch.tensor of shape (batch_size, batch_size)
        transforms_cache: (optional) Dict mapping (i,j) -> best_RT (if return_transforms=True)
    """
    dev = "cuda"
    points = points.to(dev)

    # perform ICP for each sample individually
    distance_matrix = torch.zeros(points.shape[0], points.shape[0])
    transforms_cache = {} if return_transforms else None

    for idx in tqdm(range(points.shape[0]), desc="Compute Distances", disable=silent):
        length = points.shape[0]
        combinations = torch.tensor(
            np.array(np.meshgrid([idx], range(length))).T.reshape(-1, 2), device=dev
        )

        X = torch.index_select(points, dim=0, index=combinations[:, 0])
        Y = torch.index_select(points, dim=0, index=combinations[:, 1])

        if return_transforms:
            distances, RTs = icp_distance_single(X, Y, n_iter=n_iter, return_transforms=True)
            distance_matrix[idx] = distances
            # Store transforms for each pair
            for j in range(length):
                transforms_cache[(idx, j)] = RTs._replace(
                    R=RTs.R[j].cpu(),
                    T=RTs.T[j].cpu(),
                    s=RTs.s[j].cpu()
                )
        else:
            distance_matrix[idx] = icp_distance_single(X, Y, n_iter=n_iter)

    if return_transforms:
        return distance_matrix, transforms_cache
    return distance_matrix


def icp_distance_silent(points, n_iter=30):
    """Computes the chamfer distance matrix between every two point cloud combinations using ICP."""
    warnings.filterwarnings("ignore")
    return icp_distance(points, silent=True, n_iter=n_iter)
