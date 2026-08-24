from glob import glob
import numpy as np
from tqdm.auto import tqdm

import torch
import pytorch3d.ops as ops

from scipy.spatial.distance import cdist
from hdbscan import HDBSCAN
import potpourri3d as pp3d

from utils.point_cloud_utils import normalize_point_cloud
from utils.icp_distance import icp_distance
from utils.geodesic_utils import fix_geodesic_distance_for_disconnected_components

from siamese_models import ContrastiveSimilarityModel


# ---- multiprocessing worker state for sample_geodesic_patches ----
# Each worker process builds its own heat solver once (initializer) and then
# processes a chunk of patch centers. The geodesic solve per center is the
# bottleneck and is fully independent, so this scales near-linearly with cores.
_WORKER = {}


def _init_patch_worker(points_np, origin_idx_np, patch_size):
    import potpourri3d as pp3d  # local import for spawned/forked workers
    torch.set_num_threads(1)    # avoid thread oversubscription across workers
    _WORKER["points"] = torch.tensor(points_np)
    _WORKER["solver"] = pp3d.PointCloudHeatSolver(points_np)
    _WORKER["origins"] = origin_idx_np
    _WORKER["patch_size"] = patch_size


def _process_center(i):
    points = _WORKER["points"]
    solver = _WORKER["solver"]
    origins = _WORKER["origins"]
    patch_size = _WORKER["patch_size"]

    dists = solver.compute_distance(int(i))
    dist_to_origin = dists[origins]
    sort_idx = np.argsort(dists)

    patch_idx = []
    for s in patch_size:
        idx = sort_idx[:s]
        downsampled = ops.sample_farthest_points(
            points=points[idx][None], K=512
        )[1][0].numpy()
        patch_idx.append(idx[downsampled])
    return dist_to_origin, patch_idx


def sample_geodesic_patches(
    points, num_patches=1000, patch_size=[512, 1024, 2048, 4096, 8192],
    silent=True, num_workers=None,
):
    """Sample geodesic patches from a point cloud using farthest point sampling.

    num_workers: process count for the per-center geodesic solve. None -> use
    min(cpu_count, 4). 1 disables multiprocessing (serial fallback)."""
    import os
    from multiprocessing import get_context

    # sample `num_patches` points using FPS as patch centers
    points = torch.tensor(points)
    patches_origin_idx = ops.sample_farthest_points(
        points=points[None, :], K=num_patches
    )[1][0]

    points_np = points.numpy().astype(np.float32)
    origin_np = patches_origin_idx.numpy()
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 4)

    patches_idx = []
    dists_to_origin = []
    if num_workers <= 1:
        _init_patch_worker(points_np, origin_np, patch_size)
        results = [
            _process_center(i)
            for i in tqdm(origin_np, desc="Sample Patches", disable=silent)
        ]
    else:
        ctx = get_context("fork")
        with ctx.Pool(
            num_workers, initializer=_init_patch_worker,
            initargs=(points_np, origin_np, patch_size),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_center, origin_np, chunksize=8),
                    total=len(origin_np), desc=f"Sample Patches ({num_workers}p)",
                    disable=silent,
                )
            )

    for dist_to_origin, patch_idx in results:
        dists_to_origin.append(dist_to_origin)
        patches_idx.extend(patch_idx)

    # stack patches into arrays
    patches_idx = np.stack(patches_idx)
    patches_origin_dist = (
        np.stack(dists_to_origin)
        .repeat(len(patch_size), axis=0)
        .repeat(len(patch_size), axis=1)
    )
    patches_origin_dist = fix_geodesic_distance_for_disconnected_components(
        patches_origin_dist
    )
    patches_origin_idx = patches_origin_idx.repeat_interleave(len(patch_size))

    return patches_idx, patches_origin_dist, patches_origin_idx


def embed_features(point_cloud, patches_idx, model_id, silent=True):
    """Embed patches into the latent space of the learned model."""
    # load model
    run_path = f"logs/run_{model_id}"
    ckpt_paths = f"{run_path}/*.ckpt"
    ckpt_path = sorted(glob(ckpt_paths))[0]
    model = ContrastiveSimilarityModel.load_from_checkpoint(ckpt_path).cuda().eval()

    # embed features
    features = []
    with torch.no_grad():
        for idx in tqdm(patches_idx, desc="Embed Features", disable=silent):
            input_pcl = normalize_point_cloud(
                torch.tensor(point_cloud[idx], device="cuda")
            )[None]
            features += model(input_pcl)
    features = torch.stack(features).cpu().detach().numpy()

    # compute cosine similarity
    features_dist = cdist(features, features, metric="cosine")

    return features, features_dist


def compute_clusters(distances, features=None, min_cluster_size=15, min_samples=5, silent=True):
    """Compute clusters based on cosine distance of the learned features."""
    # cluster learned features in latent space using cosine distance
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="precomputed"
    ).fit_predict(distances.astype(np.double))
    clusters_patch_idx = [
        (labels == idx).nonzero()[0] for idx in np.unique(labels) if idx >= 0
    ]

    if not silent:
        print(f"Found {len(clusters_patch_idx)} clusters")

    if features is not None:
        # sort clusters by feature variance
        clusters_features_var = [
            np.var(features[labels == idx]) for idx in np.unique(labels) if idx >= 0
        ]
        sort_idx = np.argsort(clusters_features_var)

        clusters_features_var = [clusters_features_var[idx] for idx in sort_idx]
        clusters_patch_idx = [clusters_patch_idx[idx] for idx in sort_idx]

        return clusters_patch_idx, clusters_features_var

    return clusters_patch_idx


def subdivide_clusters(
    point_cloud, patches_idx, clusters_patch_idx, patches_origin_dist, silent=True
):
    """Subdivide clusters in components based on geodesic distance of the patches"""
    clusters_merged_patches_idx = []
    for cluster in tqdm(clusters_patch_idx, desc="Subdivide Clusters", disable=silent):
        patch_distances = patches_origin_dist[cluster][:, cluster]

        # compute a clustering into components based on the geodesic distance
        labels_components = HDBSCAN(
            min_cluster_size=3, cluster_selection_epsilon=0.1, metric="precomputed"
        ).fit_predict(patch_distances.astype(np.double))

        if max(labels_components) < 1:
            continue  # filter out clusters with only a single component
        cmps_idx = [
            cluster[labels_components == i]
            for i in np.unique(labels_components)
            if i >= 0
        ]

        # merge components into a single cluster
        merged_patches_idx = [
            np.unique(np.concatenate(patches_idx[idx])) for idx in cmps_idx
        ]
        merged_patches_idx = [
            np.unique(
                ops.ball_query(
                    torch.tensor(point_cloud[idx])[None],
                    torch.tensor(point_cloud)[None],
                    K=50,
                    radius=0.02,
                ).idx
            )[1:]
            for idx in merged_patches_idx
        ]
        clusters_merged_patches_idx += [merged_patches_idx]

    return clusters_merged_patches_idx


def filter_symmetry_hypotheses(
    point_cloud, cluster_merged_patches_idx, symmetry_threshold=0.005, silent=True
):
    # filter out points that are part of multiple patches

    clusters_components_idx = []
    clusters_icp_distance = []
    for n, merged_patches_idx in enumerate(
        tqdm(cluster_merged_patches_idx, desc="Filter Symmetries", disable=silent)
    ):
        merged_patches_filtered_idx = []
        for i, idx in enumerate(merged_patches_idx):
            union = set.union(
                *[set(p_idx) for j, p_idx in enumerate(merged_patches_idx) if i != j]
            )
            merged_patches_filtered_idx += [list(set(idx) - union)]
        merged_patches_filtered_idx = [
            idx for idx in merged_patches_filtered_idx if len(idx) > 512
        ]
        if len(merged_patches_filtered_idx) < 2:
            if not silent:
                print(f"Filter out cluster {n} - single component")
            continue  # filter out clusters with only a single component

        # compute ICP-distance between components
        downsampled_patches = [
            ops.sample_farthest_points(
                normalize_point_cloud(torch.tensor(point_cloud[idx]))[None], K=512
            )[0][0]
            for idx in merged_patches_filtered_idx
        ]
        components_icp_distances = icp_distance(
            torch.stack(downsampled_patches), silent=True
        )
        if components_icp_distances.max() > symmetry_threshold:
            if not silent:
                print(
                    f"Filter out cluster {n} - too large ICP-Dist. {components_icp_distances.max()}"
                )
            continue  # filter out components with too large distance

        clusters_components_idx += [merged_patches_filtered_idx]
        clusters_icp_distance += [components_icp_distances.max()]

    if not silent:
        print(f"Remaining symmetries: {len(clusters_components_idx)}")

    return clusters_components_idx, clusters_icp_distance


def extract_symmetries(
    point_cloud,
    model_id,
    num_patches=1000,
    patch_size=[512, 1024, 2048, 4096, 8192],
    min_cluster_size=20,
    symmetry_threshold=0.005,
    silent=True,
):
    """Extract symmetries from a point cloud using a learned model."""

    patches_idx, patches_origin_dist, patches_origin_idx = sample_geodesic_patches(
        point_cloud, num_patches, patch_size, silent
    )
    features, distances = embed_features(point_cloud, patches_idx, model_id, silent)
    clusters_patch_idx, clusters_features_var = compute_clusters(
        distances, features, min_cluster_size, silent
    )
    clusters_merged_patches_idx = subdivide_clusters(
        point_cloud, patches_idx, clusters_patch_idx, patches_origin_dist, silent
    )
    clusters_components_idx, clusters_icp_distance = filter_symmetry_hypotheses(
        point_cloud, clusters_merged_patches_idx, symmetry_threshold, silent
    )

    return clusters_components_idx, clusters_icp_distance
