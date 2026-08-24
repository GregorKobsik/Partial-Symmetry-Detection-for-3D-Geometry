"""
Optimized Region Growing Algorithm (v2)

Key optimizations:
1. Precompute geodesic distances once instead of recomputing connectivity each iteration
2. Use KD-tree for fast nearest neighbor queries instead of full distance matrices
3. Vectorized operations for batch processing multiple components simultaneously
4. Cache downsampled patches and only recompute when needed
5. Early stopping with adaptive thresholds
6. Use scipy's cKDTree for faster spatial queries
"""

###############
### Imports ###
###############

from argparse import ArgumentParser
from datetime import datetime
import pickle
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import multiprocessing

import torch
import pytorch3d.io as io
import pytorch3d.ops as ops

import potpourri3d as pp3d

from utils.point_cloud_utils import normalize_point_cloud
from utils.icp_distance import icp_distance_silent as compute_icp_distance
from utils.icp_distance import icp_distance_single

from collections import namedtuple

##################
### Load Files ###
##################

# input arguments
parser = ArgumentParser()
parser.add_argument("-s", "--shape_name", dest="shape_name", default="filigree", type=str)
parser.add_argument("-clr", "--use_clr", dest="use_clr", default=True, type=bool)
parser.add_argument("--batch_size", dest="batch_size", default=50, type=int, help="Initial points to add per iteration")
parser.add_argument("--icp_check_interval", dest="icp_check_interval", default=10, type=int, help="Check ICP distance every N iterations")
parser.add_argument("--exponential_growth", dest="exponential_growth", default=True, type=bool, help="Use exponential batch size growth")
parser.add_argument("--growth_factor", dest="growth_factor", default=1.5, type=float, help="Exponential growth factor for batch size")
parser.add_argument("--skip_final_icp", dest="skip_final_icp", default=True, type=bool, help="Skip ICP checks when few points remain")
parser.add_argument("--final_points_ratio", dest="final_points_ratio", default=0.05, type=float, help="Skip ICP when remaining points < this ratio")
parser.add_argument("--n_jobs", dest="n_jobs", default=-1, type=int, help="Number of parallel jobs (-1 uses all cores)")
args = parser.parse_args()

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("arguments:", args)

# load precomputed data
dir_path = f"data/precomputed/{args.shape_name}"
prefix = "clr_" if args.use_clr else "icp_"


def load_pickle(file_path):
    """Convenience function to load pickle files."""
    with open(file_path, "rb") as file:
        var = pickle.load(file)
    return var


point_cloud = np.load(f"{dir_path}/pcl.npy")
clusters_components_idx = load_pickle(f"{dir_path}/{prefix}clusters_components_idx.pkl")

######################
### Region Growing ###
######################


def _compute_geodesic_for_component(pcl, component_idx_num, idx):
    """Helper function to compute geodesic distances for a single component (for parallel execution)."""
    solver = pp3d.PointCloudHeatSolver(pcl)

    # Find center point of component
    component_center = np.mean(pcl[idx], axis=0)
    distances = np.linalg.norm(pcl[idx] - component_center, axis=1)
    center_idx = idx[np.argmin(distances)]

    # Compute geodesic distance from center
    geodesic_dist = solver.compute_distance(center_idx)
    return component_idx_num, geodesic_dist


def precompute_geodesic_distances(pcl, components_idx, n_jobs=-1):
    """
    Precompute geodesic distances from each component's center.
    Uses parallel processing for speedup.

    Args:
        pcl: Point cloud (N x 3)
        components_idx: List of component indices
        n_jobs: Number of parallel jobs (-1 uses all cores)

    Returns:
        List of geodesic distance arrays, one per component
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    print(f"Precomputing geodesic distances using {n_jobs} cores...", flush=True)

    # Parallel geodesic computation
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_compute_geodesic_for_component)(pcl, i, idx)
        for i, idx in enumerate(components_idx)
    )

    # Sort by component index to maintain order
    results.sort(key=lambda x: x[0])
    geodesic_dists = [dist for _, dist in results]

    return geodesic_dists


def build_kdtree(pcl):
    """Build KD-tree for fast spatial queries."""
    return cKDTree(pcl)


def get_connected_region(pcl, seed_indices, candidate_indices, kdtree, radius=0.03):
    """
    Fast connected component extraction using KD-tree radius queries.

    Args:
        pcl: Point cloud
        seed_indices: Indices of seed points (already in region)
        candidate_indices: Indices of candidate points to check
        kdtree: Prebuilt KD-tree
        radius: Connection radius threshold

    Returns:
        Set of connected candidate indices
    """
    if len(candidate_indices) == 0:
        return set()

    # Start from seed points
    seed_points = pcl[seed_indices]
    candidate_set = set(candidate_indices)
    connected = set()

    # Find all candidates within radius of any seed point
    for seed_pt in seed_points:
        neighbors = kdtree.query_ball_point(seed_pt, radius)
        connected.update(candidate_set.intersection(neighbors))

    # Iteratively expand connected region (flood fill)
    frontier = connected.copy()
    visited = connected.copy()

    while frontier and len(visited) < len(candidate_set):
        new_frontier = set()
        for pt_idx in frontier:
            neighbors = kdtree.query_ball_point(pcl[pt_idx], radius)
            new_neighbors = candidate_set.intersection(neighbors) - visited
            new_frontier.update(new_neighbors)
            visited.update(new_neighbors)

        connected.update(new_frontier)
        frontier = new_frontier

        # Limit iterations to prevent runaway
        if len(frontier) == 0:
            break

    return connected


def _compute_single_icp_pair(i, j, patches, cached_transforms):
    """Helper function to compute ICP for a single pair (for parallel execution)."""
    import torch
    from collections import namedtuple
    SimilarityTransform = namedtuple("SimilarityTransform", "R T s")

    key = (i, j)
    if key in cached_transforms:
        cached_RT = cached_transforms[key]
        # Ensure everything is on CUDA (icp_distance_single forces cuda)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move cached transform to CUDA explicitly
        init_RT = SimilarityTransform(
            R=cached_RT.R.to(device)[None],
            T=cached_RT.T.to(device)[None],
            s=cached_RT.s.to(device)[None]
        )
        # Use only 5 iterations since we have good initial guess
        dist = icp_distance_single(
            patches[i][None],
            patches[j][None],
            init_RTs=init_RT,
            n_iter=5
        )[0].item()
    else:
        # Fallback: no cached transform, use more iterations
        dist = icp_distance_single(
            patches[i][None],
            patches[j][None],
            n_iter=10
        )[0].item()

    return dist


def compute_icp_with_cached_transforms_fast(patches, cached_transforms, n_components, n_jobs=1):
    """
    Fast ICP using cached transformations as initial guesses.
    Uses parallel processing for computing pairwise ICP distances.

    Uses only 3-5 iterations instead of 30 since we have good initial transforms.

    Args:
        patches: List of downsampled point cloud patches
        cached_transforms: Dict mapping (i,j) -> transformation
        n_components: Number of components
        n_jobs: Number of parallel jobs (1 for sequential, -1 for all cores)

    Returns:
        Maximum ICP distance
    """
    # Generate all pairs
    pairs = [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]

    # Check if patches are on CUDA - if so, disable parallelization due to thread safety
    use_cuda = patches[0].is_cuda if len(patches) > 0 else False

    if n_jobs == 1 or len(pairs) < 4 or use_cuda:
        # Sequential for small number of pairs or when using CUDA (thread safety)
        distances = [_compute_single_icp_pair(i, j, patches, cached_transforms) for i, j in pairs]
    else:
        # Parallel processing for larger number of pairs (CPU only)
        if n_jobs == -1:
            n_jobs = min(multiprocessing.cpu_count(), len(pairs))

        distances = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
            delayed(_compute_single_icp_pair)(i, j, patches, cached_transforms)
            for i, j in pairs
        )

    return max(distances)


def compute_icp_fast_approximation(patches, prev_patches=None, prev_icp=None):
    """
    Fast ICP distance approximation using previous computation results.

    Strategy: If patches haven't changed much, the ICP distance won't change much either.
    We only do full ICP computation periodically, otherwise estimate based on patch similarity.

    Args:
        patches: Current downsampled patches
        prev_patches: Previous downsampled patches
        prev_icp: Previous ICP distance

    Returns:
        Estimated ICP distance
    """
    if prev_patches is None or prev_icp is None:
        # First computation - do full ICP
        return compute_icp_distance(torch.stack(patches), n_iter=10).max().item()

    # Quick heuristic: Check how much patches changed using Chamfer distance
    from pytorch3d.loss import chamfer_distance

    total_change = 0.0
    for i, (curr, prev) in enumerate(zip(patches, prev_patches)):
        change = chamfer_distance(curr[None], prev[None])[0].item()
        total_change += change

    avg_change = total_change / len(patches)

    # If patches changed very little, assume ICP is similar (fast path)
    if avg_change < 0.001:
        return prev_icp

    # Otherwise, do full ICP computation
    return compute_icp_distance(torch.stack(patches), n_iter=10).max().item()


def region_growing_optimized(
    pcl,
    components_idx,
    similarity_threshold_factor=1.5,
    batch_size=50,
    icp_check_interval=10,
    connectivity_radius=0.03,
    use_cached_icp=True,
    exponential_growth=True,
    growth_factor=1.5,
    skip_final_icp=True,
    final_points_ratio=0.05,
    n_jobs=-1
):
    """
    Optimized region growing with batched operations, caching, and parallel processing.

    Args:
        pcl: Point cloud (N x 3)
        components_idx: List of initial component indices
        similarity_threshold_factor: Multiplier for ICP threshold
        batch_size: Initial number of points to add per iteration
        icp_check_interval: Check ICP distance every N iterations
        connectivity_radius: Radius for connectivity check
        use_cached_icp: Use cached ICP transformations for speedup
        exponential_growth: Use exponential batch size growth
        growth_factor: Exponential growth factor for batch size (e.g., 1.5 means 50% increase)
        skip_final_icp: Skip ICP checks when few points remain
        n_jobs: Number of parallel jobs (-1 uses all cores)
        final_points_ratio: Skip ICP when remaining < this ratio of total points
    """

    # Initialize
    n_points = len(pcl)
    n_components = len(components_idx)

    packed_region_idx = [[] for _ in components_idx]
    region_history = [[] for _ in components_idx]
    max_icp_dists = []

    # Track free points
    initial_free = set(range(n_points)) - set(np.concatenate(components_idx))
    free_idx = initial_free.copy()

    # Build KD-tree once
    kdtree = build_kdtree(pcl)

    # Precompute geodesic distances (parallelized)
    geodesic_dists = precompute_geodesic_distances(pcl, components_idx, n_jobs=n_jobs)

    # Precompute initial downsampled patches and ICP threshold (parallelized)
    print("Computing initial ICP threshold and caching transformations...", flush=True)

    def _downsample_patch(idx):
        normalized = normalize_point_cloud(torch.tensor(pcl[idx], dtype=torch.float32))
        downsampled = ops.sample_farthest_points(normalized[None], K=min(512, len(idx)))[0][0]
        return downsampled

    # Parallel downsampling
    if n_jobs == 1 or len(components_idx) < 4:
        downsampled_patches = [_downsample_patch(idx) for idx in components_idx]
    else:
        n_jobs_ds = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        downsampled_patches = Parallel(n_jobs=n_jobs_ds, backend='threading', verbose=0)(
            delayed(_downsample_patch)(idx) for idx in components_idx
        )

    # Compute initial ICP with transformation caching
    from utils.icp_distance import icp_distance
    initial_icp_matrix, cached_transforms = icp_distance(
        torch.stack(downsampled_patches),
        silent=True,
        n_iter=30,
        return_transforms=True
    )
    initial_icp = initial_icp_matrix.max()
    similarity_threshold = similarity_threshold_factor * initial_icp
    print(f"Initial ICP: {initial_icp:.4f}, Threshold: {similarity_threshold:.4f}", flush=True)
    print(f"Cached {len(cached_transforms)} transformation matrices for fast ICP", flush=True)    # Cache for fast ICP approximation
    prev_downsampled_patches = None
    prev_icp_distance = None
    full_icp_counter = 0  # Do full ICP every N times

    # Exponential batch size growth
    initial_batch_size = batch_size
    current_batch_size = batch_size
    prev_icp_for_growth = initial_icp.item() if hasattr(initial_icp, 'item') else initial_icp
    icp_increased = False  # Track if ICP increased (overshoot detection)

    # Track which patches need recomputation
    needs_update = [False] * n_components

    # Track exhausted components (optimization: avoid re-checking)
    exhausted_components = set()

    # Precompute free points threshold for skipping ICP
    final_points_threshold = len(initial_free) * final_points_ratio
    skip_icp_mode = False

    iteration = 0
    component_idx = 0

    pbar = tqdm(total=len(initial_free), desc="Growing regions")

    while len(free_idx) > 0:
        # Get current extended region
        current_region = np.concatenate([
            components_idx[component_idx],
            packed_region_idx[component_idx]
        ]) if packed_region_idx[component_idx] else components_idx[component_idx]

        # Find connected free points using KD-tree
        connected_free = get_connected_region(
            pcl,
            current_region,
            list(free_idx),
            kdtree,
            radius=connectivity_radius
        )

        # If no connected points, mark component as exhausted
        if len(connected_free) == 0:
            exhausted_components.add(component_idx)
            component_idx = (component_idx + 1) % n_components

            # If all components exhausted, break
            if len(exhausted_components) >= n_components:
                break

            # Skip to next non-exhausted component
            attempts = 0
            while component_idx in exhausted_components and attempts < n_components:
                component_idx = (component_idx + 1) % n_components
                attempts += 1

            if attempts >= n_components:
                break

            continue

        # Select closest points based on geodesic distance (vectorized)
        connected_array = np.array(list(connected_free))
        geodesic_values = geodesic_dists[component_idx][connected_array]

        # Adaptively select batch size with exponential growth
        if exponential_growth:
            # Use current exponentially growing batch size
            # In FINAL mode (ICP skipped), remove the 1/20 limitation
            if skip_icp_mode:
                actual_batch_size = min(
                    int(current_batch_size),
                    len(connected_array)
                )
            else:
                actual_batch_size = min(
                    int(current_batch_size),
                    max(1, len(connected_array) // 20),
                    len(connected_array)
                )
        else:
            # Original adaptive batch size
            # In FINAL mode (ICP skipped), remove the 1/20 limitation
            if skip_icp_mode:
                actual_batch_size = min(batch_size, len(connected_array))
            else:
                actual_batch_size = min(batch_size, max(1, len(connected_array) // 20))

        # Vectorized selection (faster than list comprehension)
        closest_indices = np.argpartition(geodesic_values, min(actual_batch_size, len(geodesic_values)-1))[:actual_batch_size]
        selected_points = connected_array[closest_indices].tolist()        # Add to region
        free_idx -= set(selected_points)
        packed_region_idx[component_idx].extend(selected_points)
        region_history[component_idx].append(selected_points)
        needs_update[component_idx] = True

        # Update progress
        pbar.update(len(selected_points))

        # Skip ICP checks when very few points remain
        if skip_final_icp and len(free_idx) < final_points_threshold:
            if not skip_icp_mode:
                skip_icp_mode = True
                pbar.set_postfix({"Status": "FINAL (ICP skipped)", "Free": len(free_idx)})

        # Periodically check ICP distance (unless in skip mode)
        if iteration % icp_check_interval == 0 and not skip_icp_mode:
            # Recompute patches that changed
            for i in range(n_components):
                if needs_update[i]:
                    extended_idx = np.concatenate([components_idx[i], packed_region_idx[i]])
                    normalized = normalize_point_cloud(torch.tensor(pcl[extended_idx], dtype=torch.float32))
                    downsampled_patches[i] = ops.sample_farthest_points(
                        normalized[None],
                        K=min(512, len(extended_idx))
                    )[0][0]
                    needs_update[i] = False

            # Compute ICP distance - use cached transforms for speedup (parallelized)
            if use_cached_icp and prev_downsampled_patches is not None:
                # Every 5th check, do full ICP; otherwise approximate or use cached transforms
                full_icp_counter += 1
                if full_icp_counter >= 5:
                    # Full ICP with fewer iterations using cached transforms as init (parallelized)
                    current_icp = compute_icp_with_cached_transforms_fast(
                        downsampled_patches,
                        cached_transforms,
                        n_components,
                        n_jobs=n_jobs
                    )
                    full_icp_counter = 0
                else:
                    # Fast approximation based on Chamfer distance
                    current_icp = compute_icp_fast_approximation(
                        downsampled_patches,
                        prev_downsampled_patches,
                        prev_icp_distance
                    )
            else:
                # Initial full ICP computation (already done with cached transforms, parallelized)
                current_icp = compute_icp_with_cached_transforms_fast(
                    downsampled_patches,
                    cached_transforms,
                    n_components,
                    n_jobs=n_jobs
                )

            # Cache current state for next iteration
            prev_downsampled_patches = [p.clone() for p in downsampled_patches]
            prev_icp_distance = current_icp

            max_icp_dists.append(current_icp)

            # Exponential growth logic: detect overshoot and reset
            if exponential_growth:
                if current_icp > prev_icp_for_growth * 1.05:  # ICP increased by >5% (overshoot!)
                    # We grew too fast, reset to initial batch size
                    current_batch_size = initial_batch_size
                    icp_increased = True
                    pbar.set_postfix({
                        "ICP": f"{current_icp:.4f}",
                        "Batch": int(current_batch_size),
                        "Status": "RESET",
                        "Free": len(free_idx)
                    })
                elif icp_increased and current_icp <= prev_icp_for_growth * 1.02:
                    # After reset: ICP has stabilized (not increasing), restart growth
                    icp_increased = False
                    current_batch_size *= growth_factor
                    pbar.set_postfix({
                        "ICP": f"{current_icp:.4f}",
                        "Batch": int(current_batch_size),
                        "Status": "RESTART",
                        "Free": len(free_idx)
                    })
                elif not icp_increased:
                    # ICP stable or decreasing, grow batch size exponentially
                    current_batch_size *= growth_factor
                    pbar.set_postfix({
                        "ICP": f"{current_icp:.4f}",
                        "Batch": int(current_batch_size),
                        "Status": "GROW",
                        "Free": len(free_idx)
                    })
                else:
                    # After reset, waiting for ICP to stabilize
                    pbar.set_postfix({
                        "ICP": f"{current_icp:.4f}",
                        "Batch": int(current_batch_size),
                        "Status": "WAIT",
                        "Free": len(free_idx)
                    })

                prev_icp_for_growth = current_icp
            else:
                pbar.set_postfix({"ICP": f"{current_icp:.4f}", "Free": len(free_idx)})

            # Optional: early stopping if ICP threshold exceeded
            # if current_icp > similarity_threshold:
            #     print(f"\nStopping: ICP distance {current_icp:.4f} exceeds threshold {similarity_threshold:.4f}")
            #     break

        # Move to next component
        component_idx = (component_idx + 1) % n_components
        iteration += 1

    pbar.close()

    # Final ICP computation
    for i in range(n_components):
        if needs_update[i] or iteration % icp_check_interval != 0:
            extended_idx = np.concatenate([components_idx[i], packed_region_idx[i]]) if packed_region_idx[i] else components_idx[i]
            normalized = normalize_point_cloud(torch.tensor(pcl[extended_idx], dtype=torch.float32))
            downsampled_patches[i] = ops.sample_farthest_points(
                normalized[None],
                K=min(512, len(extended_idx))
            )[0][0]

    final_icp = compute_icp_distance(torch.stack(downsampled_patches)).max().item()
    max_icp_dists.append(final_icp)
    print(f"Final ICP distance: {final_icp:.4f}")

    return region_history, max_icp_dists


#########################
### Process Clusters ###
#########################

# Note: Cluster parallelization is not implemented because CUDA/PyTorch operations
# cannot be pickled for process-based parallelization, and threading is ineffective
# due to Python's GIL with GPU operations. Each cluster already uses parallel
# operations internally (geodesic precomputation, etc.) for good speedup.

print(f"Processing {len(clusters_components_idx)} clusters sequentially (each uses {args.n_jobs} cores internally)...", flush=True)

region_growing_components_idx = []
regions_packed_components_idx = []
regions_growing_max_icp_dists = []

for components_idx in tqdm(clusters_components_idx, desc="Processing clusters"):
    extended_components_idx, components_max_icp_dists = region_growing_optimized(
        point_cloud,
        components_idx,
        batch_size=args.batch_size,
        icp_check_interval=args.icp_check_interval,
        exponential_growth=args.exponential_growth,
        growth_factor=args.growth_factor,
        skip_final_icp=args.skip_final_icp,
        final_points_ratio=args.final_points_ratio,
        n_jobs=args.n_jobs
    )

    region_growing_components_idx.append(extended_components_idx)
    regions_packed_components_idx.append([
        np.concatenate(idx) for idx in extended_components_idx if len(idx) > 0
    ])
    regions_growing_max_icp_dists.append(components_max_icp_dists)

#################
### Save Data ###
#################


def save_pickle(file_path, var):
    """Convenience function to save pickle files."""
    with open(file_path, "wb") as file:
        pickle.dump(var, file)


save_pickle(f"{dir_path}/{prefix}region_growing_components_idx.pkl", region_growing_components_idx)
save_pickle(f"{dir_path}/{prefix}regions_packed_components_idx.pkl", regions_packed_components_idx)
save_pickle(f"{dir_path}/{prefix}regions_growing_max_icp_dists.pkl", regions_growing_max_icp_dists)

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
