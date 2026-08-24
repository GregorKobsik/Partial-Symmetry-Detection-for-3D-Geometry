from argparse import ArgumentParser
from datetime import datetime
from tqdm.auto import tqdm
import os
import json

from glob import glob
import networkx as nx
import numpy as np

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points

from utils.icp_distance import icp_distance
from utils.point_cloud_utils import normalize_point_cloud

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

PARTNET_PATH = "data/PartNet/data_v0"
STATS_PATH = "data/PartNet/metadata/stats/train_val_test_split"
SYMMETRY_PATH = "data/PartNet/icp_part_symmetry_matrix/"
PSPSB_PATH = "data/PartNet/pre_segmented_partial_symmetry_benchmark/data_v0/"
DATA_DIR = "data_v0"
SPLITS = {"train": 0, "val": 1, "test": 2}
CATEGORIES = [
    "Bag",
    "Bed",
    "Bottle",
    "Bowl",
    "Chair",
    "Clock",
    "Dishwasher",
    "Display",
    "Door",
    "Earphone",
    "Faucet",
    "Hat",
    "Keyboard",
    "Knife",
    "Lamp",
    "Laptop",
    "Microwave",
    "Mug",
    "Refrigerator",
    "Scissors",
    "StorageFurniture",
    "Table",
    "TrashCan",
    "Vase",
]
SYM_THRESHOLDS = {
    "Bag": 0.0065,
    "Bed": 0.0001375,
    "Bottle": 0.006575,
    "Bowl": 0.017575,
    "Chair": 0.000525,
    "Clock": 0.0026875,
    "Dishwasher": 0.0001,
    "Display": 0.000325,
    "Door": 0.000325,
    "Earphone": 0.0001,
    "Faucet": 0.000325,
    "Hat": 0.0001,
    "Keyboard": 0.0000125,
    "Knife": 0.0001,
    "Lamp": 0.0002,
    "Laptop": 0.001427966,
    "Microwave": 0.000037813,
    "Mug": 0.0001,
    "Refrigerator": 0.0001,
    "Scissors": 0.0004,
    "StorageFurniture": 0.00075,
    "Table": 0.001,
    "TrashCan": 0.000775,
    "Vase": 0.000125,
}


def load_point_clouds(partnet_id):
    """Load per part point clouds"""
    try:
        return torch.load(f"{PSPSB_PATH}/{partnet_id}/point_clouds.pt")
    except:
        return None


def load_points_labels(partnet_id):
    """Load point cloud and labels."""
    try:
        points = torch.load(f"{PSPSB_PATH}/{partnet_id}/points_4096.pt")
        labels = torch.load(f"{PSPSB_PATH}/{partnet_id}/labels_4096.pt")
        return points, labels
    except:
        return None, None


def load_mesh_data(model_id):
    """Load mesh data for a given model ID."""
    paths = sorted(glob(f"{PARTNET_PATH}/{model_id}/objs/*.obj"))
    return load_objs_as_meshes(paths, load_textures=False)


def sample_point_clouds(meshes):
    """Samples normalized point clouds from meshes."""
    point_clouds = sample_points_from_meshes(meshes, num_samples=4096)
    point_clouds, _ = sample_farthest_points(point_clouds, K=512)
    return torch.stack([normalize_point_cloud(pcl) for pcl in point_clouds])


def load_symmmetry_threshold(partnet_id):
    """Load symmetry threshold."""
    # get model category
    try:
        metadata = json.load(open(f"{PARTNET_PATH}/{partnet_id}/meta.json"))
    except:
        return None
    model_cat = metadata["model_cat"]
    return SYM_THRESHOLDS[model_cat]


def load_similarity_matrix(partnet_id):
    """Load icp distance matrix defining similarities between parts."""
    try:
        return torch.load(f"{SYMMETRY_PATH}/{partnet_id}/icp_distances.pt")
    except:
        return None


def extract_point_labels(points_per_mesh, point_cloud):
    """Extract point labels from point cloud."""
    n_points_per_mesh = points_per_mesh.shape[1]
    labels_lookup = torch.concat(
        [torch.ones(n_points_per_mesh) * i for i in range(len(meshes))]
    )
    dist = torch.cdist(points_per_mesh.reshape(-1, 3), point_cloud)
    _, idx = torch.min(dist, dim=0)
    return labels_lookup[idx]


def extract_symmetric_parts(sim_matrix, sym_threshold):
    """Extract symmetric parts from point cloud using symmetry matrix.
    Args:
        sim_matrix: similarity matrix
        sym_threshold: threshold for similarity matrix
    """
    sym_matrix = sim_matrix < sym_threshold
    return list(nx.connected_components(nx.from_numpy_matrix(sym_matrix.numpy())))


def extract_symmetric_points_idx(symmetric_parts, point_labels):
    symmetric_point_idx = []
    for parts in symmetric_parts:
        if len(parts) == 1:
            continue  # skip components with only one part
        mask = torch.zeros(len(point_labels), dtype=bool)
        for p in parts:
            mask = mask | (point_labels == p)
        symmetric_point_idx.append(mask.nonzero().flatten())
    return symmetric_point_idx


def subsample_point_clouds(point_clouds):
    """
    Subsamples a batch of point clouds using farthest point sampling.

    Args:
        point_clouds (torch.Tensor): A tensor of shape (B, N, 3) representing a batch of point clouds.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - A tensor of shape (M, 3) representing the subsampled point cloud.
            - A tensor of shape (M,) representing the corresponding labels for the subsampled points.
    """
    point_labels = torch.tensor([512 * [i] for i, _ in enumerate(point_clouds)])

    point_cloud = point_clouds.reshape(1, -1, 3)
    point_label = point_labels.reshape(1, -1)

    sampled_pcl, sampled_idx = sample_farthest_points(point_cloud, K=4096)
    return sampled_pcl[0], point_label[0][sampled_idx[0]]


def compute_ICP_dists(point_clouds, symmetric_parts):
    """Computes the ICP distances between pairs of symmetric parts in a set of point clouds."""
    icp_dists = []
    for parts in symmetric_parts:
        if len(parts) > 1:
            icp_dists += [
                icp_distance(point_clouds[list(parts)], silent=True).numpy().max()
            ]
    return icp_dists


def save_data(partnet_id, point_clouds, points, labels):
    """Saves the given point clouds, points, and labels to disk."""
    os.makedirs(f"{PSPSB_PATH}/{partnet_id}", exist_ok=True)
    torch.save(point_clouds, f"{PSPSB_PATH}/{partnet_id}/point_clouds.pt")
    torch.save(points, f"{PSPSB_PATH}/{partnet_id}/points_4096.pt")
    torch.save(labels, f"{PSPSB_PATH}/{partnet_id}/labels_4096.pt")


def save_gt_data(partnet_id, sym_parts_idx, sym_points_idx, ICP_dist):
    """Saves ground truth data for a given PartNet model."""
    os.makedirs(f"{PSPSB_PATH}/{partnet_id}/ground_truth_approx", exist_ok=True)
    with open(
        f"{PSPSB_PATH}/{partnet_id}/ground_truth_approx/sym_parts_idx.json", "w"
    ) as f:
        json.dump([list(idx) for idx in sym_parts_idx], f)
    with open(
        f"{PSPSB_PATH}/{partnet_id}/ground_truth_approx/sym_points_idx.json", "w"
    ) as f:
        json.dump([list(int(i) for i in idx.numpy()) for idx in sym_points_idx], f)
    with open(f"{PSPSB_PATH}/{partnet_id}/ground_truth_approx/ICP_dist.json", "w") as f:
        json.dump(float(ICP_dist), f)


def compute_ICP_dist(point_clouds, symmetric_parts):
    """
    Computes the average ICP distance between pairs of symmetric parts in a set of point clouds.

    Args:
        point_clouds (list): A list of point clouds, where each point cloud is a numpy array of shape (N, 3).
        symmetric_parts (list): A list of lists, where each inner list contains the indices of symmetric parts in the point clouds.

    Returns:
        float: The average ICP distance between pairs of symmetric parts.
    """
    icp_dists = []
    for parts in symmetric_parts:
        if len(parts) > 1:
            icp_dists += [
                icp_distance(point_clouds[list(parts)], silent=True).numpy().max()
            ]
    return np.mean(icp_dists)


#######################
### PARSE ARGUMENTS ###
#######################

parser = ArgumentParser()
parser.add_argument(
    "-n", "--num_samples", dest="num_samples", default=2**16, type=int
)
parser.add_argument("-c", "--categories", dest="categories", default=None, type=str)
parser.add_argument("-s", "--split", dest="split", default="test", type=str)
args = parser.parse_args()

# retrive categories which we will process
if args.categories in (None, "all", "All", ["all"], ["All"]):
    args.categories = list(CATEGORIES)
if isinstance(args.categories, str):
    args.categories = [args.categories]
assert all(category in CATEGORIES for category in args.categories)

if args.split not in SPLITS:
    raise ValueError(
        (f"Split {args.split} found, but expected either train, val or test")
    )

# compose list of model directories
anno_ids = []
for category in args.categories:
    with open(f"{STATS_PATH}/{category}.{args.split}.json") as f:
        metadata_info = json.load(f)
    anno_ids += [object_info["anno_id"] for object_info in metadata_info]

####################
### PROCESS DATA ###
####################

pbar = tqdm(anno_ids, desc="Model")
for partnet_id in pbar:
    pbar.set_postfix({"partnet_id": partnet_id})
    if os.path.exists(f"{PSPSB_PATH}/{partnet_id}/ground_truth_approx"):
        continue
    os.makedirs(f"{PSPSB_PATH}/{partnet_id}/ground_truth_approx", exist_ok=True)

    # load data and sample points
    meshes = load_mesh_data(partnet_id)
    if len(meshes) > 30:
        continue  # skip objects with too many parts

    # BUG: sample points unnormalized, normalize only right before embedding
    point_clouds = sample_point_clouds(meshes)

    # load similarity matrix and symmetry threshold
    sim_matrix = load_similarity_matrix(partnet_id)
    sym_threshold = load_symmmetry_threshold(partnet_id)
    if sim_matrix is None or sym_threshold is None:
        continue

    # extract symmetries
    sym_parts_idx = extract_symmetric_parts(sim_matrix, sym_threshold)

    # extract a point-label cloud
    # BUG: point cloud overlaps multiple times with itself, thus wrong subsampling
    points, labels = subsample_point_clouds(point_clouds)
    # BUG: points are not unidistantly distributed, as subsampling is wrong
    sym_points_idx = extract_symmetric_points_idx(sym_parts_idx, labels)

    # compute ICP-Dist for gt symmetries
    ICP_dists = compute_ICP_dist(point_clouds, sym_parts_idx)

    # save data
    save_data(partnet_id, point_clouds, points, labels)
    save_gt_data(partnet_id, sym_parts_idx, sym_points_idx, ICP_dists)

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
