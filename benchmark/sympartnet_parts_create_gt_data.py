from argparse import ArgumentParser
from datetime import datetime
from tqdm.auto import tqdm
import os
from glob import glob
import json

import numpy as np
import networkx as nx

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points

from utils.point_cloud_utils import normalize_point_cloud
from utils.icp_distance import icp_distance

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

PARTNET_PATH = "data/PartNet"
SYMMETRY_PATH = "data/PartNet/icp_part_symmetry_matrix"
STATS_DIR = "metadata/stats/train_val_test_split"
PSPSB_PATH = "data/PartNet/pre_segmented_partial_symmetry_benchmark/data_v0"
DATA_DIR = "data_v0"
SPLITS = {"train": 0, "val": 1, "test": 2}
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


def load_icp_similarity_matrix(model_id):
    """Load the ICP similarity matrix for a given model ID."""
    return torch.load(f"{SYMMETRY_PATH}/{model_id}/icp_distances.pt").numpy()


def load_mesh_data(model_id):
    """Load mesh data for a given model ID."""
    paths = sorted(glob(f"{PARTNET_PATH}/{DATA_DIR}/{model_id}/objs/*.obj"))
    return load_objs_as_meshes(paths, load_textures=False)


def sample_point_clouds(meshes):
    """Samples normalized point clouds from meshes."""
    point_clouds = sample_points_from_meshes(meshes, num_samples=4096)
    point_clouds, _ = sample_farthest_points(point_clouds, K=512)
    return torch.stack([normalize_point_cloud(pcl) for pcl in point_clouds])


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


def extract_symmetric_parts(distances, sym_threshold=0.1):
    """
    Extracts symmetric parts from a distance matrix.

    Parameters:
    distances (numpy.ndarray): A distance matrix.
    sym_threshold (float): Threshold for considering two parts as symmetric.

    Returns:
    list: A list of sets, where each set contains the indices of parts that are symmetric to each other.
    """
    return list(
        nx.connected_components(nx.from_numpy_matrix(distances < sym_threshold))
    )


def extract_symmetric_points_idx(symmetric_parts, point_labels):
    """
    Extracts the indices of symmetric points from the given point labels based on the symmetric parts.

    Args:
        symmetric_parts (list): A list of lists where each inner list contains the indices of symmetric parts.
        point_labels (torch.Tensor): A tensor containing the labels of each point.

    Returns:
        list: A list of tensors where each tensor contains the indices of symmetric points.
    """
    symmetric_point_idx = []
    for i, parts in enumerate(symmetric_parts):
        if len(parts) == 1:
            continue  # skip components with only one part
        mask = torch.zeros(len(point_labels), dtype=bool)
        for p in parts:
            mask = mask | (point_labels == p)
        symmetric_point_idx.append(mask.nonzero().flatten())
    return symmetric_point_idx


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


def save_data(partnet_id, point_clouds, points, labels):
    """Saves the given point clouds, points, and labels to disk."""
    os.makedirs(f"{PSPSB_PATH}/{partnet_id}", exist_ok=True)
    torch.save(point_clouds, f"{PSPSB_PATH}/{partnet_id}/point_clouds.pt")
    torch.save(points, f"{PSPSB_PATH}/{partnet_id}/points_4096.pt")
    torch.save(labels, f"{PSPSB_PATH}/{partnet_id}/labels_4096.pt")


def save_gt_data(partnet_id, sym_parts_idx, sym_points_idx, ICP_dist):
    """Saves ground truth data for a given PartNet model."""
    os.makedirs(f"{PSPSB_PATH}/{partnet_id}/ground_truth_approx", exist_ok=True)
    with open(f"{PSPSB_PATH}/{partnet_id}/ground_truth/sym_parts_idx.json", "w") as f:
        json.dump([list(idx) for idx in sym_parts_idx], f)
    with open(f"{PSPSB_PATH}/{partnet_id}/ground_truth/sym_points_idx.json", "w") as f:
        json.dump([list(int(i) for i in idx.numpy()) for idx in sym_points_idx], f)
    with open(f"{PSPSB_PATH}/{partnet_id}/ground_truth/ICP_dist.json", "w") as f:
        json.dump(float(ICP_dist), f)


#######################
### PARSE ARGUMENTS ###
#######################

parser = ArgumentParser()
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
    with open(f"{PARTNET_PATH}/{STATS_DIR}/{category}.{args.split}.json") as f:
        metadata_info = json.load(f)
    anno_ids += [object_info["anno_id"] for object_info in metadata_info]

####################
### PROCESS DATA ###
####################


pbar = tqdm(anno_ids, desc="Model")
for partnet_id in pbar:
    pbar.set_postfix({"partnet_id": partnet_id})

    out_dir = f"{PSPSB_PATH}/{partnet_id}/ground_truth"
    if os.path.exists(out_dir):
        continue  # skip model, if it was already processed
    os.makedirs(out_dir, exist_ok=True)

    # get model category
    metadata = json.load(open(f"{PARTNET_PATH}/{DATA_DIR}/{partnet_id}/meta.json"))
    model_cat = metadata["model_cat"]
    if model_cat not in SYM_THRESHOLDS:
        print(f"Missing symmetry threshold for: {partnet_id} - {model_cat}")
        os.rmdir(out_dir)
        continue

    # load data and sample points
    meshes = load_mesh_data(partnet_id)
    if len(meshes) > 30:
        continue  # skip objects with too many parts
    # BUG: sample points unnormalized, normalize only right before embedding
    point_clouds = sample_point_clouds(meshes)

    # load similarity matrix and extract symmetries
    icp_similarity = load_icp_similarity_matrix(partnet_id)
    sym_parts_idx = extract_symmetric_parts(icp_similarity, SYM_THRESHOLDS[model_cat])

    # extract a point-label cloud
    # BUG: point cloud overlaps multiple times with itself, thus wrong subsampling
    points, labels = subsample_point_clouds(point_clouds)
    # BUG: points are not unidistantly distributed, as subsampling is wrong
    sym_points_idx = extract_symmetric_points_idx(sym_parts_idx, labels)

    # compute ICP-Dist for gt symmetries
    ICP_dist = compute_ICP_dist(point_clouds, sym_parts_idx)

    # save data
    save_data(partnet_id, point_clouds, points, labels)
    save_gt_data(partnet_id, sym_parts_idx, sym_points_idx, ICP_dist)

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
