from argparse import ArgumentParser
from datetime import datetime
from tqdm.auto import tqdm
import os
import json

from glob import glob
import networkx as nx

import torch
import pytorch3d.io as io
import pytorch3d.ops as ops


print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

PARTNET_PATH = "data/PartNet/data_v0"
STATS_PATH = "data/PartNet/metadata/stats/train_val_test_split"
SYMMETRY_PATH = "data/PartNet/icp_part_symmetry_matrix/"
EPSB_PATH = "data/PartNet/extrinsic_partial_symmetry_benchmark/data_v0/"
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


def load_meshes(partnet_id):
    """Load meshes from directory."""
    try:
        mesh_paths = sorted(glob(f"{PARTNET_PATH}/{partnet_id}/objs/*.obj"))
        return io.load_objs_as_meshes(mesh_paths, load_textures=False)
    except:
        return None


def load_point_cloud(partnet_id):
    """Load point cloud from json file."""
    try:
        with open(f"{EPSB_PATH}/{partnet_id}/point_cloud.json", "r") as f:
            return torch.tensor(json.load(f))
    except:
        return None


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


def sample_points(meshes, num_points):
    """Sample points from meshes."""
    points = ops.sample_points_from_meshes(meshes, num_samples=2 * num_points)
    return ops.sample_farthest_points(points, K=num_points)[0]


def extract_point_labels(points_per_mesh, point_cloud):
    """Extract point labels from point cloud."""
    n_points_per_mesh = points_per_mesh.shape[1]
    labels_lookup = torch.concat(
        [torch.ones(n_points_per_mesh) * i for i in range(len(meshes))]
    )
    dist = torch.cdist(points_per_mesh.reshape(-1, 3), point_cloud)
    _, idx = torch.min(dist, dim=0)
    return labels_lookup[idx]


def extract_symmetric_parts(point_labels, sim_matrix, sym_threshold):
    """Extract symmetric parts from point cloud using symmetry matrix.
    Args:
        point_labels: point labels
        sim_matrix: similarity matrix
        sym_threshold: threshold for similarity matrix
    """
    sym_matrix = sim_matrix < sym_threshold
    symmetric_parts = list(
        nx.connected_components(nx.from_numpy_matrix(sym_matrix.numpy()))
    )

    symmetric_point_idx = []
    for parts in symmetric_parts:
        if len(parts) == 1:
            continue  # skip components with only one part
        mask = torch.zeros(len(point_labels), dtype=bool)
        for p in parts:
            mask = mask | (point_labels == p)
        symmetric_point_idx.append(mask.nonzero().flatten())
    return symmetric_point_idx


def save_data(partnet_id, points, labels, symmetric_points_idx):
    """Save point cloud and symmetric parts to out_dir_path.
    Args:
        out_dir_path: path to output directory
        symmetric_points_idx: list of symmetric parts indices
    """
    os.makedirs(f"{EPSB_PATH}/{partnet_id}/ground_truth_approx", exist_ok=True)
    torch.save(points, f"{EPSB_PATH}/{partnet_id}/points_65536.pt")
    torch.save(labels, f"{EPSB_PATH}/{partnet_id}/labels_65536.pt")
    for i, idx in enumerate(symmetric_points_idx):
        torch.save(
            idx, f"{EPSB_PATH}/{partnet_id}/ground_truth_approx/symmetry_idx_{i}.pt"
        )


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
    out_dir = f"{EPSB_PATH}/{partnet_id}/ground_truth_approx"
    if os.path.exists(out_dir):
        continue  # skip model, if it was already processed
    os.makedirs(out_dir, exist_ok=True)

    meshes = load_meshes(partnet_id)
    # skip objects with too many parts
    if meshes is None or len(meshes) > 30:
        print(f"INFO: PartNet ID: {partnet_id}, Meshes: {meshes}")
        continue

    # load point clouds and extract point labels
    points = load_point_cloud(partnet_id)
    if points is None:
        print(f"INFO: PartNet ID: {partnet_id}, Points: {points}")
        continue
    points_per_mesh = sample_points(meshes, 2048)
    labels = extract_point_labels(points_per_mesh, points)

    # load symmetry data and extract symmetric parts
    sym_threshold = load_symmmetry_threshold(partnet_id)
    sim_matrix = load_similarity_matrix(partnet_id)
    if sym_threshold is None or sim_matrix is None:
        print(
            f"INFO: PartNet ID: {partnet_id}, Threshold: {sym_threshold}, Matrix: {sim_matrix}"
        )
        continue
    symmetric_points_idx = extract_symmetric_parts(labels, sim_matrix, sym_threshold)

    # save data
    save_data(partnet_id, points, labels, symmetric_points_idx)

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
