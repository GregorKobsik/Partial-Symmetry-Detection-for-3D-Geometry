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

PARTNET_PATH = "data/PartNet/"
SYMMETRY_PATH = "data/PartNet/icp_part_symmetry_matrix/"
EPSB_PATH = "data/PartNet/extrinsic_partial_symmetry_benchmark/data_v0/"
STATS_DIR = "metadata/stats/train_val_test_split"
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


def load_meshes(mesh_dir_path):
    mesh_paths = sorted(glob(mesh_dir_path + "/*.obj"))
    return io.load_objs_as_meshes(mesh_paths, load_textures=False)


def sample_points(meshes, num_samples=10_000):
    """Sample points from meshes in mesh_dir_path and return point cloud and point labels.
    Args:
        meshes: input mesh data object
        num_samples: number of sampled points
    """
    point_clouds = ops.sample_points_from_meshes(meshes, num_samples=num_samples)
    point_labels = torch.tensor([num_samples * [i] for i, _ in enumerate(point_clouds)])

    point_cloud = point_clouds.reshape(1, -1, 3)
    point_label = point_labels.reshape(1, -1)

    sampled_pcl, sampled_idx = ops.sample_farthest_points(point_cloud, K=num_samples)
    return sampled_pcl[0], point_label[0][sampled_idx[0]]


def extract_symmetric_parts(point_labels, symmetry_matrix_path, sym_threshold):
    """Extract symmetric parts from point cloud using symmetry matrix.
    Args:
        point_labels: point labels
        symmetry_matrix_path: path to symmetry matrix
        sym_threshold: threshold for symmetry matrix
    """
    similarity_matrix = torch.load(symmetry_matrix_path)
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    symmetric_parts = list(
        nx.connected_components(
            nx.from_numpy_matrix(similarity_matrix.numpy() < sym_threshold)
        )
    )

    symmetric_point_idx = []
    for i, parts in enumerate(symmetric_parts):
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
    os.makedirs(f"{EPSB_PATH}/{partnet_id}/ground_truth", exist_ok=True)
    torch.save(points, f"{EPSB_PATH}/{partnet_id}/points_65536.pt")
    torch.save(labels, f"{EPSB_PATH}/{partnet_id}/labels_65536.pt")
    for i, idx in enumerate(symmetric_points_idx):
        torch.save(idx, f"{EPSB_PATH}/{partnet_id}/ground_truth/symmetry_idx_{i}.pt")


# def save_data(out_dir_path, point_cloud, symmetric_points_idx):
#     """Save point cloud and symmetric parts to out_dir_path.
#     Args:
#         out_dir_path: path to output directory
#         point_cloud: point cloud
#         symmetric_points_idx: list of symmetric parts indices
#     """
#     os.makedirs(f"{out_dir_path}/ground_truth_approx", exist_ok=True)
#     with open(f"{out_dir_path}/point_cloud.json", "w") as f:
#         json.dump(point_cloud.tolist(), f)
#     for i, idx in enumerate(symmetric_points_idx):
#         with open(f"{out_dir_path}/ground_truth_approx/symmetry_idx_{i}.json", "w") as f:
#             json.dump(idx.tolist(), f)


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
model_dirs = []
for category in args.categories:
    with open(f"{PARTNET_PATH}/{STATS_DIR}/{category}.{args.split}.json") as f:
        model_anno_ids = json.load(f)

    for model in model_anno_ids:
        anno_id = model["anno_id"]
        model_dirs += [f"{PARTNET_PATH}/{DATA_DIR}/{anno_id}/"]

####################
### PROCESS DATA ###
####################

for subdir in tqdm(model_dirs, desc="Model"):
    partnet_id = subdir.split("/")[-2]

    out_dir = f"{EPSB_PATH}/{partnet_id}/ground_truth_approx"
    if os.path.exists(out_dir):
        continue  # skip directory, if it was already sampled
    os.makedirs(out_dir, exist_ok=True)

    # get model category
    metadata = json.load(open(subdir + "meta.json"))
    model_cat = metadata["model_cat"]
    if model_cat not in SYM_THRESHOLDS:
        print(f"Missing symmetry threshold for: {partnet_id} - {model_cat}")
        os.rmdir(out_dir)
        continue

    meshes = load_meshes(subdir + "objs")

    # skip objects with too many parts
    if len(meshes) > 30:
        continue

    # sample points
    point_cloud, point_labels = sample_points(meshes, num_samples=args.num_samples)

    # extract symmetric parts
    symmetry_matrix_path = f"{SYMMETRY_PATH}/{partnet_id}/icp_distances.pt"
    symmetric_points_idx = extract_symmetric_parts(
        point_labels, symmetry_matrix_path, SYM_THRESHOLDS[model_cat]
    )

    # save data
    save_data(out_dir, point_cloud, symmetric_points_idx)

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
