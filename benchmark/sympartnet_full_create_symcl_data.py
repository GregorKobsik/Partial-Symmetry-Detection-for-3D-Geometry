from argparse import ArgumentParser
from datetime import datetime
import json
import os
import numpy as np
from tqdm.auto import tqdm

from sym_cl import (
    sample_geodesic_patches,
    embed_features,
    compute_clusters,
    subdivide_clusters,
    filter_symmetry_hypotheses,
)

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

EPSB_PATH = "data/PartNet/extrinsic_partial_symmetry_benchmark/data_v0"


def load_point_cloud(partnet_id):
    """Load point cloud from json file."""
    point_cloud_path = f"{EPSB_PATH}/{partnet_id}/point_cloud.json"
    try:
        with open(point_cloud_path, "r") as f:
            return np.array(json.load(f), dtype=np.float32)
    except:
        return None


def save_data(partnet_id, slurm_id, clusters_components_idx, clusters_icp_distance):
    """Save data to json file."""
    os.makedirs(f"{EPSB_PATH}/{partnet_id}/SymCL_{slurm_id}", exist_ok=True)
    with open(
        f"{EPSB_PATH}/{partnet_id}/SymCL_{slurm_id}/symmetry_icp_dist.json", "w"
    ) as f:
        json.dump([float(dist) for dist in clusters_icp_distance], f)
    for n, cluster_idx in enumerate(clusters_components_idx):
        with open(
            f"{EPSB_PATH}/{partnet_id}/SymCL_{slurm_id}/symmetry_idx_{n}.json", "w"
        ) as f:
            json.dump([int(item) for row in cluster_idx for item in row], f)


PARTNET_PATH = "data/PartNet/"
SYMMETRY_PATH = "data/PartNet/icp_part_symmetry_matrix/"
STATS_DIR = "metadata/stats/train_val_test_split"
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

#######################
### PARSE ARGUMENTS ###
#######################

parser = ArgumentParser()
parser.add_argument("-id", "--model_id", dest="model_id", default=None, type=str)
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
    out_dir = f"{EPSB_PATH}/{partnet_id}/SymCL_{args.model_id}"
    if os.path.exists(out_dir):
        continue  # skip model, if it was already processed
    os.makedirs(out_dir, exist_ok=True)

    # load point cloud
    point_cloud = load_point_cloud(partnet_id)
    if point_cloud is None:
        continue  # skip shapes without valid point cloud data (e.g. with more than parts)

    # extract symmetries
    patches_idx, patches_origin_dist, patches_origin_idx = sample_geodesic_patches(
        point_cloud, num_patches=1000, patch_size=[512, 1024, 2048, 4096, 8192]
    )
    features, distances = embed_features(point_cloud, patches_idx, args.model_id)
    clusters_patch_idx, clusters_features_var = compute_clusters(
        distances, features, min_cluster_size=20
    )
    clusters_merged_patches_idx = subdivide_clusters(
        point_cloud, patches_idx, clusters_patch_idx, patches_origin_dist
    )
    clusters_components_idx, clusters_icp_distance = filter_symmetry_hypotheses(
        point_cloud, clusters_merged_patches_idx, symmetry_threshold=0.005
    )

    # save data
    save_data(partnet_id, args.model_id, clusters_components_idx, clusters_icp_distance)

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
