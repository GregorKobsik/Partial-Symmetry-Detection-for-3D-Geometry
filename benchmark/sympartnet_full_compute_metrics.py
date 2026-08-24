from argparse import ArgumentParser
from datetime import datetime
from tqdm.auto import tqdm
import os
import json
from glob import glob

import torch
import numpy as np

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

PARTNET_PATH = "data/PartNet/"
STATS_DIR = "metadata/stats/train_val_test_split"
EPSB_PATH = "data/PartNet/extrinsic_partial_symmetry_benchmark/data_v0"
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


def load_point_cloud(partnet_id):
    """Load point cloud from json file."""
    with open(f"{EPSB_PATH}/{partnet_id}/point_cloud.json", "r") as f:
        return np.array(json.load(f), dtype=np.float32)


def load_symmetry_data(partnet_id, subdir):
    """Load symmetry data for a given PartNet object."""
    data_paths = glob(f"{EPSB_PATH}/{partnet_id}/{subdir}/symmetry_idx_*.json")
    symmetry_idx = []
    for path in data_paths:
        with open(path, "r") as f:
            symmetry_idx += [json.load(f)]
    return symmetry_idx


def load_symmetry_data_gt(partnet_id, subdir):
    """Load symmetry data for a given PartNet object."""
    data_paths = glob(f"{EPSB_PATH}/{partnet_id}/{subdir}/symmetry_idx_*.pt")
    symmetry_idx = []
    for path in data_paths:
        symmetry_idx += [torch.load(path).numpy()]
    return symmetry_idx


def load_sym_icp_dist(partnet_id, subdir):
    """Load ICP-Distance values for given PartNet object."""
    with open(f"{EPSB_PATH}/{partnet_id}/{subdir}/symmetry_icp_dist.json", "r") as f:
        icp_dist = json.load(f)
    return icp_dist


def compute_IoU(prediction, ground_truth, return_COV=False):
    IoU_matrix = np.ndarray([len(prediction), len(ground_truth)])
    for i, pred_idx in enumerate(prediction):
        for j, gt_idx in enumerate(ground_truth):
            i_set, j_set = set(pred_idx), set(gt_idx)
            IoU_matrix[i, j] = len(i_set.intersection(j_set)) / len(i_set.union(j_set))
    if return_COV:
        coverage = len(np.unique(np.argmax(IoU_matrix, axis=0))) / len(ground_truth)
        return np.max(IoU_matrix, axis=1), coverage
    return np.max(IoU_matrix, axis=1)


def save_data(partnet_id, subdir, icp_dist, IoU, cov):
    metrics = {
        "Mean ICP-Dist": np.mean(icp_dist),
        "Mean IoU": np.mean(IoU),
        "COV-IoU": cov,
    }
    with open(f"{EPSB_PATH}/{partnet_id}/{subdir}/metrics.json", "w") as f:
        json.dump(metrics, f)


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

    subdir = f"SymCL_{args.model_id}"
    out_file = f"{EPSB_PATH}/{partnet_id}/{subdir}/metrics.json"
    if os.path.exists(out_file):
        os.remove(out_file)
    #    continue  # skip model, if it was already processed

    # load precomputed symmetries
    pred_symmetry_idx = load_symmetry_data(partnet_id, subdir)
    gt_symmetry_idx = load_symmetry_data_gt(partnet_id, "ground_truth_approx")

    if len(pred_symmetry_idx) == 0 or len(gt_symmetry_idx) == 0:
        continue  # skip model, if no symmetries detected

    # load precomputed icp distances
    icp_dists = load_sym_icp_dist(partnet_id, subdir)

    # compute Intersection over Union (IoU) and coverage (COV)
    IoU, cov = compute_IoU(pred_symmetry_idx, gt_symmetry_idx, return_COV=True)

    # save data
    save_data(partnet_id, subdir, icp_dists, IoU, cov)

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
