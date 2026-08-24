import os
import json
from glob import glob
from datetime import datetime
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
import icp_distance
import pytorch3d.io as io
import pytorch3d.ops as ops
from pytorch3d.structures import Meshes

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

PARTNET_PATH = "data/PartNet"
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


def normalize_point_cloud(pcl: torch.Tensor) -> torch.Tensor:
    """Normalize the point cloud to a unit sphere centered at 0."""
    pcl = pcl.detach().clone()
    pcl -= torch.mean(pcl, dim=0)[None]
    pcl /= (
        torch.cdist(pcl, torch.zeros(1, 3, device=pcl.device)).max(dim=0).values[None]
    )
    return pcl


def load_meshes(mesh_dir_path):
    mesh_paths = sorted(glob(mesh_dir_path + "/*.obj"))
    return io.load_objs_as_meshes(mesh_paths, load_textures=False)


def sample_point_clouds(meshes: Meshes, K: int):
    """Samples normalized point clouds from meshes."""
    point_clouds = ops.sample_points_from_meshes(meshes, num_samples=4 * K)
    point_clouds, _ = ops.sample_farthest_points(point_clouds, K=K)
    return torch.stack([normalize_point_cloud(pcl) for pcl in point_clouds])


parser = ArgumentParser()
parser.add_argument(
    "-in", "--input_dir", dest="input_dir", default="data/PartNet/data_v0/"
)
parser.add_argument(
    "-out",
    "--output_dir",
    dest="output_dir",
    default="data/PartNet/icp_part_symmetry_matrix/",
)
parser.add_argument("-n", "--num_points", dest="num_points", default=1024, type=int)
parser.add_argument("-c", "--categories", dest="categories", default="Vase", type=str)
parser.add_argument("-s", "--split", dest="split", default="train", type=str)
args = parser.parse_args()

# retrive categories which we will process
if args.categories in (None, "all", "All", ["all"], ["All"]):
    args.categories = list(CATEGORIES)
if isinstance(args.categories, str):
    args.categories = [args.categories]
assert all(category in CATEGORIES for category in args.categories)
assert (
    args.split in SPLITS
), f"Split {args.split} found, but expected either train, val or test"


# compose list of model directories
anno_ids = []
for category in args.categories:
    with open(f"{PARTNET_PATH}/{STATS_DIR}/{category}.{args.split}.json") as f:
        metadata_info = json.load(f)
    anno_ids += [object_info["anno_id"] for object_info in metadata_info]

# get all model dirs in PartNet dataset
child_dirs = os.listdir(path=args.input_dir)

pbar = tqdm(anno_ids, desc="Model")
for partnet_id in pbar:
    pbar.set_postfix({"partnet_id": partnet_id})

    # check if distances are already computed and create output dir, otherwise skip model
    out_dirname = f"{args.output_dir}/{partnet_id}"
    if os.path.exists(out_dirname):
        continue  # skip computation, hack for multi-processing
    os.makedirs(out_dirname, exist_ok=True)

    # load part models and sample random pointclouds
    meshes = load_meshes(f"{args.input_dir}/{partnet_id}/objs/")
    # if len(meshes) > 30:
    #    continue  # skip objects with too many parts

    # sample point clouds
    parts_pcl = sample_point_clouds(meshes, K=args.num_points)

    # compute ICP distances (Chamfer distance, after ICP registration) and save them
    icp_distances = icp_distance.icp_distance(parts_pcl)
    torch.save(icp_distances, f"{out_dirname}/icp_distances.pt")

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
