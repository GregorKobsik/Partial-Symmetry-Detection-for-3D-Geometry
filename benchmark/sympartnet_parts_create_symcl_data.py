from argparse import ArgumentParser
from datetime import datetime
from tqdm.auto import tqdm
import os
import torch
import numpy as np
import json
from glob import glob
import networkx as nx

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from scipy.spatial.distance import cdist

from utils.point_cloud_utils import normalize_point_cloud
from utils.icp_distance import icp_distance

from siamese_models import ContrastiveSimilarityModel

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

PARTNET_PATH = "data/PartNet"
STATS_DIR = "metadata/stats/train_val_test_split"
DATA_DIR = "data_v0"
PSPSB_PATH = "data/PartNet/pre_segmented_partial_symmetry_benchmark/data_v0"
SPLITS = {"train": 0, "val": 1, "test": 2}
SYM_THRESHOLDS = {
    "37809": 0.025,  # SymCL, released checkpoint (epoch 999)
    "37810": 0.025,  # CLR-Features, 2000 epochs
    "39938": 0.025,  # SymCL 500 epochs
    "39939": 0.025,  # SymCL 500 epochs, no origin augment
    "39940": 0.025,  # SymCL 500 epochs, euclidean patches
    "40358": 0.025,  # SymCL 500 epochs, no reflections
    "40357": 0.025,  # SymCL 500 epochs, no scaling
    "40337": 0.025,  # SymCL 500 epochs, one size only
    "38464": 0.00025,  # ICP-Finetuning, 2000 + 20 epochs
    "38470": 0.00025,  # ICP-only, 20 epochs
    "38575": 0.00025,  # ICP-only, 200 epochs
    "39927": 0.00025,  # SymML, 50 epochs
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


def load_mesh_data(model_id):
    paths = sorted(glob(f"{PARTNET_PATH}/{DATA_DIR}/{model_id}/objs/*.obj"))
    return load_objs_as_meshes(paths, load_textures=False)


def load_point_cloud(model_id):
    points = torch.load(f"{PSPSB_PATH}/{model_id}/points_4096.pt")
    labels = torch.load(f"{PSPSB_PATH}/{model_id}/labels_4096.pt")
    return points, labels


def save_pred_data(partnet_id, model_id, sym_parts_idx, sym_points_idx, ICP_dist):
    """Saves ground truth data for a given PartNet model."""
    os.makedirs(f"{PSPSB_PATH}/{partnet_id}/SymCL_{model_id}", exist_ok=True)
    with open(
        f"{PSPSB_PATH}/{partnet_id}/SymCL_{model_id}/sym_parts_idx.json", "w"
    ) as f:
        json.dump([list(idx) for idx in sym_parts_idx], f)
    with open(
        f"{PSPSB_PATH}/{partnet_id}/SymCL_{model_id}/sym_points_idx.json", "w"
    ) as f:
        json.dump([list(int(i) for i in idx.numpy()) for idx in sym_points_idx], f)
    with open(f"{PSPSB_PATH}/{partnet_id}/SymCL_{model_id}/ICP_dist.json", "w") as f:
        json.dump(float(ICP_dist), f)


def sample_point_clouds(meshes):
    point_clouds = sample_points_from_meshes(meshes, num_samples=4096)
    point_clouds, _ = sample_farthest_points(point_clouds, K=512)
    return torch.stack([normalize_point_cloud(pcl) for pcl in point_clouds])


def load_model(model_id):
    # load model
    run_path = "logs/" + f"run_{model_id}"
    ckpt_paths = f"{run_path}/checkpoints/*.ckpt"
    ckpt_path = sorted(glob(ckpt_paths))[0]
    return ContrastiveSimilarityModel.load_from_checkpoint(ckpt_path).cuda().eval()


def embed_features(point_clouds, model):
    # embed features
    features = []
    with torch.no_grad():
        for pcl in point_clouds:
            features += model(pcl.to("cuda")[None])
    features = torch.stack(features).cpu().detach().numpy()

    # compute cosine similarity
    features_dist = cdist(features, features, metric="cosine")

    return features, features_dist


def extract_symmetric_parts(distances, sym_threshold=0.1):
    return list(
        nx.connected_components(nx.from_numpy_matrix(distances < sym_threshold))
    )


def extract_symmetric_point_idx(symmetric_parts, point_labels):
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

model = load_model(args.model_id)

pbar = tqdm(anno_ids, desc="Model")
for partnet_id in pbar:
    pbar.set_postfix({"partnet_id": partnet_id})

    out_dir = f"{PSPSB_PATH}/{partnet_id}/SymCL_{args.model_id}"
    if os.path.exists(out_dir):
        continue  # skip model, if it was already processed
    os.makedirs(out_dir, exist_ok=True)

    # load mesh and sample point clouds
    meshes = load_mesh_data(partnet_id)
    if len(meshes) > 30:
        continue  # skip models with too many parts
    point_clouds = sample_point_clouds(meshes)

    # embed features into latent space and symmetries based on distances between the patches
    features, distances = embed_features(point_clouds, model)
    pred_symmetric_parts_idx = extract_symmetric_parts(
        distances, SYM_THRESHOLDS[args.model_id]
    )

    # extract symmetry idx from symmetric parts
    _, labels = load_point_cloud(partnet_id)
    prediction_idx = extract_symmetric_point_idx(pred_symmetric_parts_idx, labels)
    ICP_dist_pred = compute_ICP_dist(point_clouds, pred_symmetric_parts_idx)

    # save predicted data
    save_pred_data(
        partnet_id,
        args.model_id,
        pred_symmetric_parts_idx,
        prediction_idx,
        ICP_dist_pred,
    )


print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
