from argparse import ArgumentParser
import numpy as np
import torch
from utils.icp_distance import icp_distance

parser = ArgumentParser()
parser.add_argument("-s", "--shape_name", dest="shape_name", default="filigree", type=str)
args = parser.parse_args()
data_dir = f"data/precomputed/{args.shape_name}"

# load data
pcl = np.load(f"{data_dir}/pcl.npy")
patches_idx = np.load(f"{data_dir}/patches_idx.npy")

# compute pairwise ICP distances
print("### Compute ICP Distances", flush=True)

patches = torch.tensor(pcl[patches_idx], dtype=torch.float32)
icp_distances = icp_distance(patches).numpy()

# save distances
np.save(f"{data_dir}/icp_distances.npy", icp_distances)