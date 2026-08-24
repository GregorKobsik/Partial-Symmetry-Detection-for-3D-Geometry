###############
### Imports ###
###############

import os
from argparse import ArgumentParser
from datetime import datetime
import pickle
import numpy as np

from utils.geodesic_utils import fix_geodesic_distance_for_disconnected_components

from sym_cl import compute_clusters, subdivide_clusters, filter_symmetry_hypotheses

##################
### Load Files ###
##################

# input arguments
parser = ArgumentParser()
parser.add_argument("-s", "--shape_name", dest="shape_name", default="filigree", type=str)
parser.add_argument("-clr", "--use_clr", dest="use_clr", default=False, type=bool)
parser.add_argument(
    "-st", "--symmetry_threshold", dest="symmetry_threshold", default=0.005, type=float
)
args = parser.parse_args()

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("arguments:", args)

# load precomputed data
dir_path = f"data/precomputed/{args.shape_name}"
os.makedirs(dir_path, exist_ok=True)
prefix = "clr_" if args.use_clr else "icp_"

point_cloud = np.load(f"{dir_path}/pcl.npy")
patches_idx = np.load(f"{dir_path}/patches_idx.npy")
patches_origin_dist = np.load(f"{dir_path}/patches_origin_dist.npy")
patches_origin_dist = fix_geodesic_distance_for_disconnected_components(patches_origin_dist)

features = np.load(f"{dir_path}/clr_features.npy")
distances = np.load(f"{dir_path}/{prefix}distances.npy")

#########################
### Cluster Distances ###
#########################

clusters_patch_idx, clusters_features_var = compute_clusters(distances, features, min_cluster_size=20)

##########################
### Extract Symmetries ###
##########################

clusters_merged_patches_idx = subdivide_clusters(
    point_cloud, patches_idx, clusters_patch_idx, patches_origin_dist, silent=False)

# Run the ICP filter once with an infinite threshold to keep EVERY candidate
# cluster + its max intra-cluster ICP distance. The thresholded survivors are
# then just the subset with icp <= symmetry_threshold (no second clustering /
# ICP pass needed -- this used to be a separate precompute_cluster_icp.py run).
all_components_idx, all_icp_distance = filter_symmetry_hypotheses(
    point_cloud, clusters_merged_patches_idx, np.inf, silent=False)
all_icp_distance = [float(x) for x in all_icp_distance]

clusters_components_idx = [
    c for c, d in zip(all_components_idx, all_icp_distance) if d <= args.symmetry_threshold]
clusters_icp_distance = [
    d for d in all_icp_distance if d <= args.symmetry_threshold]
print(f"Survivors at threshold {args.symmetry_threshold}: "
      f"{len(clusters_components_idx)}/{len(all_components_idx)} candidates")


#################
### Save Data ###
#################

def save_pickle(file_path, var):
    """Convinience function to quicker saving of vars with pickle."""
    with open(file_path, "wb") as file:
        pickle.dump(var, file)


save_pickle(f"{dir_path}/{prefix}clusters_components_idx.pkl", clusters_components_idx)
save_pickle(f"{dir_path}/{prefix}clusters_icp_distance.pkl", clusters_icp_distance)
# unfiltered candidates for the explorer notebook's live threshold slider
save_pickle(f"{dir_path}/{prefix}clusters_all_components_idx.pkl", all_components_idx)
save_pickle(f"{dir_path}/{prefix}clusters_all_icp.pkl", all_icp_distance)

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
