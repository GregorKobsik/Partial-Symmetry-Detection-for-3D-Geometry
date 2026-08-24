import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
from utils.icp_point_to_plane import icp_distance_o3d, icp_distance_p3d


parser = ArgumentParser()
parser.add_argument("-d", "--directory_path", dest="dir_path", default=None, type=str)
method_choices = ["point2point", "point2plane"]
parser.add_argument("-m", "--method", dest="method", default="point2point", type=str, choices=method_choices)
framework_choices = ["pytorch3d", "open3d"]
parser.add_argument("-f", "--framework", dest="framework", default="pytorch3d", type=str, choices=framework_choices)
args = parser.parse_args()

print("path", args.dir_path, flush=True)

pnts = torch.load(f"{args.dir_path}/points_normalized.pt")
# pnts = torch.load(f"{args.dir_path}/points_B.pt")
nrmls = torch.load(f"{args.dir_path}/normals.pt") if args.method == "point2normals" else None

if args.framework == "pytorch3d":
    icp_distances = icp_distance_p3d(pnts, nrmls)
else:
    icp_distances = icp_distance_o3d(pnts, nrmls)

print("Computed ICP distances:", icp_distances.shape)
output_dir = f"{args.dir_path}/{args.method}_{args.framework}"
os.makedirs(output_dir, exist_ok=True)
torch.save(icp_distances, f"{output_dir}/icp_distances.pt")
# torch.save(icp_distances, f"{output_dir}/icp_distances_B.pt")
print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
