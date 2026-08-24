from argparse import ArgumentParser
import os
import numpy as np
import torch
import pytorch3d.io as io
import pytorch3d.ops as ops
from pytorch3d.structures import Meshes
from utils.point_cloud_utils import normalize_point_cloud

from sym_cl import sample_geodesic_patches

parser = ArgumentParser()
parser.add_argument("-s", "--shape_name", dest="shape_name", default="filigree", type=str)
parser.add_argument("-n", "--num_patches", dest="num_patches", default=10, type=int)
parser.add_argument("-np", "--num_points", dest="num_points", default=2**16, type=int)
args = parser.parse_args()

shape_name = args.shape_name
num_patches = args.num_patches
patch_size = [512, 1024, 2048, 4096, 8192]

# sample patches
print("Shape Name:", args.shape_name)
print("### Sample Patches:", num_patches, flush=True)

# load .obj mesh file
file_path = f"data/meshes/{args.shape_name}.obj"
shape_obj = io.load_obj(file_path, load_textures=False)
verts, faces = shape_obj[0], shape_obj[1][0]

verts = normalize_point_cloud(torch.tensor(verts, dtype=torch.float32))
mesh_obj = Meshes(verts=[verts], faces=[faces])

###############################
### Sample Geodesic Patches ###
###############################

points = ops.sample_points_from_meshes(mesh_obj, args.num_points)[0].detach().clone()
patches_idx, patches_origin_dist, patches_origin_idx = sample_geodesic_patches(points, num_patches=num_patches, silent=False)

# save patches
os.makedirs(f"data/precomputed/{args.shape_name}", exist_ok=True)
np.save(f"data/precomputed/{args.shape_name}/pcl.npy", points)
np.save(f"data/precomputed/{args.shape_name}/patches_idx.npy", patches_idx)
np.save(f"data/precomputed/{args.shape_name}/patches_origin_dist.npy", patches_origin_dist)
np.save(f"data/precomputed/{args.shape_name}/patches_origin_idx.npy", patches_origin_idx)


