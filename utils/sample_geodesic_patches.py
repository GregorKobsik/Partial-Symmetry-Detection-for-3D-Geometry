import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
import pytorch3d.ops as ops

from .geodesic_patches import GeodesicPatchSampler

parser = ArgumentParser()
parser.add_argument("-d", "--directory_path", dest="dir_path", default=None, type=str)
parser.add_argument("-n", "--num_patches", dest="num_patches", default=1_000, type=int)
parser.add_argument("-s", "--patch_size", dest="patch_size", default=2**10, type=int)
args = parser.parse_args()

patch_sampler = GeodesicPatchSampler(
    file_path=args.dir_path,
    num_features=2**9,
    initial_pcl_size=2**16,
    patch_sizes=[args.patch_size],
    origin_augmentation=0.1,
)

_, patch_indizes = ops.sample_farthest_points(patch_sampler.points[None, :], K=args.num_patches)

patches_points = torch.tensor([])
patches_normals = torch.tensor([])
patches_normalized = torch.tensor([])

patches_A_points = torch.tensor([])
patches_A_normals = torch.tensor([])
patches_B_points = torch.tensor([])
patches_B_normals = torch.tensor([])

for idx in tqdm(patch_indizes[0], desc="Sample Patches"):
    patches = patch_sampler.sample_augmented_patches(idx, num_augmented_samples=2)

    patches_points = torch.cat([patches_points, patches['original_points']])
    patches_normals = torch.cat([patches_normals, patches['original_normals']])
    patches_normalized = torch.cat([patches_normalized, patches['normalized_points']])

    patches_A_points = torch.cat([patches_A_points, patches['augmented_points'][0:1]])
    patches_A_normals = torch.cat([patches_A_normals, patches['augmented_normals'][0:1]])
    patches_B_points = torch.cat([patches_B_points, patches['augmented_points'][1:2]])
    patches_B_normals = torch.cat([patches_B_normals, patches['augmented_normals'][1:2]])

patch_size_dir = f"/patch_size_{args.patch_size}"
#patch_size_dir = ""
os.makedirs(f"{args.dir_path}{patch_size_dir}", exist_ok=True)
torch.save(patches_points, f"{args.dir_path}{patch_size_dir}/points.pt")
torch.save(patches_normals, f"{args.dir_path}{patch_size_dir}/normals.pt")
torch.save(patches_normalized, f"{args.dir_path}{patch_size_dir}/points_normalized.pt")

torch.save(patches_A_points, f"{args.dir_path}{patch_size_dir}/points_A.pt")
torch.save(patches_A_normals, f"{args.dir_path}{patch_size_dir}/normals_A.pt")
torch.save(patches_B_points, f"{args.dir_path}{patch_size_dir}/points_B.pt")
torch.save(patches_B_normals, f"{args.dir_path}{patch_size_dir}/normals_B.pt")

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
