import os
import json
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
import pytorch3d.io as io
import pytorch3d.ops as ops
from pytorch3d.structures import Meshes, join_meshes_as_scene

from geodesic_patches import GeodesicPatchSampler

parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--data_path",
    dest="data_path",
    default="data/PartNet/data_v0/",
    type=str,
)
parser.add_argument(
    "-o",
    "--out_path",
    dest="out_path",
    default="data/PartNet/patches/",
    type=str,
)
args = parser.parse_args()

print("INFO: script started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

model_dirs = Path(args.data_path).glob("*/")

for subdir in tqdm(model_dirs, desc="OBJ-file"):
    # skip directory, if it was already sampled
    out_dir = f"{args.out_path}/{subdir.parts[-1]}"
    if os.path.exists(out_dir):
        continue  # skip already sampled shapes
    os.makedirs(out_dir, exist_ok=True)

    # load meshes and join them to a single one
    dirs = Path(subdir).glob(f"objs/*.obj")
    meshes = []
    for model_dir in dirs:
        obj_model = io.load_obj(model_dir, load_textures=False)
        meshes += [Meshes(verts=[obj_model[0]], faces=[obj_model[1][0]])]
    scene = join_meshes_as_scene(meshes)

    # create patch sampler
    patch_sampler = GeodesicPatchSampler(
        mesh_object=scene,
        num_features=2**9,
        initial_pcl_size=2**16,
        origin_augmentation=0.1,
    )

    for patch_size, num_samples in [
        (512, 128),
        (1024, 64),
        (2048, 32),
        (4096, 16),
        (8192, 8),
    ]:
        patch_sampler.patch_sizes = [patch_size]
        _, patch_indizes = ops.sample_farthest_points(
            patch_sampler.points[None, :], K=num_samples
        )

        patches = torch.tensor([])
        for idx in patch_indizes[0]:
            p_out = patch_sampler.sample_augmented_patches(idx, num_augmented_samples=2)
            patches = torch.cat([patches, p_out["augmented_points"][None, :]])

        torch.save(patches, f"{out_dir}/patches_{patch_size}.pt")

print("INFO: script finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
