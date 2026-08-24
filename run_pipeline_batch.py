"""
Batch Symmetry Detection Pipeline

Runs the complete symmetry detection pipeline for all shapes in a directory:
1. Sample point cloud and geodesic patches
2. Embed features using SymCL model
3. Extract symmetry clusters
4. Region growing optimization

Usage:
    python run_pipeline_batch.py --input_dir data/meshes --output_dir data/precomputed
"""

from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
from tqdm.auto import tqdm
import json
import pickle

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from pytorch3d.structures import Meshes

import potpourri3d as pp3d

from sym_cl import (
    sample_geodesic_patches,
    embed_features,
    compute_clusters,
    subdivide_clusters,
    filter_symmetry_hypotheses,
)
from utils.icp_distance import icp_distance
from utils.point_cloud_utils import normalize_point_cloud

print("INFO: Batch pipeline started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)


def validate_model_checkpoint(model_id):
    """Validate that the model checkpoint exists."""
    from glob import glob
    run_path = f"logs/run_{model_id}"
    ckpt_paths = f"{run_path}/*.ckpt"
    ckpts = glob(ckpt_paths)

    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint found for model {model_id} at {ckpt_paths}\n"
            f"Available model directories: {os.listdir('logs') if os.path.exists('logs') else 'logs/ not found'}"
        )

    return sorted(ckpts)[0]


def normalize_mesh(mesh):
    """Normalize mesh to unit sphere."""
    verts, faces = mesh.verts_packed(), mesh.faces_packed()
    verts = normalize_point_cloud(verts)
    return Meshes(verts=[verts], faces=[faces])


def load_pickle(file_path):
    """Load variable from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(file_path, var):
    """Save variable to pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(var, f)


def process_shape(mesh_path, output_dir, model_id=39938, skip_existing=True):
    """
    Process a single shape through the complete pipeline.

    Args:
        mesh_path: Path to input mesh (.obj file)
        output_dir: Directory to save results
        model_id: Model ID for feature embedding (39938 for SymCL)
        skip_existing: Skip if results already exist

    Returns:
        True if successful, False otherwise
    """
    shape_name = Path(mesh_path).stem
    shape_output_dir = os.path.join(output_dir, shape_name)
    os.makedirs(shape_output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing: {shape_name}")
    print(f"{'='*70}")

    # Check if already processed
    if skip_existing and os.path.exists(f"{shape_output_dir}/clr_region_growing_components_idx.pkl"):
        print(f"✓ Results already exist for {shape_name}, skipping...")
        return True

    try:
        # ====================================================================
        # Step 1: Load mesh and sample point cloud
        # ====================================================================
        print("\n[1/4] Sampling point cloud and geodesic patches...")

        if os.path.exists(f"{shape_output_dir}/pcl.npy"):
            print("  Loading existing point cloud...")
            pcl = np.load(f"{shape_output_dir}/pcl.npy")
            patches_idx = np.load(f"{shape_output_dir}/patches_idx.npy")
            patches_origin_dist = np.load(f"{shape_output_dir}/patches_origin_dist.npy")
        else:
            # Load and normalize mesh
            mesh = load_objs_as_meshes([mesh_path], load_textures=False)
            mesh = normalize_mesh(mesh)

            # Save normalized mesh
            save_obj(
                f"{shape_output_dir}/model_normalized.obj",
                mesh.verts_packed(),
                mesh.faces_packed()
            )

            # Sample point cloud
            pcl = sample_points_from_meshes(mesh, num_samples=2**16)[0].numpy()

            # Sample geodesic patches
            patches_idx, patches_origin_dist, patches_origin_idx = sample_geodesic_patches(
                pcl,
                num_patches=1000,
                patch_size=[512, 1024, 2048, 4096, 8192],
                silent=False,
            )

            # Save
            np.save(f"{shape_output_dir}/pcl.npy", pcl)
            np.save(f"{shape_output_dir}/patches_idx.npy", patches_idx)
            np.save(f"{shape_output_dir}/patches_origin_dist.npy", patches_origin_dist.astype(np.float16))
            np.save(f"{shape_output_dir}/patches_origin_idx.npy", patches_origin_idx)

        print(f"  Point cloud: {len(pcl)} points")
        print(f"  Patches: {len(patches_idx)}")

        # ====================================================================
        # Step 2: Embed features using SymCL
        # ====================================================================
        print("\n[2/4] Embedding features with SymCL...")

        if os.path.exists(f"{shape_output_dir}/clr_patches_distances.npy"):
            print("  Loading existing distances...")
            distances = np.load(f"{shape_output_dir}/clr_patches_distances.npy")
        else:
            # Embed patches
            _, distances = embed_features(pcl, patches_idx, model_id)
            np.save(f"{shape_output_dir}/clr_patches_distances.npy", distances.astype(np.float16))

        print(f"  Distance matrix: {distances.shape}")

        # ====================================================================
        # Step 3: Extract symmetry clusters
        # ====================================================================
        print("\n[3/4] Extracting symmetry clusters...")

        if (os.path.exists(f"{shape_output_dir}/clr_clusters_components_idx.pkl") and
            os.path.exists(f"{shape_output_dir}/clr_clusters_icp_distance.pkl")):
            print("  Loading existing clusters...")
            clusters_components_idx = load_pickle(f"{shape_output_dir}/clr_clusters_components_idx.pkl")
            clusters_icp_distance = load_pickle(f"{shape_output_dir}/clr_clusters_icp_distance.pkl")
        else:
            # Compute clusters
            clusters_patch_idx = compute_clusters(
                distances, features=None, min_cluster_size=15, min_samples=5, silent=False
            )
            print(f"  Found {len(clusters_patch_idx)} initial clusters")

            # Subdivide clusters
            clusters_merged_patches_idx = subdivide_clusters(
                pcl, patches_idx, clusters_patch_idx, patches_origin_dist, silent=False
            )
            print(f"  Merged patches: {len(clusters_merged_patches_idx)} clusters")

            # Filter symmetry hypotheses
            clusters_components_idx, clusters_icp_distance = filter_symmetry_hypotheses(
                pcl, clusters_merged_patches_idx, symmetry_threshold=0.005, silent=False
            )
            print(f"  Final symmetry clusters: {len(clusters_components_idx)}")

            # Save symmetry detection results
            save_pickle(f"{shape_output_dir}/clr_clusters_components_idx.pkl", clusters_components_idx)
            save_pickle(f"{shape_output_dir}/clr_clusters_icp_distance.pkl", clusters_icp_distance)

        print(f"  Clusters: {len(clusters_components_idx)}")

        # ====================================================================
        # Step 4: Region growing
        # ====================================================================
        print("\n[4/4] Running region growing...")

        # Run region growing as subprocess to avoid module loading issues
        import subprocess
        import sys

        # Save clusters to temporary location for region growing script
        temp_dir = f"{shape_output_dir}/temp_rg_input"
        os.makedirs(temp_dir, exist_ok=True)
        save_pickle(f"{temp_dir}/clr_clusters_components_idx.pkl", clusters_components_idx)

        # Copy point cloud to expected location for region growing
        import shutil
        os.makedirs(f"data/precomputed/{shape_name}", exist_ok=True)
        shutil.copy(f"{shape_output_dir}/pcl.npy", f"data/precomputed/{shape_name}/pcl.npy")
        shutil.copy(f"{shape_output_dir}/clr_clusters_components_idx.pkl",
                    f"data/precomputed/{shape_name}/clr_clusters_components_idx.pkl")

        # Run region growing script
        cmd = [
            sys.executable,
            "region_growing.py",
            "-s", shape_name,
            "-clr", "True",
            "--batch_size", "50",
            "--icp_check_interval", "10",
            "--exponential_growth", "True",
            "--growth_factor", "1.5",
            "--skip_final_icp", "True",
            "--final_points_ratio", "0.05",
            "--n_jobs", "-1"
        ]

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Region growing failed with exit code {result.returncode}")

        # Copy results back to output directory
        shutil.copy(f"data/precomputed/{shape_name}/clr_region_growing_components_idx.pkl",
                    f"{shape_output_dir}/clr_region_growing_components_idx.pkl")
        shutil.copy(f"data/precomputed/{shape_name}/clr_regions_packed_components_idx.pkl",
                    f"{shape_output_dir}/clr_regions_packed_components_idx.pkl")
        shutil.copy(f"data/precomputed/{shape_name}/clr_regions_growing_max_icp_dists.pkl",
                    f"{shape_output_dir}/clr_regions_growing_max_icp_dists.pkl")

        # Cleanup temporary directories
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(f"data/precomputed/{shape_name}", ignore_errors=True)

        print("  ✓ Region growing completed successfully")

        print(f"\n{'='*70}")
        print(f"✓ Successfully processed {shape_name}")
        print(f"{'='*70}")

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ Error processing {shape_name}: {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = ArgumentParser(description="Batch symmetry detection pipeline")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing mesh files (.obj)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/precomputed",
        help="Output directory for results (default: data/precomputed)"
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=37809,
        help="Model ID for SymCL (default: 39938 / 37809)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip shapes that already have results"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.obj",
        help="File pattern to match (default: *.obj)"
    )

    args = parser.parse_args()

    # Validate model checkpoint exists
    try:
        ckpt_path = validate_model_checkpoint(args.model_id)
        print(f"\n✓ Found model checkpoint: {ckpt_path}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return

    # Find all mesh files
    input_path = Path(args.input_dir)
    mesh_files = sorted(input_path.glob(args.pattern))

    if not mesh_files:
        print(f"No mesh files found in {args.input_dir} matching pattern {args.pattern}")
        return

    print(f"\nFound {len(mesh_files)} mesh files to process:")
    for f in mesh_files:
        print(f"  - {f.name}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each shape
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }

    for mesh_file in mesh_files:
        shape_name = mesh_file.stem

        # Check if already exists
        if args.skip_existing:
            shape_output = os.path.join(args.output_dir, shape_name)
            if os.path.exists(f"{shape_output}/clr_region_growing_components_idx.pkl"):
                print(f"\n✓ Skipping {shape_name} (already processed)")
                results['skipped'].append(shape_name)
                continue

        success = process_shape(
            str(mesh_file),
            args.output_dir,
            model_id=args.model_id,
            skip_existing=args.skip_existing
        )

        if success:
            results['success'].append(shape_name)
        else:
            results['failed'].append(shape_name)

    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total shapes: {len(mesh_files)}")
    print(f"Successfully processed: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")

    if results['success']:
        print(f"\n✓ Success:")
        for name in results['success']:
            print(f"    {name}")

    if results['failed']:
        print(f"\n✗ Failed:")
        for name in results['failed']:
            print(f"    {name}")

    if results['skipped']:
        print(f"\n⊘ Skipped:")
        for name in results['skipped']:
            print(f"    {name}")

    print("="*70)
    print("INFO: Batch pipeline finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
