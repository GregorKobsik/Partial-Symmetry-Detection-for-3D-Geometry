import numpy as np
import torch

import potpourri3d as pp3d
import pytorch3d.io as io
import pytorch3d.ops as ops
from pytorch3d.structures import Meshes
from pytorch3d.transforms.transform3d import Transform3d

# from parse_off import read_off
# from utils.parse_off import read_off


class GeodesicPatchSampler:
    """
    Load and precompute data and solvers based on a single object.

    Args:
        file_path: Path to an OBJ-model.
    """

    def __init__(
        self,
        file_path=None,
        mesh_object=None,
        num_features=2**9,
        initial_pcl_size=2**16,
        patch_sizes=[2 ** (10 + i) for i in range(5)],
        origin_augmentation=0.1,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.patch_sizes = patch_sizes
        self.origin_augmentation = origin_augmentation

        self.data_transform = PointCloudAugmentationAndNormalization(num_features)

        if mesh_object is not None:
            mesh = mesh_object
        elif file_path is not None:
            mesh = self.load_model(file_path)
        else:
            print("ERROR: Either `file_path` or `mesh_object` must be specified.")
        self.points, self.normals = self.sample_points_and_normals(
            mesh, initial_pcl_size
        )
        self.solver = self.initialize_solver(self.points)

    def load_model(self, file_path):
        """
        Loads the OBJ/OFF-model into memory and creates a `Meshes` object from the vertices and faces.
        """
        file_path = (
            file_path
            if file_path[-4:] in (".obj", ".off")
            else file_path + "/model.obj"
        )
        if file_path[-4:] == ".obj":
            obj_model = io.load_obj(file_path, load_textures=False)
            verts, faces = obj_model[0], obj_model[1][0]
        # elif file_path[-4:] == ".off":
        #     verts, faces = read_off(file_path)
        else:
            print(
                "ERROR: Function supports only .obj/.off files. File path:", file_path
            )
        verts = self.data_transform.normalize(verts[None, :])[0]
        return Meshes(verts=[verts], faces=[faces])

    def sample_points_and_normals(self, mesh, N):
        """
        Sample uniformly a mesh to obtain a point cloud with corresponding normal vectors.
        """
        points, normals = ops.sample_points_from_meshes(mesh, N, return_normals=True)
        points = points[0].detach().clone()
        normals = normals[0].detach().clone()
        return points, normals

    def initialize_solver(self, points):
        """
        Initialize `Point Cloud Heat Solver` with uniformly sampled `initial_pcl_size` points on the mesh surface.
        """
        points = self.points.numpy().astype(np.float32)
        return pp3d.PointCloudHeatSolver(points)

    def sample_patch(self, idx, size):
        """
        Sample a single geodesic patch, given and index and the point cloud size.

        todo: add AxisAllignedLinearPicewiseWarping as additional data augmentation technique before sampling
        """
        # compute geodesic distances from a selected point (idx)
        dists = self.solver.compute_distance(idx)
        # sort all points based on distance
        sort_idx = np.argsort(dists)

        if isinstance(size, int):
            # extract a randomly sized patch
            patch = self.points[sort_idx[:size]].clone()
            normals = self.normals[sort_idx[:size]].clone()
            return patch, normals, sort_idx[:size]
        elif isinstance(size, list):
            # extract a randomly sized patches
            patch = []
            normals = []
            indizes = []
            for s in size:
                patch += [self.points[sort_idx[:s]].clone()]
                normals += [self.normals[sort_idx[:s]].clone()]
                indizes += [sort_idx[:s]]
            return patch, normals, indizes

    def sample_augmented_patches(self, idx, num_augmented_samples=2):
        """
        Sample a geodesic patch from the surface of the loaded object with corresponding augmented patches.
        """
        # define different patch sizes and select one size for all augmentations
        patch_size = int(np.random.choice(self.patch_sizes))
        # sample a geodesic patch originating at selected idx
        points_original, normals_original, _ = self.sample_patch(idx, patch_size)
        # downsample the number of points
        downsampled = self.data_transform.fps(
            points_original.clone()[None, :],
            normals_original.clone()[None, :],
        )
        # normalize point cloud data
        normalized = self.data_transform.normalize(
            downsampled[0].clone(),
            downsampled[1].clone(),
        )

        points_augmented = []
        normals_augmented = []
        # compute geodesic distances from a selected point (idx) and sort them
        sort_idx = np.argsort(self.solver.compute_distance(idx))
        # sample additional patches in the proximity of the first patch
        for _ in range(num_augmented_samples):
            # select a random index in the neighbourhood of the origin
            max_deviation_from_origin = int(patch_size * self.origin_augmentation)
            rand_idx = np.random.choice(sort_idx[: max_deviation_from_origin + 1])
            # sample a novel random patch
            pnts, nrmls, _ = self.sample_patch(rand_idx, patch_size)
            points_augmented.append(pnts)
            normals_augmented.append(nrmls)

        # convert sampled points to tensors
        points_augmented = torch.tensor(np.stack(points_augmented))
        normals_augmented = torch.tensor(np.stack(normals_augmented))
        # augment point cloud data, normalize it and downsample the number of points.
        augmented = self.data_transform(points_augmented, normals_augmented)

        return {
            "original_points": downsampled[0],
            "original_normals": downsampled[1],
            "normalized_points": normalized[0],
            "normalized_normals": normalized[1],
            "augmented_points": augmented[0],
            "augmented_normals": augmented[1],
        }


class PointCloudAugmentationAndNormalization:
    """
    Provide data augmentation, normalization and downsampling on point cloud data.
    """

    def __init__(self, num_features=2**9):
        super().__init__()
        self.num_features = num_features

    def __call__(self, points, normals, augment=True, normalize=True, fps=True):
        """
        Apply data augmentation and normalization:
            - (TODO) random jitter
            - random anisotropic scaling
            - random flipping
            - normalize translation to COG
            - normalize scale to uniform sphere

        Args:
            points: point cloud tensor with dimension [B x N x 3]
            normals: normal vector tensor with dimension [B x N x 3]
        """
        if augment:
            points, normals = self.augment(points, normals)
        if normalize:
            points, normals = self.normalize(points, normals)
        if fps:
            points, normals = self.fps(points, normals)
        return points, normals

    def augment(self, pnts, nrmls):
        """
        Apply data augmentation to archive reflection and similarity invariance.

        Args:
            pnts: point cloud tensor with dimension [B x N x 3]
            nrmls: normal vector tensor with dimension [B x N x 3]

        Return: augmented point cloud with normals [B x N x 3], [B x N x 3]
        """
        # TODO: add random jitter per point (Noise Invariance)

        for i in range(pnts.shape[0]):
            t = Transform3d()
            # random scaling independent for each axis (Similarity Invariance)
            t = t.scale(*np.random.uniform(low=0.8, high=1.2, size=[3]))
            # random flipping independent for each axis (Reflective Invariance)
            t = t.scale(*np.random.choice([-1, 1], size=[3]))

            # apply all transformations combined
            pnts[i] = t.transform_points(pnts[i])
            nrmls[i] = t.transform_normals(nrmls[i])

        return pnts, nrmls

    def normalize(self, pnts, nrmls=None):
        """
        Normalize the point cloud to a unit sphere centered at 0.
            - normalize translation to COG
            - normalize scale to uniform sphere

        Args:
            pnts: point cloud tensor with dimension [B x N x 3]
            nrmls: normal vector tensor with dimension [B x N x 3]

        Return: normalized point cloud with normals [B x N x 3], [B x N x 3]
        """
        # move to COG (Translation Invaraince)
        pnts -= torch.mean(pnts, dim=1)[:, None, :]
        # normalize to uniform sphere (Scale Invariance)
        pnts /= (
            torch.cdist(pnts, torch.zeros(1, 3, device=pnts.device))
            .max(dim=1)
            .values[:, None, :]
        )

        if nrmls is None:
            return pnts
        else:
            return pnts, nrmls

    def fps(self, pnts, nrmls):
        """
        Perform farthest point sampling to reduce number of points in given point cloud to `self.num_features` (F).

        Args:
            pnts: point cloud tensor with dimension [B x N x 3]
            nrmls: normal vector tensor with dimension [B x N x 3]

        Return: downsampled point cloud data with normals [B x F x 3], [B x F x 3], where F << N.
        """
        pnts, selected_idx = ops.sample_farthest_points(pnts, K=self.num_features)
        nrmls_out = torch.zeros_like(pnts)
        for i in range(selected_idx.shape[0]):
            nrmls_out[i] = nrmls[i, selected_idx[i]]
        return pnts, nrmls_out
        # TODO: check if necessary:
        # return pnts, nrmls[selected_idx]
