import random

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# from riconv.riconv2_utils import compute_LRA

from .contrastive_icp_dataset import ContrastiveICPDataset, extract_distance_matrix


class PretrainPartNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path: str = "data/",
        batch_size: int = 8,
        lra_feat: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.lra_feat = lra_feat

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            ContrastivePartNetDataset(
                self.file_path, mode="train", lra_feat=self.lra_feat, max_size=10_000
            ),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=remove_distance_matrix,
            shuffle=shuffle,
            drop_last=True,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            ContrastiveICPDataset(
                "data/multivariate/uniform_1024/",
                mode="val",
                lra_feat=self.lra_feat,
            ),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=remove_distance_matrix,
            shuffle=shuffle,
            drop_last=True,
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            ContrastiveICPDataset(
                "data/multivariate/uniform_1024/",
                mode="test",
                lra_feat=self.lra_feat,
            ),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=remove_distance_matrix,
            shuffle=shuffle,
            drop_last=True,
        )

    def predict_dataloader(self, shuffle=False):
        return DataLoader(
            ContrastiveICPDataset(
                "data/multivariate/uniform_1024/",
                mode="predict",
                lra_feat=self.lra_feat,
            ),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=remove_distance_matrix,
            shuffle=shuffle,
            drop_last=True,
        )


def remove_distance_matrix(data):
    """Collate function to pass only point clouds.
    Input: [N x 3/6], [N x 3/6], [N x 3]
    Return:
        - point cloud data A: [B x N x 3/6]
        - point cloud data B: [B x N x 3/6]
    """
    pcl_A = torch.stack([d[0] for d in data])
    pcl_B = torch.stack([d[1] for d in data])
    return pcl_A, pcl_B


def proxy_distance_matrix(data):
    """Collate function to proxide a artificial proxy ICP distance matrix.
    Input: [N x 3/6], [N x 3/6], [N x 3]
    Return:
        - point cloud data A: [B x N x 3/6]
        - point cloud data B: [B x N x 3/6]
        - proxy ICP distance matrix: [B x B]
    """
    pcl_A = torch.stack([d[0] for d in data])
    pcl_B = torch.stack([d[1] for d in data])
    B = len(pcl_A)
    return pcl_A, pcl_B, torch.ones((B, B), device=pcl_A.device)


class ContrastivePartNetDataset(Dataset):
    def __init__(
        self,
        file_path: str = "",
        mode: str = "train",
        max_size: int = 10_000,
        lra_feat: bool = False,
    ):
        super().__init__()
        self.lra_feat = lra_feat

        # load paths to patches
        self.paths = []
        for patch_size in (512, 1024, 2048, 4096, 8192):
            self.paths += (
                open(file_path + f"{mode}_glob_patches_{patch_size}.txt", "r")
                .read()
                .splitlines()
            )
        assert len(self.paths) > 0, "Please create a valid glob file for patches."
        self.n_paths = len(self.paths)
        n_patches = self.n_paths * 64
        self.n_patches = min(n_patches, max_size)

        self.points_A = torch.tensor([])
        self.points_B = torch.tensor([])

    def __len__(self) -> int:
        """
        Return number of objects in the dataset.
        """
        return self.n_patches

    def __getitem__(self, idx):
        """
        N: point cloud size
        X: dataset size

        Return: [N x 3], [N x 3]
        """
        if len(self.points_A) < min(1_000, self.n_patches):
            # if we have less than 1_000 patches in memory, load a new batch
            points = torch.load(self.paths[random.randint(0, self.n_paths - 1)])
            if self.lra_feat:
                # append LRA features, if we use RIConv++ encoder
                # points_a = torch.cat((points[:, 0], compute_LRA(points[:, 0])), 2)
                # points_b = torch.cat((points[:, 1], compute_LRA(points[:, 1])), 2)
                print(
                    "RIConv2 not implemented, yet. Wait for port from pointnet2 to pytorch3d."
                )
            else:
                points_a = points[:, 0]
                points_b = points[:, 1]

            # append new patches
            self.points_A = torch.cat([self.points_A, points_a])
            self.points_B = torch.cat([self.points_B, points_b])
            # random permutation of patches - shuffle them
            idx = torch.randperm(self.points_A.shape[0])
            self.points_A = self.points_A[idx]
            self.points_B = self.points_B[idx]

        points_a, self.points_A = self.points_A[0], self.points_A[1:]
        points_b, self.points_B = self.points_B[0], self.points_B[1:]

        return points_a, points_b
