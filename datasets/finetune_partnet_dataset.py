import random

import torch
from torch.utils.data import Dataset, DataLoader

from .icp_distance_dataset import ICPDistanceDataModule, ICPDistanceDataset


class FinetunePartNetDataModule(ICPDistanceDataModule):
    def __init__(
        self,
        file_path: str = "data/PartNet/patches/Chair/",
        batch_size: int = 8,
        lra_feat: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.lra_feat = lra_feat

    def train_dataloader(self):
        return DataLoader(
            FinetunePartNetDataset(self.file_path, mode="train", lra_feat=self.lra_feat, batch_size=self.batch_size),
            batch_size=1,
            num_workers=6,
            collate_fn=self.pass_through,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        file_path = "data/multivariate/uniform_1024/"
        return DataLoader(
            ICPDistanceDataset(file_path, mode="val", lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=self.extract_distance_matrix,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            FinetunePartNetDataset(self.file_path, mode="test", lra_feat=self.lra_feat, batch_size=self.batch_size),
            batch_size=1,
            num_workers=6,
            collate_fn=self.pass_through,
            shuffle=True,
            drop_last=True,
        )

    def predict_dataloader(self):
        file_path = "data/multivariate/uniform_1024/"
        return DataLoader(
            ICPDistanceDataset(file_path, mode="predict", lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=self.extract_distance_matrix,
            shuffle=False,
            drop_last=True,
        )

    def pass_through(self, data):
        """Collate function to pass the output of the dataset throught the dataloader."""
        return data[0]

# TODO: Implment FinetunePartNetDataset (copy from FinetuneModelNetDataset)

class FinetunePartNetDataset(Dataset):
    def __init__(
        self,
        file_path: str = "",
        max_size: int = 10_000,
        mode: str = "train",
        batch_size=1,
        lra_feat: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.lra_feat = lra_feat

        self.patches_paths = sorted(open(file_path + mode + "_glob_patches_1024.txt", "r").read().splitlines())
        self.icp_distances_paths = sorted(
            open(file_path + mode + "_glob_icp_distances_1024.txt", "r").read().splitlines()
        )
        assert len(self.patches_paths) == len(
            self.icp_distances_paths
        ), "The number of patch files does not match the number of ICP distance files."

        self.n_paths = min(len(self.patches_paths), max_size)

    def __len__(self) -> int:
        """
        Return number of objects in the dataset.
        """
        return self.n_paths

    def __getitem__(self, idx):
        """
        Return a batched normalized point cloud with corresponding ICP distance matrix.
        Note: The output is already batched.

        N: point cloud size
        X: dataset size

        Return: [N x 3/6], [N]
        """
        # load patches and icp distances
        points = torch.load(self.patches_paths[idx])
        icp_distances = torch.load(self.icp_distances_paths[idx])

        # select random batches from a single file (only intra-shape ICP distances computed)
        batch_idx = torch.multinomial(torch.arange(len(points), dtype=torch.float), self.batch_size)
        selected_points = points[batch_idx, 0]
        selected_icp_distances = torch.index_select(
            torch.index_select(icp_distances, dim=0, index=batch_idx), dim=1, index=batch_idx
        )

        if self.lra_feat:
            print("RIConv2 not implemented, yet. Wait for port from pointnet2 to pytorch3d.")

        return selected_points, selected_icp_distances
