import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
#from riconv.riconv2_utils import compute_LRA


class ICPDistanceDataModule(pl.LightningDataModule):

    def __init__(
        self,
        file_path: str = "data/table/uniform_1024/",
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
            ICPDistanceDataset(self.file_path, mode='train', lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=self.extract_distance_matrix,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ICPDistanceDataset(self.file_path, mode='val', lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=self.extract_distance_matrix,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            ICPDistanceDataset(self.file_path, mode='test', lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=self.extract_distance_matrix,
            shuffle=False,
            drop_last=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            ICPDistanceDataset(self.file_path, mode='predict', lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=self.extract_distance_matrix,
            shuffle=False,
            drop_last=True,
        )

    def extract_distance_matrix(self, data):
        """ Collate function to extract the correct ICP distance matrix for each batch.
        Input: [N x 3/6], [X x X], [1]
        Return:
            - point cloud data: [B x N x 3/6]
            - filtered ICP distance matrix: [B x B]
        """
        pcl_normalized = torch.stack([d[0] for d in data])
        idx = torch.tensor([d[2] for d in data])
        matrix_full = data[0][1]
        matrix_filtered = torch.index_select(torch.index_select(matrix_full, dim=0, index=idx), dim=1, index=idx)
        return pcl_normalized, matrix_filtered


class ICPDistanceDataset(Dataset):

    def __init__(
        self,
        file_path: str = "",
        max_size: int = None,
        mode: str = 'train',
        lra_feat: bool = False,
    ):
        super().__init__()
        self.normalized_data = torch.load(f"{file_path}/{mode}/points_normalized.pt")
        if lra_feat:
            # append LRA features, if we use RIConv++ encoder
            #self.normalized_data = torch.cat((self.normalized_data, compute_LRA(self.normalized_data)), 2)
            print("RIConv2 not implemented, yet. Wait for port from pointnet2 to pytorch3d.")

        self.distance_data = torch.load(f"{file_path}/{mode}/point2point_pytorch3d/icp_distances.pt")

        # limit dataset size
        if max_size is not None:
            self.normalized_data = self.normalized_data[:max_size]
            self.distance_data = self.distance_data[:max_size, :max_size]

        # smooth out registration errors
        self.distance_data = (self.distance_data.transpose(0, 1) + self.distance_data) / 2.0

    def __len__(self) -> int:
        """
        Return a virtual number of objects in the dataset.
        """
        return self.normalized_data.shape[0]

    def __getitem__(self, idx):
        """
        Return the normalized point cloud. Additionally, return the whole ICP distance matrix and corresponding data idx.
        The ICP distance matrix needs to be filtered out in the collate function as a postpropcessing step.

        N: point cloud size
        X: dataset size

        Return: [N x 3/6], [X x X], [1]
        """
        return self.normalized_data[idx], self.distance_data, idx
