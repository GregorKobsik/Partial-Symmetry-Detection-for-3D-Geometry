import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
# from riconv.riconv2_utils import compute_LRA


class ContrastiveICPDataModule(pl.LightningDataModule):

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

    def train_dataloader(self, shuffle=False):
        return DataLoader(
            ContrastiveICPDataset(self.file_path, mode='train', lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=extract_distance_matrix,
            shuffle=shuffle,
            drop_last=True,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            ContrastiveICPDataset(self.file_path, mode='val', lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=extract_distance_matrix,
            shuffle=shuffle,
            drop_last=True,
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            ContrastiveICPDataset(self.file_path, mode='test', lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=extract_distance_matrix,
            shuffle=shuffle,
            drop_last=True,
        )

    def predict_dataloader(self, shuffle=False):
        return DataLoader(
            ContrastiveICPDataset(self.file_path, mode='predict', lra_feat=self.lra_feat),
            batch_size=self.batch_size,
            num_workers=6,
            collate_fn=extract_distance_matrix,
            shuffle=shuffle,
            drop_last=True,
        )


def extract_distance_matrix(data):
    """ Collate function to extract the correct ICP distance matrix for each batch.
    Input: [N x 3/6], [N x 3/6], [X x X], [1]
    Return:
        - point cloud data A: [B x N x 3/6]
        - point cloud data B: [B x N x 3/6]
        - filtered ICP distance matrix: [B x B]
    """
    pcl_A = torch.stack([d[0] for d in data])
    pcl_B = torch.stack([d[1] for d in data])
    idx = torch.tensor([d[3] for d in data])
    matrix_full = data[0][2]
    matrix_filtered = torch.index_select(torch.index_select(matrix_full, dim=0, index=idx), dim=1, index=idx)
    return pcl_A, pcl_B, matrix_filtered


class ContrastiveICPDataset(Dataset):

    def __init__(
        self,
        file_path: str = "",
        max_size: int = None,
        mode: str = 'train',
        lra_feat: bool = False,
    ):
        super().__init__()
        self.points_A = torch.load(f"{file_path}/{mode}/points_A.pt")
        self.points_B = torch.load(f"{file_path}/{mode}/points_B.pt")
        if lra_feat:
            # append LRA features, if we use RIConv++ encoder
            # self.points_A = torch.cat((self.points_A, compute_LRA(self.points_A)), 2)
            # self.points_B = torch.cat((self.points_B, compute_LRA(self.points_B)), 2)
            print("RIConv2 not implemented, yet. Wait for port from pointnet2 to pytorch3d.")

        self.distances_A = torch.load(f"{file_path}/{mode}/point2point_pytorch3d/icp_distances_A.pt")

        # limit dataset size
        if max_size is not None:
            self.points_A = self.points_A[:max_size]
            self.points_B = self.points_B[:max_size]
            self.distances_A = self.distances_A[:max_size, :max_size]

    def __len__(self) -> int:
        """
        Return a virtual number of objects in the dataset.
        """
        return self.points_A.shape[0]

    def __getitem__(self, idx):
        """
        N: point cloud size
        X: dataset size

        Return: [N x 3], [N x 3], [X x X], 1
        """
        return self.points_A[idx], self.points_B[idx], self.distances_A, idx
