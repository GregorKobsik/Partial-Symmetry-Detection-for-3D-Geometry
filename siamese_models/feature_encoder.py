import torch.nn as nn

from vector_neurons import VNPointNetEncoder, VNDGCNNEncoder
# from riconv import RIConv2Encoder


class FeatureEncoder(nn.Module):

    def __init__(
        self,
        encoder_type: str = "VNPointNet",
    ):
        super().__init__()
        if encoder_type == "VNPointNet":
            # works directly with x,y,z point cloud data
            self.feature_enc = VNPointNetEncoder(head_dim=128)
        elif encoder_type == "VNDGCNN":
            # works directly with x,y,z point cloud data
            self.feature_enc = VNDGCNNEncoder(head_dim=128)
        elif encoder_type == "RIConv2":
            # works with x,y,z point cloud data and precomputed LRA features
            # Install pointnet++ cuda operation library with $ python riconv/pointnet2/setup.py install
            # required PyTroch version < 1.11, as PyTorch 1.11 removed THC/THC.h files
            # self.feature_enc = RIConv2Encoder(head_dim=128)
            print("RIConv2 not supported, yet. Waiting for port from pointnet2 to pytorch3d.")

        self.proj_net = nn.Sequential(
            nn.Sequential(nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64)),
            nn.Sequential(nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32)),
        )

    def forward(self, x):
        """
        Input: [B x N x 3/6], Output: [B x 32]
        B: batch
        N: points (=512)
        """
        assert (len(x.shape) == 3)
        return self.proj_net(self.feature_enc(x))
