import torch
import torch.nn as nn
import torch.nn.functional as F
import vector_neurons.vn_layers as vnn
from vector_neurons.utils import get_graph_feature_cross


class STNkd(nn.Module):
    def __init__(self, d=64):
        super(STNkd, self).__init__()
        self.conv1 = vnn.VNLinearLeakyReLU(d, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = vnn.VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = vnn.VNLinearLeakyReLU(128 // 3, 1024 // 3, dim=4, negative_slope=0.0)

        self.fc1 = vnn.VNLinearLeakyReLU(1024 // 3, 512 // 3, dim=3, negative_slope=0.0)
        self.fc2 = vnn.VNLinearLeakyReLU(512 // 3, 256 // 3, dim=3, negative_slope=0.0)

        self.pool = vnn.VNMeanPool()

        self.fc3 = vnn.VNLinear(256 // 3, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class VNPointNetEncoder(nn.Module):
    def __init__(self, n_knn=20, head_dim=128):
        super(VNPointNetEncoder, self).__init__()
        self.n_knn = n_knn
        self.head_dim = head_dim

        self.conv_pos = vnn.VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.pool = vnn.VNMeanPool()

        self.conv1 = vnn.VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = vnn.VNLinearLeakyReLU(64 // 3 * 2, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = vnn.VNLinear(128 // 3, 1024 // 3)
        self.bn3 = vnn.VNBatchNorm(1024 // 3, dim=4)

        self.std_feature = vnn.VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)
        self.fstn = STNkd(d=64 // 3)
        self.fc1 = nn.Linear(1024 // 3 * 6, 512)

        if self.head_dim > 0:
            self.head = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.Dropout(p=0.4),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(256, head_dim),
            )

    def forward(self, x):
        # incoming B N D
        x = x.transpose(1, 2)
        B, D, N = x.size()
        x = x.unsqueeze(1)

        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x)

        x = self.conv1(x)

        x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
        x = torch.cat((x, x_global), 1)

        x = self.conv2(x)
        x = self.bn3(self.conv3(x))

        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, _ = self.std_feature(x)
        x = x.view(B, -1, N)

        x = torch.max(x, -1, keepdim=False)[0]
        x = self.fc1(x)

        if self.head_dim > 0:
            x = self.head(x)

        return x
