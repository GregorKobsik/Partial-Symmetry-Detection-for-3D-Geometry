import torch
import torch.nn as nn
import torch.nn.functional as F
import vector_neurons.vn_layers as vnn


class VNDGCNNEncoder(nn.Module):

    def __init__(self, n_knn=20, head_dim=128):
        super(VNDGCNNEncoder, self).__init__()
        self.n_knn = n_knn
        self.head_dim = head_dim

        self.conv1 = vnn.VNEdgeConv(2, 64 // 3)
        self.conv2 = vnn.VNEdgeConv(64 // 3 * 2, 64 // 3)
        self.conv3 = vnn.VNEdgeConv(64 // 3 * 2, 128 // 3)
        self.conv4 = vnn.VNEdgeConv(128 // 3 * 2, 256 // 3)

        self.conv5 = vnn.VNLinearLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2, 1024 // 3, dim=4, share_nonlinearity=True)

        self.std_feature = vnn.VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024 // 3) * 12, 512)

        if self.head_dim > 0:
            self.head = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.5),
                nn.Linear(256, head_dim),
            )

    def forward(self, x):
        # incoming B N D
        x = x.transpose(1, 2)
        B, D, N = x.size()
        x = x.unsqueeze(1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, num_points)

        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x1, x2), 1)
        x = self.linear1(x)

        if self.head_dim > 0:
            x = self.head(x)

        return x