import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Select features from neighbors.

    Args:
        x: the input features with shape (B, PTS, D).
        k: the number of neighbors.
        idx: the indices indicating the K neighbors for each input point
                with shape (B, PTS, K).

    Returns:
        The selected features with shape # (B, PTS, K, 2*D)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, dim_latent, dim_feature, k=20, depth=4, block_size=32, dropout=0.1,
                 output_at='vertices'):  # bb - Building block
        super(DGCNN, self).__init__()

        self.num_neighs = k
        self.latent_dim = dim_latent
        self.input_features = 6
        self.output_at = output_at

        self.depth = depth
        bb_size = block_size
        dim_feature = dim_feature

        self.convs = []

        for i in range(self.depth):
            in_features = self.input_features if i == 0 else bb_size * (2 ** (i + 1)) * 2
            out_features = bb_size * 4 if i == 0 else in_features
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, bias=False), nn.BatchNorm2d(out_features),
                nn.LeakyReLU(negative_slope=0.2),
            )
            )
        last_in_dim = bb_size * 2 * sum([2 ** i for i in range(1, self.depth + 1, 1)])

        self.convs.append(
            nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim),
                nn.LeakyReLU(negative_slope=0.2),
            )
        )
        self.convs = nn.ModuleList(self.convs)

        input_latent_dim = self.latent_dim

        self.linear1 = nn.Linear(input_latent_dim * 2, bb_size * 64, bias=False)
        self.bn6 = nn.Identity()  # nn.BatchNorm1d(bb_size * 64)
        self.dp1 = nn.Dropout(p=dropout)

        self.linear2 = nn.Linear(bb_size * 64, bb_size * 32)
        self.bn7 = nn.Identity()  # nn.BatchNorm1d(bb_size * 32)
        self.dp2 = nn.Dropout(p=dropout)

        self.linear3 = nn.Linear(bb_size * 32, dim_feature)

    def forward_per_point(self, x, start_neighs=None):
        x = x.transpose(1, 2)  # DGCNN assumes BxFxN

        if (start_neighs is None):
            start_neighs = knn(x, k=self.num_neighs)

        x = get_graph_feature(x, k=self.num_neighs, idx=start_neighs)
        # other = x[:, :3, :, :]
        #
        # if (self.hparams.concat_xyz_to_inv):
        #     x = torch.cat([x, other], dim=1)

        outs = [x]
        for conv in self.convs[:-1]:
            if (len(outs) > 1):
                x = get_graph_feature(outs[-1], k=self.num_neighs, idx=None)
            x = conv(x)
            outs.append(x.max(dim=-1, keepdim=False)[0])

        x = torch.cat(outs[1:], dim=1)
        features = self.convs[-1](x)
        return features.transpose(1, 2)
        # It is advised

    def aggregate_all_points(self, features_per_point):
        # if (features_per_point.shape[1] == self.hparams.num_points):
        features_per_point = features_per_point.transpose(1, 2)
        batch_size = features_per_point.size(0)
        x1 = features_per_point.max(-1)[0].view(batch_size, -1)
        x2 = features_per_point.mean(-1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

    def forward(self, x, start_neighs=None):
        features_per_point = self.forward_per_point(x, start_neighs=start_neighs)

        if self.output_at == 'vertices':
            out = features_per_point
        elif self.output_at == 'global':
            out = self.aggregate_all_points(features_per_point)
        else:
            raise ValueError

        return out
