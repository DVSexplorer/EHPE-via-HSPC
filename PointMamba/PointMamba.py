import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tools.logger import *


class SelectiveScan(nn.Module):


    def __init__(self, d_model, d_state=16, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A = nn.Parameter(torch.randn(1, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_state, d_model))
        self.D = nn.Parameter(torch.randn(d_model))

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            nn.init.normal_(self.A, mean=-0.5, std=0.1)
            nn.init.normal_(self.B, std=0.1)
            nn.init.normal_(self.C, std=0.1)
            nn.init.zeros_(self.D)

    def forward(self, x):

        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        B, N, E = x.shape
        if E != self.d_model:
            raise ValueError(f"Expected feature dimension {self.d_model}, got {E}")

        A = -torch.exp(self.A)
        u = x.view(B * N, E)

        delta = F.softplus(u @ self.B).view(B, N, -1)
        h = torch.zeros(B, self.d_state, device=x.device)

        outputs = []
        for i in range(N):
            h = h * torch.exp(delta[:, i] * A)
            h = h + (x[:, i] @ self.B)
            y_i = h @ self.C + x[:, i] * self.D
            outputs.append(y_i)

        return self.dropout(torch.stack(outputs, dim=1))


class TransformerBlock(nn.Module):


    def __init__(self, d_points, d_model, k):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        self.k = k

    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / (k.size(-1) ** 0.5), dim=-2)

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        return self.fc2(res) + pre


class TransitionDown(nn.Module):


    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


def sample_and_group_all(xyz, points):

    device = xyz.device
    B, N, C = xyz.shape

    new_xyz = torch.zeros(B, 1, C).to(device)

    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:

        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)

        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.npoints = cfg.num_points
        self.nblocks = 4
        self.nneighbor = 16
        self.transformer_dim = 128
        d_points = 5


        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )


        self.transformer1 = TransformerBlock(32, self.transformer_dim, self.nneighbor)


        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList()

        for i in range(self.nblocks):
            channel = 32 * 2 ** (i + 1)

            self.transition_downs.append(
                TransitionDown(
                    self.npoints // 4 ** (i + 1),
                    self.nneighbor,
                    [channel // 2 + 3, channel, channel]
                )
            )


            self.transformers.append(
                TransformerBlock(channel, self.transformer_dim, self.nneighbor)
            )


            self.mamba_blocks.append(
                SelectiveScan(
                    d_model=channel,
                    d_state=32,
                )
            )


        self.norms = nn.ModuleList([
            nn.LayerNorm(32 * 2 ** (i + 1)) for i in range(self.nblocks)
        ])

    def forward(self, x):
        B = x.size(0)
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))


        features_list = []

        for i in range(self.nblocks):

            xyz, points = self.transition_downs[i](xyz, points)


            points = self.transformers[i](xyz, points)


            if len(points.shape) == 2:
                N = 1
                C = points.shape[1]
                points = points.view(B, N, C)


            points_enhanced = self.mamba_blocks[i](points)
            points = self.norms[i](points + points_enhanced)

            features_list.append(points)


        global_features = [F.adaptive_avg_pool1d(f.transpose(1, 2), 1).squeeze(-1)
                           for f in features_list]
        global_feature = torch.cat(global_features, dim=-1)

        return global_feature

    def forward(self, x):
        B = x.size(0)
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))

        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)


            if len(points.shape) == 2:
                N = 1
                C = points.shape[1]
                points = points.view(B, N, C)

            points_enhanced = self.mamba_blocks[i](points)
            points = points + points_enhanced

        global_feature = points.mean(1)
        return global_feature


class Pose_PointMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Backbone(config)
        self.num_joints = config.num_joints
        self.sizeH = config.sensor_sizeH
        self.sizeW = config.sensor_sizeW


        backbone_out_dim = 512


        self.decoder = nn.Sequential(
            nn.Linear(backbone_out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_joints * 128),
            nn.ReLU()
        )


        self.mlp_head_x = nn.Linear(128, self.sizeW)
        self.mlp_head_y = nn.Linear(128, self.sizeH)

    def forward(self, x):
        batch_size = x.size(0)


        if x.shape[1] <= 5 and x.shape[2] > 5:
            x = x.permute(0, 2, 1)

        points = self.backbone(x)


        features = self.decoder(points)


        features = features.view(batch_size, self.num_joints, 128)

        pred_x = self.mlp_head_x(features)
        pred_y = self.mlp_head_y(features)

        return pred_x, pred_y


def square_distance(src, dst):
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    device = points.device
    B = points.shape[0]


    if len(points.shape) == 2:
        points = points.unsqueeze(0)
        B = 1

    if len(idx.shape) == 3:
        S, K = idx.shape[1], idx.shape[2]
        view_shape = (B, 1, 1)
        repeat_shape = (1, S, K)
    else:
        S = idx.shape[1]
        view_shape = (B, 1)
        repeat_shape = (1, S)

    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(*view_shape).repeat(*repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False):
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)

    if knn:
        dists = square_distance(new_xyz, xyz)
        idx = dists.argsort()[:, :, :nsample]
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx