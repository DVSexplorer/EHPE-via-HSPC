import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv_pos = nn.Sequential(
            nn.Conv1d(3, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1)
        )

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        position_embedding = self.conv_pos(xyz)
        return position_embedding.permute(0, 2, 1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, pos=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if pos is not None:
            q = q + pos.reshape(B, self.num_heads, N, self.head_dim)
            k = k + pos.reshape(B, self.num_heads, N, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlockV3(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MultiHeadSelfAttention(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

        self.pos_embedding = PositionalEncoding(channels)

    def forward(self, x, xyz):
        pos = self.pos_embedding(xyz)
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x


class DynamicGraphConv(nn.Module):
    def __init__(self, channels_in, channels_out, k=16):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in * 2, channels_out, 1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU()
        )

    def forward(self, x, xyz):


        B, N, C = x.shape


        k = min(self.k, N)


        inner = -2 * torch.matmul(xyz, xyz.transpose(2, 1))
        xx = torch.sum(xyz ** 2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]


        batch_indices = torch.arange(B, device=x.device).view(-1, 1, 1).expand(-1, N, k)
        point_indices = torch.arange(N, device=x.device).view(1, -1, 1).expand(B, -1, k)

        features = x[batch_indices, idx, :]

        central_features = x.unsqueeze(2).expand(-1, -1, k, -1)
        edge_features = torch.cat([central_features, features - central_features], dim=-1)


        edge_features = edge_features.permute(0, 3, 1, 2)
        out = self.conv(edge_features)
        out = out.max(dim=-1)[0]

        return out.permute(0, 2, 1)


class BackboneV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints = cfg.num_points
        nblocks = 4
        d_points = 5
        base_channels = 64


        self.fc1 = nn.Sequential(
            nn.Linear(d_points, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, base_channels)
        )

        self.transformer1 = TransformerBlockV3(base_channels, num_heads=8)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()

        for i in range(nblocks):
            channel = base_channels * 2 ** (i + 1)
            self.transition_downs.append(
                nn.Linear(channel // 2, channel)
            )
            self.transformers.append(
                TransformerBlockV3(channel, num_heads=8)
            )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        xyz = x[..., :3]
        points = self.fc1(x)


        points = self.transformer1(points, xyz)


        for i in range(len(self.transition_downs)):
            points = self.transition_downs[i](points)
            points = self.transformers[i](points, xyz)


        return points.mean(1)


class Pose_PointTransformerV3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = BackboneV3(args)
        self.args = args
        self.sizeH = args.sensor_sizeH
        self.sizeW = args.sensor_sizeW
        self.num_joints = args.num_joints

        base_channels = 64
        final_channels = base_channels * (2 ** 4)


        self.fc2 = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_joints * 128),
            nn.ReLU(),
        )

        self.mlp_head_x = nn.Linear(128, self.sizeW)
        self.mlp_head_y = nn.Linear(128, self.sizeH)

    def forward(self, x):
        batch_size = x.size(0)


        features = self.backbone(x)

        res = self.fc2(features)
        res = res.view(batch_size, self.num_joints, -1)


        pred_x = self.mlp_head_x(res)
        pred_y = self.mlp_head_y(res)

        return pred_x, pred_y



def furthest_point_sample(xyz, npoint):

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points, idx):

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points