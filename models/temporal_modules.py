import torch
import torch.nn as nn


class MultiScaleTemporalEncoding(nn.Module):

    def __init__(self, channels, scales=[1, 3, 5, 7]):
        super().__init__()
        assert channels % len(scales) == 0,
        self.scales = scales
        self.branch_channels = channels // len(scales)

        self.convs = nn.ModuleList([
            nn.Conv1d(channels, self.branch_channels, k, padding=k // 2)
            for k in scales
        ])
        self.fusion = nn.Conv1d(self.branch_channels * len(scales), channels, 1)
        

        self.freq_encoder = nn.Sequential(
            nn.Linear(20, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, channels)
        )
        

        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x):

        B, C, N, T = x.shape
        

        x_time = x.view(B, C, N * T)
        outputs = [conv(x_time) for conv in self.convs]
        concatenated = torch.cat(outputs, dim=1)
        time_out = self.fusion(concatenated)
        time_out = time_out.view(B, C, N, T)

        x_padded = torch.nn.functional.pad(x.permute(0, 1, 3, 2), (0, 0, 0, 20-T))
        x_fft = torch.fft.rfft(x_padded, dim=2).abs()
        x_fft = x_fft.permute(0, 1, 3, 2)
        x_fft = x_fft.reshape(B, C, N, -1)
        

        x_fft = x_fft.permute(0, 2, 3, 1)
        x_fft = x_fft.reshape(B * N, -1, C)
        freq_features = self.freq_encoder(x_fft.transpose(1, 2))
        freq_features = freq_features.reshape(B, N, C).permute(0, 2, 1)
        

        freq_features = freq_features.unsqueeze(-1).expand(-1, -1, -1, T)
        

        combined = time_out + self.gate * freq_features
        
        return combined


class MultiHeadSpatioTemporalAttention(nn.Module):

    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, "c"
        self.num_heads = num_heads
        head_dim = channels // num_heads
        

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        

        self.output_proj = nn.Conv2d(channels, channels, 1)

        self.temporal_attention = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, (1, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):

        B, C, N, T = x.shape

        x_flat = x.view(B, C, N*T).permute(0, 2, 1)
        

        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        

        attn_out = attn_out.permute(0, 2, 1).view(B, C, N, T)
        

        spatial_gate = self.spatial_gate(attn_out)
        

        temporal_gate = self.temporal_attention(attn_out)
        

        gated = attn_out * spatial_gate * temporal_gate
        

        output = self.output_proj(gated)
        
        return output

class AdaptiveTimeStep(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.timestep_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(),
            nn.Conv2d(channels//4, 1, 1),
            nn.Sigmoid()
        )
        

        self.time_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        

        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x):

        time_weights = self.timestep_predictor(x)
        

        time_features = self.time_encoder(x)
        

        return x * time_weights + self.gate * time_features
