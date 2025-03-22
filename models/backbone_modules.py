import torch
import torch.nn as nn
from neuron_models import AdaptiveIzhikevichNeuron, AdaptiveLIFNeuron
from spikingjelly.clock_driven import neuron


class PyramidFeatureExtraction(nn.Module):

    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU()
        )
        

        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        

        out = torch.cat([out1, out2, out3, out4], dim=1)
        

        return self.fusion(out)

class SpikingFeatureProcessor(nn.Module):

    
    def __init__(self, channels, time_steps=5, neuron_model='adaptive_lif'):
        super().__init__()
        self.time_steps = time_steps
        

        if neuron_model == 'adaptive_lif':
            self.spike_neurons = AdaptiveLIFNeuron(
                tau_mem=20.0,
                tau_thresh=60.0
            )
        elif neuron_model == 'izhikevich':
            self.spike_neurons = AdaptiveIzhikevichNeuron(
                a=0.02, 
                b=0.2, 
                c=-65.0, 
                d=8.0
            )
        else:

            self.spike_neurons = neuron.MultiStepLIFNode(
                tau=2.0, 
                detach_reset=True,
                backend='torch'
            )
        

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(channels, channels*2, 3, padding=1),
            nn.BatchNorm1d(channels*2),
            nn.ReLU(),
            nn.Conv1d(channels*2, channels, 1),
            nn.BatchNorm1d(channels)
        )
        

        self.time_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Sigmoid()
        )
        

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        batch_size, channels, n_points = x.shape
        

        x_steps = x.unsqueeze(-1).repeat(1, 1, 1, self.time_steps)
        

        x_spike = self.spike_neurons(x_steps)
        

        x_spike = x_spike.view(batch_size, channels, -1)
        

        x_features = self.feature_extractor(x_spike)
        

        x_features = x_features.view(batch_size, channels, n_points, self.time_steps)
        

        x_temporal = x_features.permute(0, 2, 3, 1)
        x_reshaped = x_temporal.reshape(batch_size * n_points, self.time_steps, channels)
        
        lstm_out, _ = self.lstm(x_reshaped)
        lstm_out = lstm_out.reshape(batch_size, n_points, self.time_steps, channels)
        

        spike_features = lstm_out[:, :, -1, :].permute(0, 2, 1)
        

        gate = self.time_gate(spike_features)
        return spike_features * gate


class HierarchicalPointEncoder(nn.Module):

    
    def __init__(self, in_channels, out_channels, num_points=1024):
        super().__init__()
        self.num_points = num_points
        

        self.point_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        

        self.local_encoder = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        

        self.global_encoder = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        

        self.fusion = nn.Sequential(
            nn.Conv1d(1024 + 256 + 128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):

        B, C, N = x.shape
        

        point_feat = self.point_encoder(x)
        

        local_feat = self.local_encoder(point_feat)
        

        global_feat_max, _ = torch.max(self.global_encoder(local_feat), dim=2, keepdim=True)
        global_feat = global_feat_max.expand(-1, -1, N)
        

        concat_feat = torch.cat([point_feat, local_feat, global_feat], dim=1)
        

        fused_feat = self.fusion(concat_feat)
        
        return fused_feat


class HyperDenseBlock(nn.Module):

    def __init__(self, channels, expansion=4):
        super().__init__()
        self.expansion = expansion
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // expansion, 3, padding=1),
                nn.BatchNorm2d(channels // expansion),
                nn.GELU(),
                nn.Conv2d(channels // expansion, channels, 1)
            ) for _ in range(4)
        ])
        self.fusion = nn.Conv2d(channels * 5, channels, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        features = [branch(x) for branch in self.branches]
        concatenated = torch.cat(features + [x], dim=1)
        out = self.fusion(concatenated)
        

        if x.size(-1) == 1:
            out = out.squeeze(-1)
            
        return out


class SpikeConvGate(nn.Module):

    def __init__(self, channels, neuron_model='lif'):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
        

        if neuron_model == 'adaptive_lif':
            self.spike = AdaptiveLIFNeuron(tau_mem=20.0, tau_thresh=60.0)
        elif neuron_model == 'izhikevich':
            self.spike = AdaptiveIzhikevichNeuron(a=0.02, b=0.2)
        else:
            self.spike = neuron.MultiStepLIFNode(tau=2.0)
            
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        if x.dim() == 4:
            x = x.squeeze(-1)
        x_4d = x.unsqueeze(-1) if x.dim() == 3 else x
        spike = self.spike(x_4d)
        conv_out = self.conv(spike.squeeze(-1))
        return x + self.gate * conv_out


class Squeeze(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)