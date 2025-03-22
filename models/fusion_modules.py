import torch
import torch.nn as nn
from neuron_models import AdaptiveIzhikevichNeuron, AdaptiveLIFNeuron
from spikingjelly.clock_driven import neuron


class DynamicFeatureEnhancement(nn.Module):

    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        

        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels)
        )

        self.gate = nn.Parameter(torch.ones(1))
        
    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(-1)
            

        ca = self.channel_attention(x)
        x_ca = x * ca
        

        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial)
        x_sa = x_ca * sa

        enhanced = self.feature_enhancement(x_sa)
        

        output = x + self.gate * enhanced
        

        if x.size(-1) == 1:
            output = output.squeeze(-1)
            
        return output


class CrossModalAdaptiveFusion(nn.Module):

    
    def __init__(self, channels, num_heads=8, neuron_model='adaptive_lif'):
        super().__init__()
        

        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1)
        )
        

        if neuron_model == 'adaptive_lif':
            self.neuron = AdaptiveLIFNeuron(
                tau_mem=20.0,
                tau_thresh=60.0
            )
        elif neuron_model == 'izhikevich':
            self.neuron = AdaptiveIzhikevichNeuron(
                a=0.02, 
                b=0.2, 
                c=-65.0, 
                d=8.0
            )
        else:

            self.neuron = neuron.MultiStepLIFNode(
                tau=2.0, 
                detach_reset=True,
                backend='torch'
            )
        

        self.event_encoder = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            self.neuron,
            nn.Conv1d(channels, channels, 1)
        )
        

        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels*2, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 2, 1),
            nn.Softmax(dim=1)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )
        

        self.fusion_transform = nn.Sequential(
            nn.Conv1d(channels, channels*2, 1),
            nn.BatchNorm1d(channels*2),
            nn.GELU(),
            nn.Conv1d(channels*2, channels, 1),
            nn.BatchNorm1d(channels)
        )
        

        self.gate = nn.Parameter(torch.ones(1))
        
    def forward(self, spatial_feat, event_feat):

        s_feat = self.spatial_encoder(spatial_feat)
        

        if isinstance(self.neuron, (AdaptiveLIFNeuron, AdaptiveIzhikevichNeuron)):

            event_feat_t = event_feat.unsqueeze(-1).repeat(1, 1, 1, 5)

            event_spikes = self.neuron(event_feat_t)

            e_input = event_spikes.mean(dim=-1)

            e_feat = self.event_encoder[0](e_input)
            e_feat = self.event_encoder[1](e_feat)
            e_feat = self.event_encoder[3](e_feat)
        else:

            e_feat = self.event_encoder(event_feat)
        

        combined = torch.cat([s_feat, e_feat], dim=1)
        weights = self.weight_generator(combined)
        

        weighted_s = s_feat * weights[:, 0:1, :]
        weighted_e = e_feat * weights[:, 1:2, :]
        weighted_sum = weighted_s + weighted_e
        

        feat_for_attn = weighted_sum.permute(0, 2, 1)
        attn_out, _ = self.cross_attention(
            feat_for_attn, 
            feat_for_attn, 
            feat_for_attn
        )

        attn_out = attn_out.permute(0, 2, 1)
        

        transformed = self.fusion_transform(attn_out)
        

        output = weighted_sum + self.gate * transformed
        
        return output
