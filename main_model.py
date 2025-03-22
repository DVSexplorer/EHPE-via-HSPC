import torch
import torch.nn as nn

from backbone_modules import PyramidFeatureExtraction, SpikingFeatureProcessor, HierarchicalPointEncoder
from fusion_modules import DynamicFeatureEnhancement, CrossModalAdaptiveFusion
from skeleton_modules import SkeletonConstraintModule
from temporal_modules import MultiScaleTemporalEncoding, MultiHeadSpatioTemporalAttention, AdaptiveTimeStep
from prediction_header import Header

class UltraHybridSpikeNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spatial_channels = args.spatial_channels if hasattr(args, 'spatial_channels') else 2
        self.event_channels = args.event_channels if hasattr(args, 'event_channels') else 3
        self.enhanced_event_channels = 4 if hasattr(args, 'use_enhanced_features') and args.use_enhanced_features else 0
        self.total_event_channels = self.event_channels + self.enhanced_event_channels
        self.feature_dim = args.feature_dim if hasattr(args, 'feature_dim') else 126
        

        self.neuron_model = args.neuron_model if hasattr(args, 'neuron_model') else 'adaptive_lif'

        self.spatial_encoder = nn.Sequential(
            PyramidFeatureExtraction(self.spatial_channels, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, self.feature_dim, 1)
        )

        self.event_encoder = nn.Sequential(
            nn.Conv1d(self.total_event_channels, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            SpikingFeatureProcessor(64, neuron_model=self.neuron_model),
            nn.Conv1d(64, self.feature_dim, 1)
        )

        self.point_encoder = HierarchicalPointEncoder(
            in_channels=self.spatial_channels + self.total_event_channels,
            out_channels=self.feature_dim
        )


        self.spatial_feature_enhancer = DynamicFeatureEnhancement(self.feature_dim)
        self.event_feature_enhancer = DynamicFeatureEnhancement(self.feature_dim)

        self.cross_modal_fusion = CrossModalAdaptiveFusion(
            channels=self.feature_dim,
            num_heads=8,
            neuron_model=self.neuron_model
        )
        

        self.temporal_processor = nn.Sequential(
            MultiScaleTemporalEncoding(self.feature_dim),
            MultiHeadSpatioTemporalAttention(self.feature_dim, num_heads=8),
            AdaptiveTimeStep(self.feature_dim)
        )

        self.skeleton_constraint = SkeletonConstraintModule(
            num_joints=args.num_joints,
            feature_dim=self.feature_dim,
            use_spiking_gcn=True if self.neuron_model != 'lif' else False
        )

        self.global_feature_fusion = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim*2, 1),
            nn.BatchNorm2d(self.feature_dim*2),
            nn.GELU(),
            nn.Conv2d(self.feature_dim*2, self.feature_dim, 1),
            nn.BatchNorm2d(self.feature_dim)
        )

        self.header = Header(args)
        
    def forward(self, x):

        batch_size = x.shape[0]

        x_spatial = x[:, :, :self.spatial_channels].permute(0, 2, 1)
        

        if self.enhanced_event_channels > 0:

            x_event = x[:, :, self.spatial_channels:].permute(0, 2, 1)
        else:

            x_event = x[:, :, self.spatial_channels:self.spatial_channels+self.event_channels].permute(0, 2, 1)  # [B, 3, N]
            

        x_point = x.permute(0, 2, 1)  # [B, C, N]
        


        s_feat = self.spatial_encoder(x_spatial)  # [B, 126, N]
        s_feat_enhanced = self.spatial_feature_enhancer(s_feat)  # [B, 126, N]

        e_feat = self.event_encoder(x_event)  # [B, 126, N]
        e_feat_enhanced = self.event_feature_enhancer(e_feat)  # [B, 126, N]
        

        p_feat = self.point_encoder(x_point)  # [B, 126, N]
        

        modal_fused = self.cross_modal_fusion(s_feat_enhanced, e_feat_enhanced)  # [B, 126, N]
        

        fused = modal_fused + p_feat * 0.5  # [B, 126, N]
        

        fused = fused.unsqueeze(-1)  # [B, 126, N, 1]
        

        temporal = self.temporal_processor(fused)  # [B, 126, N, 1]
        

        global_features = self.global_feature_fusion(temporal)  # [B, 126, N, 1]
        

        features_for_skeleton = global_features.squeeze(-1)  # [B, 126, N]
        

        output_x, output_y, confidence = self.header(global_features)  # [B, num_joints, out_h/w]

        joints_pred = torch.zeros(batch_size, output_x.size(1), 2, device=x.device)
        joints_pred[:, :, 0] = output_x.mean(dim=2)
        joints_pred[:, :, 1] = output_y.mean(dim=2)

        skeleton_weights, bone_lengths = self.skeleton_constraint(features_for_skeleton, joints_pred)
        

        bone_consistency_loss = self._compute_bone_consistency_loss(bone_lengths)
        
        return output_x, output_y, confidence, bone_consistency_loss
    
    def _compute_bone_consistency_loss(self, bone_lengths):


        if not bone_lengths:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            

        std_sum = 0
        for length in bone_lengths:

            std = torch.std(length, dim=0)
            std_sum += std.mean()
            
        return std_sum / len(bone_lengths)
