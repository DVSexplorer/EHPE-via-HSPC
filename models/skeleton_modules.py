import torch
import torch.nn as nn
from neuron_models import AdaptiveIzhikevichNeuron, AdaptiveLIFNeuron
from spikingjelly.clock_driven import neuron


class GraphConvolution(nn.Module):

    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj):

        
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        output = output + self.bias
        
        return nn.functional.relu(output)


class SpikingGraphConvolution(nn.Module):

    
    def __init__(self, in_features, out_features, neuron_model='adaptive_lif'):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        

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
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj, time_steps=5):

        

        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        output = output + self.bias

        output_t = output.unsqueeze(-1).repeat(1, 1, 1, time_steps)
        

        spike_output = self.neuron(output_t)
        

        return spike_output.mean(dim=-1)


class SkeletonConstraintModule(nn.Module):

    
    def __init__(self, num_joints, feature_dim=126, embedding_dim=32, use_spiking_gcn=True):
        super().__init__()
        self.num_joints = num_joints
        self.use_spiking_gcn = use_spiking_gcn
        

        self.register_buffer('adjacency_matrix', self._build_adjacency_matrix(num_joints))
        

        self.joint_embedding = nn.Parameter(torch.randn(num_joints, embedding_dim))
        

        if use_spiking_gcn:
            self.gcn_layers = nn.ModuleList([
                SpikingGraphConvolution(embedding_dim, embedding_dim * 2, neuron_model='adaptive_lif'),
                SpikingGraphConvolution(embedding_dim * 2, embedding_dim * 4, neuron_model='izhikevich'),
                SpikingGraphConvolution(embedding_dim * 4, embedding_dim * 2, neuron_model='adaptive_lif'),
                SpikingGraphConvolution(embedding_dim * 2, embedding_dim, neuron_model='izhikevich')
            ])
        else:
            self.gcn_layers = nn.ModuleList([
                GraphConvolution(embedding_dim, embedding_dim * 2),
                GraphConvolution(embedding_dim * 2, embedding_dim * 4),
                GraphConvolution(embedding_dim * 4, embedding_dim * 2),
                GraphConvolution(embedding_dim * 2, embedding_dim)
            ])
        

        self.skeleton_fusion = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        

        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        

        self.structure_encoder = nn.Sequential(
            nn.Linear(num_joints * 2, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def _build_adjacency_matrix(self, num_joints):
        """构建骨架邻接矩阵"""
        A = torch.zeros(num_joints, num_joints)
        

        if num_joints == 13:
            connections = [
                (0, 1), (1, 2), (2, 3),
                (3, 4), (4, 5), (5, 6),
                (3, 7), (7, 8), (8, 9),
                (2, 10), (10, 11),
                (2, 12), (12, 13)
            ]
        else:

            connections = [(i, i+1) for i in range(num_joints-1)]
            

        for i, j in connections:
            if i < num_joints and j < num_joints:
                A[i, j] = A[j, i] = 1
            

        A = A + torch.eye(num_joints)
        

        D = torch.sum(A, dim=1)
        D_sqrt_inv = torch.diag(torch.pow(D, -0.5))
        return D_sqrt_inv @ A @ D_sqrt_inv
        
    def forward(self, features, joints_pred=None):

        batch_size = features.shape[0]
        

        if features.dim() == 4:
            features = features.squeeze(-1)
            

        x = self.joint_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        

        for gcn in self.gcn_layers:
            if self.use_spiking_gcn and isinstance(gcn, SpikingGraphConvolution):

                x = gcn(x, self.adjacency_matrix, time_steps=5)
            else:

                x = gcn(x, self.adjacency_matrix)
            

        skeleton_features = self.skeleton_fusion(x)
        

        weights = self.weight_generator(skeleton_features)
        

        skeleton_weights = weights.permute(0, 2, 1)
        

        if joints_pred is not None:

            edges = []
            for i in range(self.adjacency_matrix.shape[0]):
                for j in range(i+1, self.adjacency_matrix.shape[0]):
                    if self.adjacency_matrix[i, j] > 0:
                        edges.append((i, j))
            

            bone_lengths = []
            bone_vectors = []
            
            for i, j in edges:
                if i < joints_pred.shape[1] and j < joints_pred.shape[1]:
                    bone = joints_pred[:, i, :] - joints_pred[:, j, :]
                    bone_vectors.append(bone)
                    length = torch.norm(bone, dim=1, keepdim=True)
                    bone_lengths.append(length)
            

            if bone_vectors:

                structure_input = torch.cat([bv.flatten(1) for bv in bone_vectors], dim=1)

                structure_features = self.structure_encoder(structure_input)
                

                structure_features = structure_features.unsqueeze(1)
                enhanced_weights = skeleton_weights * (1.0 + 0.1 * structure_features.mean(dim=2, keepdim=True))
                

                return enhanced_weights, bone_lengths
            

            return skeleton_weights, bone_lengths
            
        return skeleton_weights
