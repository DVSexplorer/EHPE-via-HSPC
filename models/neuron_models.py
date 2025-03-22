import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate


class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.gt(x, 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        return grad_input

class AdaptiveIzhikevichNeuron(nn.Module):

    
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0, dt=1.0, spike_threshold=30.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))
        self.d = nn.Parameter(torch.tensor(d))
        self.dt = dt
        self.spike_threshold = spike_threshold
        

        self.register_buffer('v', None)
        self.register_buffer('u', None)
        
    def reset_state(self, batch_size, *size):

        if size:
            self.v = torch.zeros(batch_size, *size, device=self.a.device)
            self.u = torch.zeros(batch_size, *size, device=self.a.device)
        else:
            self.v = None
            self.u = None
    
    def forward(self, x):


        orig_shape = x.shape
        

        if len(orig_shape) == 3:
            x = x.unsqueeze(-1)
        
        batch_size, channels, n_points, time_steps = x.shape
        

        if self.v is None or self.v.shape[0] != batch_size:
            self.reset_state(batch_size, channels, n_points)

        spikes = torch.zeros_like(x)
        mem_potentials = torch.zeros_like(x)
        

        v = self.v
        u = self.u
        

        for t in range(time_steps):

            dv = (0.04 * v**2 + 5 * v + 140 - u + x[..., t])
            v = v + self.dt * dv
            du = self.a * (self.b * v - u)
            u = u + self.dt * du

            spike = (v >= self.spike_threshold).float()
            

            v = v * (1.0 - spike) + self.c * spike
            u = u + self.d * spike
            

            spikes[..., t] = spike
            mem_potentials[..., t] = v
        

        self.v = v.detach()
        self.u = u.detach()
        

        if len(orig_shape) == 3:
            spikes = spikes.squeeze(-1)
            
        return spikes


class AdaptiveLIFNeuron(nn.Module):

    
    def __init__(self, tau_mem=20.0, tau_thresh=60.0, thresh_rest=1.0, reset_mechanism='subtract'):
        super().__init__()
        self.tau_mem = nn.Parameter(torch.tensor(tau_mem))
        self.tau_thresh = nn.Parameter(torch.tensor(tau_thresh))
        self.thresh_rest = thresh_rest
        self.reset_mechanism = reset_mechanism
        

        self.register_buffer('mem', None)
        self.register_buffer('threshold', None)
        
    def reset_state(self, batch_size, *size):

        if size:
            self.mem = torch.zeros(batch_size, *size, device=self.tau_mem.device)
            self.threshold = torch.ones(batch_size, *size, device=self.tau_mem.device) * self.thresh_rest
        else:
            self.mem = None
            self.threshold = None
    
    def forward(self, x):


        orig_shape = x.shape
        

        if len(orig_shape) == 3:
            x = x.unsqueeze(-1)
        
        batch_size, channels, n_points, time_steps = x.shape
        

        if self.mem is None or self.mem.shape[0] != batch_size:
            self.reset_state(batch_size, channels, n_points)
        

        spikes = torch.zeros_like(x)
        

        mem = self.mem
        threshold = self.threshold
        

        spike_function = surrogate.ATan()

        for t in range(time_steps):

            mem = mem * torch.exp(-1.0 / self.tau_mem) + x[..., t]
            

            spike = spike_function(mem - threshold)
            

            if self.reset_mechanism == 'subtract':
                mem = mem - threshold * spike
            else:
                mem = mem * (1.0 - spike)
            

            threshold = threshold * torch.exp(-1.0 / self.tau_thresh) + spike * self.thresh_rest
            

            spikes[..., t] = spike
        

        self.mem = mem.detach()
        self.threshold = threshold.detach()
        

        if len(orig_shape) == 3:
            spikes = spikes.squeeze(-1)
            
        return spikes


def get_neuron(neuron_type='lif', **kwargs):
    if neuron_type == 'adaptive_lif':
        return AdaptiveLIFNeuron(**kwargs)
    elif neuron_type == 'izhikevich':
        return AdaptiveIzhikevichNeuron(**kwargs)
    else:
        return neuron.LIFNode(**kwargs)
