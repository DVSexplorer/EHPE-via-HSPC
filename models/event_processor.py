import numpy as np
import torch
from scipy import stats

class EnhancedEventProcessor:

    
    def __init__(self, spatial_bins=(346, 260), time_bins=10, use_enhanced_features=True):
        self.spatial_bins = spatial_bins
        self.time_bins = time_bins
        self.use_enhanced_features = use_enhanced_features
        
    def process_events(self, events):


        H, W = self.spatial_bins
        if self.use_enhanced_features:

            output = torch.zeros((9, H, W), dtype=torch.float32)
        else:

            output = torch.zeros((5, H, W), dtype=torch.float32)
        

        events_by_pixel = {}
        for x, y, t, p in events:
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                if (x, y) not in events_by_pixel:
                    events_by_pixel[(x, y)] = []
                events_by_pixel[(x, y)].append((t, p))
        

        for (x, y), pixel_events in events_by_pixel.items():

            times = np.array([t for t, _ in pixel_events])
            polarities = np.array([p for _, p in pixel_events])
            

            output[0, y, x] = x / W
            output[1, y, x] = y / H
            

            t_avg = np.mean(times)
            p_acc = np.sum(polarities)
            event_cnt = len(pixel_events)
            

            output[2, y, x] = t_avg / np.max(times) if np.max(times) > 0 else 0
            if self.use_enhanced_features:

                t_std = np.std(times) if len(times) > 1 else 0
                output[3, y, x] = t_std / (np.max(times) - np.min(times) + 1e-6) if len(times) > 1 else 0
                

                p_pos = np.sum(polarities == 1)
                p_neg = np.sum(polarities == 0)
                p_ratio = p_pos / (p_pos + p_neg + 1e-6)
                output[4, y, x] = p_acc
                output[5, y, x] = p_ratio
                

                output[6, y, x] = np.log1p(event_cnt) / 10.0
                

                if event_cnt > 5 and len(set(times)) > 1:

                    sorted_indices = np.argsort(times)
                    time_norm = (times[sorted_indices] - times[sorted_indices[0]]) / (times[sorted_indices[-1]] - times[sorted_indices[0]] + 1e-6)
                    

                    neighbors = self._get_neighbors(events, x, y)
                    if neighbors:
                        neighbor_xs = [nx for nx, _, _ in neighbors]
                        neighbor_ys = [ny for _, ny, _ in neighbors]
                        neighbor_ts = [nt for _, _, nt in neighbors]
                        

                        if len(neighbor_xs) > 1:
                            try:
                                slope_x, _, _, _, _ = stats.linregress(neighbor_ts, neighbor_xs)
                                output[7, y, x] = np.tanh(slope_x)
                            except:
                                output[7, y, x] = 0
                                

                        if len(neighbor_ys) > 1:
                            try:
                                slope_y, _, _, _, _ = stats.linregress(neighbor_ts, neighbor_ys)
                                output[8, y, x] = np.tanh(slope_y)
                            except:
                                output[8, y, x] = 0
            else:

                output[3, y, x] = p_acc
                output[4, y, x] = np.log1p(event_cnt) / 10.0
        
        return output
    
    def _get_neighbors(self, events, x, y, radius=3, max_dt=1000):

        neighbors = []
        
        for ex, ey, et, _ in events:
            dx, dy = ex - x, ey - y
            dist = np.sqrt(dx**2 + dy**2)
            if 0 < dist <= radius:
                neighbors.append((ex, ey, et))
                
        return neighbors
    
    def create_time_surfaces(self, events, decay_factor=1000):

        H, W = self.spatial_bins
        time_surface = torch.zeros((2, H, W), dtype=torch.float32)
        last_timestamp = torch.zeros((2, H, W), dtype=torch.float32)
        

        events = sorted(events, key=lambda e: e[2])
        
        for x, y, t, p in events:
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                p_idx = int(p)

                last_timestamp[p_idx, y, x] = t
                

                for i in range(H):
                    for j in range(W):
                        for k in range(2):
                            if last_timestamp[k, i, j] > 0:
                                dt = t - last_timestamp[k, i, j]
                                time_surface[k, i, j] = np.exp(-dt / decay_factor)
        
        return time_surface
    
    def create_voxel_grid(self, events, bins=None):

        if bins is None:
            bins = [self.spatial_bins[0], self.spatial_bins[1], self.time_bins]
            
        H, W, T = bins
        voxel_grid = torch.zeros((2, H, W, T), dtype=torch.float32)
        

        if events:
            t_min = min(e[2] for e in events)
            t_max = max(e[2] for e in events)
            dt = (t_max - t_min) / T if t_max > t_min else 1.0
        else:
            return voxel_grid
            

        for x, y, t, p in events:
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                t_idx = min(int((t - t_min) / dt), T - 1)
                p_idx = int(p)
                voxel_grid[p_idx, y, x, t_idx] += 1
                
        return voxel_grid
