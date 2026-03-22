"""
graph_fusion.py
---------------
Multi-camera Graph Neural Network fusion.
Each camera = node, road connections = edges.
GCN propagates traffic predictions across the road network.

Built using pure Python/torch — no numpy bridge (safe on NumPy 2.x).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        A   = adj + torch.eye(adj.size(0), device=adj.device)
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        A_n = A / (deg * deg.t()).sqrt()
        return F.relu(torch.bmm(
            A_n.unsqueeze(0).expand(x.size(0), -1, -1),
            self.W(x)
        ))


class MultiCameraGNN(nn.Module):
    """
    Two-layer GCN fusing predictions from N cameras.
    Input  : (batch, N, in_features)  — one feature per camera
    Output : (batch, N, 1)            — fused prediction per camera
    """
    def __init__(self, in_features=1, hidden=16, n_cameras=4, dropout=0.2):
        super().__init__()
        self.gc1     = GraphConvLayer(in_features, hidden)
        self.gc2     = GraphConvLayer(hidden, hidden)
        self.fc_out  = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.gc1(x, adj))
        h = self.gc2(h, adj)
        return self.fc_out(h)


def build_adjacency(locations: List[dict], max_distance_km: float = 2.0) -> torch.Tensor:
    """
    Build adjacency matrix from GPS locations using haversine distance.
    Uses pure Python — no numpy required.
    """
    N = len(locations)

    def haversine(a, b):
        R    = 6371.0
        lat1 = math.radians(a['lat']); lon1 = math.radians(a['lon'])
        lat2 = math.radians(b['lat']); lon2 = math.radians(b['lon'])
        dlat = lat2 - lat1; dlon = lon2 - lon1
        x    = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, x))))

    flat = [
        1.0 if (i != j and haversine(locations[i], locations[j]) <= max_distance_km) else 0.0
        for i in range(N) for j in range(N)
    ]
    return torch.tensor(flat, dtype=torch.float32).view(N, N)


def run_gnn_fusion(camera_densities: list, locations: list,
                   max_distance_km: float = 2.0) -> list:
    """
    Convenience function: run GNN on a list of scalar densities.

    Parameters
    ----------
    camera_densities : list of floats [0..1], one per camera
    locations        : list of dicts with 'lat', 'lon'
    max_distance_km  : road connection threshold

    Returns
    -------
    fused_densities : list of floats — GNN-adjusted predictions
    """
    N   = len(camera_densities)
    adj = build_adjacency(locations, max_distance_km)
    x   = torch.tensor([[d] for d in camera_densities],
                        dtype=torch.float32).unsqueeze(0)  # (1, N, 1)
    gnn = MultiCameraGNN(in_features=1, hidden=16, n_cameras=N)
    with torch.no_grad():
        out = gnn(x, adj).squeeze(0).squeeze(-1)          # (N,)
    return [float(v) for v in out]