from torch import nn
import torch
import torch.nn.functional as F

class GVendiCodebook(nn.Module):
    def __init__(self, K: int, d: int):
        super().__init__()
        self.K = K
        self.d = d
        self.centroids = nn.Parameter(torch.randn(K, d), requires_grad=False)

    def forward(self, x):
        # Compute distances from x to all centroids
        distances = torch.cdist(x, self.centroids)
        # Return indices of the closest centroids
        return torch.argmin(distances, dim=-1)
