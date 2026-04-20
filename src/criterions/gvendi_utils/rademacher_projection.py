import torch
import torch.nn as nn



class RademacherProjection(nn.Module):
    def __init__(
        self,
        param_dim:  int,
        proj_dim:   int  = 256,
        seed:       int  = 42,
        block_size: int  = 4096,
    ):
        super().__init__()
        self.param_dim  = param_dim
        self.proj_dim   = proj_dim
        self.seed       = seed
        self.block_size = block_size
        self.scale      = proj_dim ** -0.5

    @torch.no_grad()
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """g: (B, param_dim) → projected: (B, proj_dim)"""
        B, D = g.shape
        out  = torch.zeros(B, self.proj_dim, device=g.device, dtype=g.dtype)
        rng  = torch.Generator(device=g.device)
        for start in range(0, D, self.block_size):
            end = min(start + self.block_size, D)
            rng.manual_seed(self.seed + start)
            Pi  = (torch.randint(0, 2, (end - start, self.proj_dim),
                                 generator=rng, device=g.device,
                                 dtype=g.dtype) * 2 - 1)
            out += g[:, start:end] @ Pi
        return out * self.scale