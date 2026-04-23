from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from concurrent.futures import ThreadPoolExecutor
from torch.func import vmap, grad, functional_call

@dataclass
class VLMDistillConfig:
    proj_dim:        int   = 256
    block_size:      int   = 4096                   
    num_grad_layers: int   = 1             

def get_last_layer_id(model):
    layer_ids = []

    for n, _ in model.named_parameters():
        if "layers" in n:
            split_name = n.split(".")
            for i, name in enumerate(split_name):
                if "layers" in name:
                    layer_ids.append(int(split_name[i + 1]))
                    break

    return max(layer_ids)
            

def _sq_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        (a ** 2).sum(-1, keepdim=True)
        + (b ** 2).sum(-1, keepdim=True).T
        - 2.0 * (a @ b.T)
    ).clamp(min=0.0)

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

class VLMGradientExtractor:

    def __init__(self, num_layers: int, last_layer_id: int):
        self.num_layers = num_layers
        self.last_layer_id = last_layer_id

        self.extracted_layer_ids = ["layers." + str(last_layer_id - id) + "." for id in range(num_layers)]

    def _tail_params(self, model: nn.Module) -> List[nn.Parameter]:
        named = [(n, p) for n, p in model.named_parameters()
                if p.requires_grad and any(k in n for k in self.extracted_layer_ids)]

        params = []
        for _, p in reversed(named):
            params.append(p)
        return params

    def _per_sample_flat_grads(
        self,
        params: List[nn.Parameter],
        per_sample_loss: torch.Tensor,   # (b,)
        retain_graph: bool,
    ) -> torch.Tensor:
        """(b, Σ|θ|) per-sample gradient matrix — batched backward."""
        b = per_sample_loss.size(0)

        eye = torch.eye(b, device=per_sample_loss.device, dtype=per_sample_loss.dtype)

        batched_grads = torch.autograd.grad(
            outputs=per_sample_loss,
            inputs=params,
            grad_outputs=eye,            # (b, b) — batched seed vectors
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
            is_grads_batched=True,       # <-- key flag
        )

        parts = []
        for g, p in zip(batched_grads, params):
            if g is not None:
                parts.append(g.reshape(b, -1))          # (b, numel_i)
            else:
                parts.append(p.new_zeros(b, p.numel())) # (b, numel_i)

        return torch.cat(parts, dim=1)   # (b, Σ|θ|)

    def extract(
        self,
        model:            nn.Module,
        per_sample_loss:  torch.Tensor,  
    ):

        grad_param = self._tail_params(model)
        return self._per_sample_flat_grads(grad_param, per_sample_loss, retain_graph=False)

class GvendiTopologyExtract(nn.Module):
    def __init__(self, data_args, training_args, distiller):
        super().__init__()

        if dist.is_initialized():
            self.world_size   = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size   = 1
            self.process_rank = 0

        self.args      = training_args
        self.distiller = distiller
        self.last_layer_id = get_last_layer_id(distiller.teacher)
        cfg = self._build_gvendi_config(training_args)

        self.extractor = VLMGradientExtractor(cfg.num_grad_layers, self.last_layer_id)
        self.extracted_layer_ids = self.extractor.extracted_layer_ids

        teacher_dim = self._stream_dims(distiller.teacher, self.extractor)
        self.proj_T = RademacherProjection(teacher_dim, cfg.proj_dim, 10, cfg.block_size)
        self.io_executor = ThreadPoolExecutor(max_workers=4) 

        self.teacher_cache_dir = getattr(data_args, "teacher_cache_dir", None)
        
        for n, p in distiller.teacher.named_parameters():
            p.requires_grad_(False)

            for layer_name in self.extracted_layer_ids:
                if layer_name in n:
                    p.requires_grad_(True)

    @staticmethod
    def _build_gvendi_config(args) -> VLMDistillConfig:
        return VLMDistillConfig(
            proj_dim        = 512,
            block_size      = 4096,
            num_grad_layers = 1,
        )
    
    def _check_cached_teacher_list(
        self,
        sample_ids: List[str],
    ) -> bool:

        if self.teacher_cache_dir is None:
            return False
        
        for sid in sample_ids:
            path = os.path.join(self.teacher_cache_dir, f"{sid}.pt")
            if not os.path.isfile(path):
                return False
        return True

    @staticmethod
    def _stream_dims(
        model: nn.Module,
        extractor: VLMGradientExtractor,
    ):
        ps = extractor._tail_params(model)
        return max(sum(p.numel() for p in ps), 1)        

    def _dist_gather(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        bufs = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(bufs, t)
        bufs[self.process_rank] = t
        return torch.cat(bufs, dim=0)

    @staticmethod
    def _teacher_per_sample_signal(
        teacher_qry_reps: torch.Tensor,
        teacher_pos_reps: torch.Tensor,
    ) -> torch.Tensor:
        return ((teacher_qry_reps - teacher_pos_reps) ** 2).sum(dim=-1)

    def _save_teacher_cache(self, cache_dir, sample_ids, grad_teacher):
        os.makedirs(cache_dir, exist_ok=True)
        
        grad_teacher_cpu = grad_teacher.detach().cpu().half()

        def save_task(sid, data):
            cache_path = os.path.join(cache_dir, f"{sid}.pt")
            torch.save({"grad_teacher": data}, cache_path)

        for i, sid in enumerate(sample_ids):
            self.io_executor.submit(save_task, str(sid), grad_teacher_cpu[i])

    def forward(
        self,
        distiller,
        input_data: Dict,
    ) -> Dict[str, torch.Tensor]:

        teacher_model = distiller.teacher

        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]
    
        teacher_model.eval()

        with torch.enable_grad():
            t_qry_out = teacher_model.encode_input(teacher_qry_input)
            t_pos_out = teacher_model.encode_input(teacher_pos_input)

            t_qry_reps, *_ = t_qry_out
            t_pos_reps, *_ = t_pos_out

            t_per_sample = self._teacher_per_sample_signal(
                t_qry_reps, t_pos_reps
            )

            grad_teacher_raw = self.extractor.extract(
                teacher_model,
                t_per_sample,
            )

        grad_teacher = F.normalize(self.proj_T(grad_teacher_raw), dim=-1)

        if self.teacher_cache_dir is not None:
            self._save_teacher_cache(
                self.teacher_cache_dir,
                input_data.get("sample_ids", None),
                grad_teacher,
            )

        log = {
            "loss": torch.tensor(0.0),
        }
        return log