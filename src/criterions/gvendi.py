from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

@dataclass
class VLMDistillConfig:
    proj_dim:        int   = 512
    block_size:      int   = 4096
    K_codebook:      int   = 60
    sinkhorn_reg:    float = 0.05
    sinkhorn_iters:  int   = 100
    lam_ot:          float = 1.2
    lam_commit:      float = 1.0
    num_grad_layers: int   = 1

def _sq_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        (a ** 2).sum(-1, keepdim=True)
        + (b ** 2).sum(-1, keepdim=True).T
        - 2.0 * (a @ b.T)
    ).clamp(min=0.0)

def get_last_layer_id(model):
    layer_ids = []

    for n, params in model.named_parameters():
        if "layers" in n and params.requires_grad:
            split_name = n.split(".")
            for i, name in enumerate(split_name):
                if "layers" in name:
                    layer_ids.append(int(split_name[i + 1]))
                    break
    print("Extracting gradients from layers:", sorted(set(layer_ids)))
    return max(layer_ids)


def _sinkhorn_log(cost: torch.Tensor, reg: float, iters: int) -> torch.Tensor:
    b, K = cost.shape
    log_a = torch.full((b,), -math.log(b), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((K,), -math.log(K), device=cost.device, dtype=cost.dtype)
    log_u = torch.zeros(b, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(K, device=cost.device, dtype=cost.dtype)
    M = cost / reg
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_v.unsqueeze(0) - M, dim=1)
        log_v = log_b - torch.logsumexp(log_u.unsqueeze(1) - M, dim=0)
    return (log_u.unsqueeze(1) + log_v.unsqueeze(0) - M).exp()


class RademacherProjection(nn.Module):
    def __init__(self, param_dim: int, proj_dim: int = 512,
                 seed: int = 42, block_size: int = 4096):
        super().__init__()
        self.param_dim  = param_dim
        self.proj_dim   = proj_dim
        self.seed       = seed
        self.block_size = block_size
        self.scale      = proj_dim ** -0.5

    @torch.no_grad()
    def forward(self, g: torch.Tensor) -> torch.Tensor:
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


class GVendiCodebook(nn.Module):
    def __init__(self, K: int, d: int):
        super().__init__()
        self.centroids = nn.Parameter(F.normalize(torch.randn(K, d), dim=-1))

    @property
    def normalized(self) -> torch.Tensor:
        return F.normalize(self.centroids, dim=-1)


class LinearProjector(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.w.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)

class VLMGradientExtractor:

    def __init__(self, num_layers: int, last_layer_id: int):
        self.num_layers   = num_layers
        self.last_layer_id = last_layer_id
        self.extracted_layer_ids = [
            f"layers.{last_layer_id - i}."
            for i in range(num_layers)
        ]

    def _tail_params(self, model: nn.Module) -> List[nn.Parameter]:
        return [
            p for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in self.extracted_layer_ids)
        ]

    def _per_sample_flat_grads(
        self,
        params: List[nn.Parameter],
        per_sample_loss: torch.Tensor,   # (b,)
        retain_graph: bool = False,
    ) -> torch.Tensor:                   # (b, ΣD)
        b   = per_sample_loss.size(0)
        eye = torch.eye(b, device=per_sample_loss.device, dtype=per_sample_loss.dtype)

        batched_grads = torch.autograd.grad(
            outputs=per_sample_loss,
            inputs=params,
            grad_outputs=eye,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
            is_grads_batched=True,
        )
        parts = []
        for g, p in zip(batched_grads, params):
            parts.append(g.reshape(b, -1) if g is not None
                         else p.new_zeros(b, p.numel()))
        return torch.cat(parts, dim=1)   # (b, ΣD)

    def extract(
        self,
        model: nn.Module,
        per_sample_loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        params = self._tail_params(model)
        return self._per_sample_flat_grads(params, per_sample_loss, retain_graph)

class GVendiVLMCriterion(nn.Module):

    def __init__(self, data_args, training_args, distiller):
        super().__init__()

        if dist.is_initialized():
            self.world_size   = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size   = 1
            self.process_rank = 0

        self.args = training_args

        self.w_rkd    = getattr(training_args, "w_rkd_loss",    0.5)
        self.w_gvendi = getattr(training_args, "w_gvendi_loss", 1.0)

        cfg = self._build_config(training_args)
        self.cfg = cfg

        self.last_layer_id = get_last_layer_id(distiller.student)

        self.extractor = VLMGradientExtractor(cfg.num_grad_layers, self.last_layer_id)
        self.extracted_layer_ids = self.extractor.extracted_layer_ids

        student_dim = self._probe_dim(distiller.student, self.extractor)
        self.proj_S = RademacherProjection(student_dim, cfg.proj_dim, seed=20,
                                           block_size=cfg.block_size)
        self.pj     = LinearProjector(cfg.proj_dim)
        self.book = GVendiCodebook(cfg.K_codebook, cfg.proj_dim)

        phase1_ckpt = getattr(training_args, "gvendi_phase1_ckpt", None)
        if phase1_ckpt and os.path.isfile(phase1_ckpt):
            ckpt = torch.load(phase1_ckpt, map_location="cpu", weights_only=True)
            self.book.load_state_dict(ckpt["book_state_dict"])
            print(f"[GVendiVLMCriterion] Loaded phase-1 codebook from {phase1_ckpt}")
        else:
            print("[GVendiVLMCriterion] No phase-1 checkpoint found — codebook trained from scratch.")

        self.teacher_cache_dir = getattr(data_args, "teacher_cache_dir", None)
        self.io_executor = ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def _build_config(args) -> VLMDistillConfig:
        return VLMDistillConfig(
            proj_dim        = getattr(args, "gvendi_proj_dim",       512),
            block_size      = getattr(args, "gvendi_block_size",     4096),
            K_codebook      = getattr(args, "gvendi_K_codebook",      60),
            sinkhorn_reg    = getattr(args, "gvendi_sinkhorn_reg",   0.05),
            sinkhorn_iters  = getattr(args, "gvendi_sinkhorn_iters",  100),
            lam_ot          = getattr(args, "gvendi_lam_ot",          1.2),
            lam_commit      = getattr(args, "gvendi_lam_commit",      1.0),
            num_grad_layers = getattr(args, "gvendi_num_grad_layers",   1),
        )

    @staticmethod
    def _probe_dim(model: nn.Module, extractor: VLMGradientExtractor) -> int:
        ps = extractor._tail_params(model)
        return max(sum(p.numel() for p in ps), 1)

    def _dist_gather(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        bufs = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(bufs, t)
        bufs[self.process_rank] = t
        return torch.cat(bufs, dim=0)

    def _check_cached_teacher_exists(
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

    def _load_cached_teacher(
        self,
        sample_ids: List[str],
        device: torch.device,
    ) -> torch.Tensor:                          

        grads = []
        for sid in sample_ids:
            path   = os.path.join(self.teacher_cache_dir, f"{sid}.pt")
            cached = torch.load(path, map_location="cpu", weights_only=True)
            grads.append(cached["grad_teacher"])
        return torch.stack(grads).to(device=device, dtype=torch.float32)  # (b, d)

    @staticmethod
    def _student_per_sample_signal(
        s_qry: torch.Tensor,
        t_qry: torch.Tensor,
    ) -> torch.Tensor:                          # (b,)
        d = min(s_qry.shape[-1], t_qry.shape[-1])
        return ((s_qry[..., :d] - t_qry.detach()[..., :d]) ** 2).sum(dim=-1)

    @staticmethod
    def _pairwise_dist(x: torch.Tensor) -> torch.Tensor:
        n = (x ** 2).sum(1, keepdim=True)
        return (n + n.T - 2.0 * x @ x.T).clamp(min=0.0)

    def _rkd_distance(self, s_qry, s_pos, t_qry, t_pos) -> torch.Tensor:
        ds = self._pairwise_dist(torch.cat([s_qry, s_pos]))
        dt = self._pairwise_dist(torch.cat([t_qry, t_pos]))
        mask = torch.triu(torch.ones_like(ds, dtype=torch.bool), diagonal=1)
        ds, dt = ds[mask], dt[mask]
        ds = ds / (ds.mean().detach() + 1e-8)
        dt = dt / (dt.mean().detach() + 1e-8)
        diff = (ds - dt).abs()
        return torch.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5).mean()

    def _rkd_angle(self, s_qry, s_pos, t_qry, t_pos) -> torch.Tensor:
        def _angles(x):
            d = x.unsqueeze(0) - x.unsqueeze(1)
            e = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
            return torch.einsum("ijd,kjd->ijk", e, e)
        ps = _angles(torch.cat([s_qry, s_pos]))
        pt = _angles(torch.cat([t_qry, t_pos]))
        n  = ps.shape[0]
        idx  = torch.arange(n, device=ps.device)
        mask = torch.ones(n, n, n, dtype=torch.bool, device=ps.device)
        mask[idx, idx, :] = mask[idx, :, idx] = mask[:, idx, idx] = False
        diff = (ps[mask] - pt[mask]).abs()
        return torch.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5).mean()

    def forward(self, distiller, input_data: Dict) -> Dict[str, torch.Tensor]:
        cfg           = self.cfg
        student_model = distiller.student
        teacher_model = distiller.teacher

        student_qry_input = input_data["student_inputs"]["qry"]
        student_pos_input = input_data["student_inputs"]["pos"]
        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]
        device = student_qry_input["input_ids"].device

        sample_ids  = input_data["sample_ids"]
        
        if not self._check_cached_teacher_exists(sample_ids):
            return {"skip_batch": True}
        
        with torch.no_grad():
            teacher_model.eval()
            t_qry_reps, *_ = teacher_model.encode_input(teacher_qry_input)
            t_pos_reps, *_ = teacher_model.encode_input(teacher_pos_input)

        s_qry_reps, *_ = student_model.encode_input(student_qry_input)
        s_pos_reps, *_ = student_model.encode_input(student_pos_input)

        all_s_qry = self._dist_gather(s_qry_reps) if self.world_size > 1 else s_qry_reps
        all_s_pos = self._dist_gather(s_pos_reps) if self.world_size > 1 else s_pos_reps

        scores  = distiller.student.compute_similarity(all_s_qry, all_s_pos)
        scores  = scores.view(all_s_qry.size(0), -1)
        targets = torch.arange(scores.size(0), device=device)
        targets *= all_s_qry.size(0) // all_s_pos.size(0)
        L_contrastive = nn.CrossEntropyLoss()(scores / distiller.temperature, targets)

        L_rkd = (
            self._rkd_distance(s_qry_reps, s_pos_reps, t_qry_reps, t_pos_reps)
            + self._rkd_angle  (s_qry_reps, s_pos_reps, t_qry_reps, t_pos_reps)
        ) / 2.0
        
        gt_projected = self._load_cached_teacher(sample_ids, device)  

        # for n, p in student_model.named_parameters():
        #     print(n)

        s_per_sample = self._student_per_sample_signal(s_qry_reps, t_qry_reps)

        # with torch.enable_grad():
        gs_raw = self.extractor.extract(
            student_model,
            s_per_sample,
            retain_graph=True,
        )                                            

        gs_projected = F.normalize(self.proj_S(gs_raw), dim=-1)   
        
        gs_aligned   = self.pj.to(device)(gs_projected)                       

        C     = self.book.normalized.to(device)                     # (K, d)
        cost  = _sq_euclidean(gt_projected, C)                     # (b, K)
        with torch.no_grad():
            gamma  = _sinkhorn_log(cost, cfg.sinkhorn_reg, cfg.sinkhorn_iters)
            k_star = cost.argmin(dim=1)                            # (b,)
        L_OT = (gamma * cost).sum()

        centroid_targets = C.detach()[k_star]                       # (b, d)
        L_commit = F.mse_loss(gs_aligned, centroid_targets, reduction="sum")

        L_gvendi = cfg.lam_ot * L_OT + cfg.lam_commit * L_commit

        total_loss = (
            L_contrastive
            + self.w_rkd    * L_rkd
            + self.w_gvendi * L_gvendi
        )

        return {
            "loss":              total_loss,
            "contrastive_loss":  L_contrastive,
            "rkd_loss":          L_rkd,
            "gvendi_ot":         L_OT,
            "gvendi_commit":     L_commit,
            "gvendi_total":      L_gvendi,
        }