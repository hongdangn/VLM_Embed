from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from src.utils import print_rank

@dataclass
class GVendiLossComponents:
    contrastive:       float = 0.0
    rkd_distance:      float = 0.0
    rkd_angle:         float = 0.0
    ot_fusion:         float = 0.0
    align_fusion:      float = 0.0
    cross_modal_ot:    float = 0.0  
    total:             float = 0.0


@dataclass
class VLMDistillConfig:
    proj_dim:        int   = 256
    block_size:      int   = 4096          

    K_fusion:        int   = 60
    K_cross:         int   = 40            

    sinkhorn_reg:    float = 0.05          
    sinkhorn_iters:  int   = 100           

    lam_fusion:      float = 1.2

    lam_cross_ot:    float = 0.5           

    num_grad_layers: int   = 1             

def _sinkhorn_log(
    cost:  torch.Tensor,   
    reg:   float,          
    iters: int,
) -> torch.Tensor:
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


class GVendiCodebook(nn.Module):
    def __init__(self, K: int, d: int):
        super().__init__()
        self.K = K
        self.d = d
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


def wasserstein_codebook_loss(
    g_teacher:  torch.Tensor,  
    book:       GVendiCodebook,
    reg:        float,
    iters:      int,
) -> torch.Tensor:
    C    = book.normalized                           # (K, d)
    cost = _sq_euclidean(g_teacher, C)              # (b, K)
    with torch.no_grad():
        gamma = _sinkhorn_log(cost, reg, iters)     # (b, K)  — detached weights
    return (gamma * cost).sum()

def commitment_loss(
    g_student:  torch.Tensor,   # (b, d) student grads after P_φ
    g_teacher:  torch.Tensor,   # (b, d) teacher grads (for assignment lookup)
    book:       GVendiCodebook,
) -> torch.Tensor:
    C      = book.normalized.detach()               # (K, d)  — stop-gradient
    with torch.no_grad():
        k_star = _sq_euclidean(g_teacher, C).argmin(dim=1)   # (b,)
    return F.mse_loss(g_student, C[k_star], reduction="sum")

class CrossModalCodebook(nn.Module):

    def __init__(self, K: int, d: int, alpha: float = 0.5):
        super().__init__()
        self.K     = K
        self.alpha = alpha
        self.v_centroids = nn.Parameter(F.normalize(torch.randn(K, d), dim=-1))
        self.t_centroids = nn.Parameter(F.normalize(torch.randn(K, d), dim=-1))

    @property
    def v_norm(self) -> torch.Tensor:
        return F.normalize(self.v_centroids, dim=-1)

    @property
    def t_norm(self) -> torch.Tensor:
        return F.normalize(self.t_centroids, dim=-1)

    def cost_matrix(
        self, gv: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.alpha       * _sq_euclidean(gv, self.v_norm)
            + (1-self.alpha) * _sq_euclidean(gt, self.t_norm)
        )


def cross_modal_ot_loss(
    gt_v:        torch.Tensor,       
    gt_t:        torch.Tensor,       
    gs_v_proj:   torch.Tensor,       
    gs_t_proj:   torch.Tensor,       
    cross_book:  CrossModalCodebook,
    reg:         float,
    iters:       int,
) -> torch.Tensor:
    cost_T = cross_book.cost_matrix(gt_v, gt_t)              # (b, K)
    with torch.no_grad():
        gamma  = _sinkhorn_log(cost_T, reg, iters)           # (b, K)
        k_star = cost_T.argmin(dim=1)                        # (b,)
    loss_ot = (gamma * cost_T).sum()

    v_tgt = cross_book.v_norm.detach()[k_star]               # (b, d)
    t_tgt = cross_book.t_norm.detach()[k_star]               # (b, d)
    loss_commit = (
        cross_book.alpha       * F.mse_loss(gs_v_proj, v_tgt, reduction="sum")
        + (1-cross_book.alpha) * F.mse_loss(gs_t_proj, t_tgt, reduction="sum")
    )
    return loss_ot + loss_commit


class VLMGradientExtractor:

    VISUAL_KEYS = (
        "vision_tower", "visual_encoder", "image_encoder",
        "patch_embed", "vit.", "clip_vision",
    )
    TEXT_KEYS = (
        "language_model", "llm.", "text_model.",
        "embed_tokens", "lm_head", "transformer.",
    )
    FUSION_KEYS = (
        "mm_projector", "vision_proj", "connector",
        "cross_attention", "perceiver", "aligner",
    )

    def __init__(self, num_layers: int = 2):
        self.num_layers = num_layers

    def _tail_params(
        self, model: nn.Module, keys: Tuple[str, ...]
    ) -> List[nn.Parameter]:
        named   = [(n, p) for n, p in model.named_parameters()
                   if p.requires_grad and any(k in n for k in keys)]
        seen, ps = set(), []
        for name, p in reversed(named):
            blk = ".".join(name.split(".")[:4])
            seen.add(blk)
            if len(seen) > self.num_layers:
                break
            ps.append(p)
        if not ps:                        # fallback: any matching param
            ps = [p for _, p in named[-10:]]
        return ps

    def _per_sample_flat_grads(
        self,
        params:       List[nn.Parameter],
        per_sample_loss: torch.Tensor,   # (b,)
        retain_graph: bool,
    ) -> torch.Tensor:
        """(b, Σ|θ|) per-sample gradient matrix for `params`."""
        rows = []
        for loss_i in per_sample_loss:
            grads = torch.autograd.grad(
                loss_i, params,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=True,
            )
            flat = torch.cat([
                g.reshape(-1) if g is not None
                else p.new_zeros(p.numel())
                for g, p in zip(grads, params)
            ])
            rows.append(flat)
        return torch.stack(rows)   # (b, D)

    def extract(
        self,
        model:            nn.Module,
        per_sample_loss:  torch.Tensor,   # (b,)  un-reduced
        retain_graph:     bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p_vis = self._tail_params(model, self.VISUAL_KEYS)
        p_txt = self._tail_params(model, self.TEXT_KEYS)
        p_fus = self._tail_params(model, self.FUSION_KEYS)

        all_p = [p for p in model.parameters() if p.requires_grad]
        n     = len(all_p)
        if not p_vis: p_vis = all_p[: n // 3]
        if not p_txt: p_txt = all_p[n // 3: 2 * n // 3]
        if not p_fus: p_fus = all_p[2 * n // 3:]

        g_v = self._per_sample_flat_grads(p_vis, per_sample_loss, retain_graph=True)
        g_t = self._per_sample_flat_grads(p_txt, per_sample_loss, retain_graph=True)
        g_f = self._per_sample_flat_grads(p_fus, per_sample_loss, retain_graph=retain_graph)
        return g_v, g_t, g_f

class VLMDistillGVendi(nn.Module):

    def __init__(
        self,
        teacher_visual_dim: int,
        teacher_text_dim:   int,
        teacher_fusion_dim: int,
        student_visual_dim: int,
        student_text_dim:   int,
        student_fusion_dim: int,
        cfg: VLMDistillConfig,
    ):
        super().__init__()
        self.cfg = cfg
        d        = cfg.proj_dim

        # ── Rademacher projectors (6 total, one per stream × model side) ──
        self.proj_T_v = RademacherProjection(teacher_visual_dim,  d, 10, cfg.block_size)
        self.proj_T_t = RademacherProjection(teacher_text_dim,    d, 20, cfg.block_size)
        self.proj_T_f = RademacherProjection(teacher_fusion_dim,  d, 30, cfg.block_size)
        self.proj_S_v = RademacherProjection(student_visual_dim,  d, 40, cfg.block_size)
        self.proj_S_t = RademacherProjection(student_text_dim,    d, 50, cfg.block_size)
        self.proj_S_f = RademacherProjection(student_fusion_dim,  d, 60, cfg.block_size)

        self.book_f = GVendiCodebook(cfg.K_fusion,  d)

        self.pj_v = LinearProjector(d)
        self.pj_t = LinearProjector(d)
        self.pj_f = LinearProjector(d)

        self.cross_book     = CrossModalCodebook(cfg.K_cross, d)          # [1]
        # self.adapt_w        = AdaptiveModalityWeighter(cfg.ema_decay)      # [5]

        self.extractor = VLMGradientExtractor(cfg.num_grad_layers)

    def _project_bundle(
        self,
        g_v: torch.Tensor,
        g_t: torch.Tensor,
        g_f: torch.Tensor,
        is_teacher: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pv = self.proj_T_v if is_teacher else self.proj_S_v
        pt = self.proj_T_t if is_teacher else self.proj_S_t
        pf = self.proj_T_f if is_teacher else self.proj_S_f
        return (
            F.normalize(pv(g_v), dim=-1),
            F.normalize(pt(g_t), dim=-1),
            F.normalize(pf(g_f), dim=-1),
        )


    def forward(
        self,
        t_per_sample_loss: torch.Tensor,   # (b,) teacher per-sample signal
        s_per_sample_loss: torch.Tensor,   # (b,) student per-sample signal
        teacher_model:     nn.Module,
        student_model:     nn.Module,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, Optional[GVendiLossComponents]]:
        cfg = self.cfg

        # ── Extract raw per-sample gradients ──────────────────────────────
        with torch.enable_grad():
            gt_v_raw, gt_t_raw, gt_f_raw = self.extractor.extract(
                teacher_model, t_per_sample_loss, retain_graph=True)
            gs_v_raw, gs_t_raw, gs_f_raw = self.extractor.extract(
                student_model, s_per_sample_loss, retain_graph=True)

        gt_v_raw = gt_v_raw.detach()
        gt_t_raw = gt_t_raw.detach()
        gt_f_raw = gt_f_raw.detach()

        gt_v, gt_t, gt_f = self._project_bundle(gt_v_raw, gt_t_raw, gt_f_raw, True)
        gs_v, gs_t, gs_f = self._project_bundle(gs_v_raw, gs_t_raw, gs_f_raw, False)

        # self.adapt_w.update(gs_v, gs_t)
        # w_v, w_t = self.adapt_w.weights()

        gs_v_p = self.pj_v(gs_v)
        gs_t_p = self.pj_t(gs_t)
        gs_f_p = self.pj_f(gs_f)

        # ── Base paper: Stage-1 (Sinkhorn OT) + Stage-2 (commitment) ──────
        # L_OT_v   = wasserstein_codebook_loss(gt_v, self.book_v,
        #                                      cfg.sinkhorn_reg, cfg.sinkhorn_iters)
        # L_OT_t   = wasserstein_codebook_loss(gt_t, self.book_t,
        #                                      cfg.sinkhorn_reg, cfg.sinkhorn_iters)
        L_OT_f   = wasserstein_codebook_loss(gt_f, self.book_f,
                                             cfg.sinkhorn_reg, cfg.sinkhorn_iters)
        # L_cm_v   = commitment_loss(gs_v_p, gt_v, self.book_v)
        # L_cm_t   = commitment_loss(gs_t_p, gt_t, self.book_t)
        L_cm_f   = commitment_loss(gs_f_p, gt_f, self.book_f)

        # stream_v = w_v * cfg.lam_visual * (L_OT_v + L_cm_v)
        # stream_t = w_t * cfg.lam_text   * (L_OT_t + L_cm_t)
        stream_f =       cfg.lam_fusion * (L_OT_f + L_cm_f)

        L_cross = cfg.lam_cross_ot * cross_modal_ot_loss(
            gt_v, gt_t, gs_v_p, gs_t_p,
            self.cross_book, cfg.sinkhorn_reg, cfg.sinkhorn_iters)

        # L_aug  = cfg.lam_augment  * self.augmenter(gs_v_p, gs_t_p,
        #                                             self.book_v, self.book_t)

        # L_spec = cfg.lam_spectral * self.spectral_reg(gs_v_p, gs_t_p)

        # L_cons = cfg.lam_consensus * cross_modal_consensus_loss(
        #     gs_v_p, gs_t_p, gt_v, gt_t)

        # L_proto = cfg.lam_prototype * self.proto_pairs(
        #     gt_v, gt_t, gs_v_p, gs_t_p)

        # total = stream_v + stream_t + stream_f \
        #       + L_cross + L_aug + L_spec + L_cons + L_proto

        total = stream_f + L_cross

        comps = None
        if return_components:
            comps = GVendiLossComponents(
                ot_fusion          = L_OT_f.item(),
                align_fusion       = L_cm_f.item(),
                cross_modal_ot     = L_cross.item(),
                total              = total.item(),
            )
        return total, comps


class GvendiTopologyExtract(nn.Module):
    """
    Phase-1-only criterion:
      - extract teacher per-sample gradients
      - project them with fixed random projections
      - train the codebooks with OT losses
      - optionally cache teacher projected gradients to disk
      - save learned codebooks for phase 2
    """

    def __init__(self, args, distiller):
        super().__init__()

        if dist.is_initialized():
            self.world_size   = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size   = 1
            self.process_rank = 0

        self.args      = args
        self.distiller = distiller

        cfg = self._build_gvendi_config(args)

        teacher = distiller.teacher
        student = distiller.student

        extractor = VLMGradientExtractor(cfg.num_grad_layers)
        t_vd, t_td, t_fd = self._stream_dims(teacher, extractor)
        s_vd, s_td, s_fd = self._stream_dims(student, extractor)

        print_rank(
            f"[GVendiVLMCriterion] stream dims  "
            f"T=({t_vd},{t_td},{t_fd})  S=({s_vd},{s_td},{s_fd})"
        )

        self.gvendi = VLMDistillGVendi(
            t_vd, t_td, t_fd,
            s_vd, s_td, s_fd,
            cfg,
        )

        # phase-1 cache folder (optional)
        self.teacher_cache_dir = getattr(args, "teacher_cache_dir", None)

    @staticmethod
    def _build_gvendi_config(args) -> VLMDistillConfig:
        return VLMDistillConfig(
            proj_dim        = getattr(args, "gvendi_proj_dim",       256),
            block_size      = getattr(args, "gvendi_block_size",     4096),
            K_fusion        = getattr(args, "gvendi_K_fusion",        60),
            K_cross         = getattr(args, "gvendi_K_cross",         40),
            sinkhorn_reg    = getattr(args, "gvendi_sinkhorn_reg",   0.05),
            sinkhorn_iters  = getattr(args, "gvendi_sinkhorn_iters",   100),
            lam_fusion      = getattr(args, "gvendi_lam_fusion",      1.2),
            lam_cross_ot    = getattr(args, "gvendi_lam_cross_ot",    0.5),
            num_grad_layers = getattr(args, "gvendi_num_grad_layers",   2),
        )

    @staticmethod
    def _stream_dims(
        model: nn.Module,
        extractor: VLMGradientExtractor,
    ) -> Tuple[int, int, int]:
        def _dim(keys):
            ps = extractor._tail_params(model, keys)
            return max(sum(p.numel() for p in ps), 1)

        return (
            _dim(VLMGradientExtractor.VISUAL_KEYS),
            _dim(VLMGradientExtractor.TEXT_KEYS),
            _dim(VLMGradientExtractor.FUSION_KEYS),
        )

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

    @staticmethod
    def _save_teacher_cache(
        cache_dir: str,
        sample_ids,
        gt_v: torch.Tensor,
        gt_t: torch.Tensor,
        gt_f: torch.Tensor,
    ):
        os.makedirs(cache_dir, exist_ok=True)

        if not isinstance(sample_ids, (list, tuple)):
            sample_ids = [sample_ids]

        if len(sample_ids) != gt_v.size(0):
            raise ValueError(
                f"sample_ids length ({len(sample_ids)}) != batch size ({gt_v.size(0)})"
            )

        for i, sid in enumerate(sample_ids):
            sid = str(sid)
            cache_path = os.path.join(cache_dir, f"{sid}.pt")
            torch.save(
                {
                    "gt_v": gt_v[i].detach().cpu().half(),
                    "gt_t": gt_t[i].detach().cpu().half(),
                    "gt_f": gt_f[i].detach().cpu().half(),
                },
                cache_path,
            )

    def save_phase1_checkpoint(self, path: str):
        torch.save(
            {
                "gvendi_state_dict": self.gvendi.state_dict(),
                "cfg": self.gvendi.cfg.__dict__,
            },
            path,
        )

    def forward(
        self,
        distiller,
        input_data: Dict,
    ) -> Dict[str, torch.Tensor]:

        teacher_model = distiller.teacher

        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]
        device = teacher_qry_input["input_ids"].device

        sample_id = input_data.get("sample_id", None)

        # phase 1: teacher gradients must be enabled
        for p in teacher_model.parameters():
            p.requires_grad_(True)

        teacher_model.eval()

        with torch.enable_grad():
            t_qry_out = teacher_model.encode_input(teacher_qry_input)
            t_pos_out = teacher_model.encode_input(teacher_pos_input)

            t_qry_reps, *_ = t_qry_out
            t_pos_reps, *_ = t_pos_out

            t_per_sample = self._teacher_per_sample_signal(
                t_qry_reps, t_pos_reps
            )

            gt_v_raw, gt_t_raw, gt_f_raw = self.gvendi.extractor.extract(
                teacher_model,
                t_per_sample,
                retain_graph=True,
            )

        # fixed random projections, then normalize
        gt_v = F.normalize(self.gvendi.proj_T_v(gt_v_raw), dim=-1)
        gt_t = F.normalize(self.gvendi.proj_T_t(gt_t_raw), dim=-1)
        gt_f = F.normalize(self.gvendi.proj_T_f(gt_f_raw), dim=-1)

        # optional cache save for phase 2
        if self.teacher_cache_dir is not None and sample_id is not None:
            self._save_teacher_cache(
                self.teacher_cache_dir,
                sample_id,
                gt_v,
                gt_t,
                gt_f,
            )

        cfg = self.gvendi.cfg

        # Stage-1 OT for fusion codebook
        cost_f = _sq_euclidean(gt_f, self.gvendi.book_f.normalized)
        with torch.no_grad():
            gamma_f = _sinkhorn_log(cost_f, cfg.sinkhorn_reg, cfg.sinkhorn_iters)
        L_OT_f = (gamma_f * cost_f).sum()

        cost_cross = self.gvendi.cross_book.cost_matrix(gt_v, gt_t)
        with torch.no_grad():
            gamma_cross = _sinkhorn_log(cost_cross, cfg.sinkhorn_reg, cfg.sinkhorn_iters)
        L_cross = (gamma_cross * cost_cross).sum()

        total_loss = cfg.lam_fusion * L_OT_f + cfg.lam_cross_ot * L_cross

        # freeze teacher again after topology extraction
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        log = {
            "loss": total_loss,
            "gvendi_ot_fusion": L_OT_f.detach(),
            "gvendi_cross_modal_ot": L_cross.detach(),
        }
        return log