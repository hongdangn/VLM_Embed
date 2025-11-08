import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from .soft_DTW import SoftDTW
import ot
# ot.backend.get_backend('pytorch')

class StrongerKD(nn.Module):
    def __init__(self, args):
        super(StrongerKD, self).__init__()
        self.args = args
        self.rkd_loss_weight = 0.5
        self.simple_kd_weight = 0.5
        self.intra_rkd_weight = 0.5
        self.img_align_loss_weight = 0.1
        self.cross_modal_kd_weight = 0.001
        self.ot_loss_weight = 0.5
        self.num_chosen_hidden_states = 3

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.01)

    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        with torch.no_grad():
            teacher_model.eval()
            t_qry_reps, t_qry_hidden_states, t_qry_img_feats, t_qry_layers_embeds, t_qry_attention = \
                                                                    teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos_reps, t_pos_hidden_states, t_pos_img_feats, t_pos_layers_embeds, t_pos_attention = \
                                                                    teacher_model.encode_input(input_data['teacher_inputs']['pos'])

            # t_qry_hidden_states, t_qry_img_feats = torch.stack(t_qry_hidden_states, dim=0), torch.stack(t_qry_img_feats, dim=0)
            # t_pos_hidden_states, t_pos_img_feats = torch.stack(t_pos_hidden_states, dim=0), torch.stack(t_pos_img_feats, dim=0)

        s_qry_reps, s_qry_hidden_states, s_qry_img_feats, s_qry_layers_embeds, s_qry_attention = \
                                                                    student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos_reps, s_pos_hidden_states, s_pos_img_feats, s_pos_layers_embeds, s_pos_attention = \
                                                                    student_model.encode_input(input_data['student_inputs']['pos'])

        # s_qry_hidden_states, s_qry_img_feats = torch.stack(s_qry_hidden_states, dim=0), torch.stack(s_qry_img_feats, dim=0)
        # s_pos_hidden_states, s_pos_img_feats = torch.stack(s_pos_hidden_states, dim=0), torch.stack(s_pos_img_feats, dim=0)

        ## contrastive
        scores = student_model.compute_similarity(s_qry_reps, s_pos_reps)
        scores = scores.view(s_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (s_qry_reps.size(0) // s_pos_reps.size(0))
        contrastive_loss = self.cross_entropy_loss(scores / self.distiller.temperature, target)

        ## image alignments
        img_align_loss = 0.0
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0
        batch_size = s_qry_reps.size(0)

        for i in range(batch_size):
            if s_qry_img_feats is not None and t_qry_img_feats is not None:
                if cur_idx_qry_img < len(s_qry_img_feats) and cur_idx_qry_img < len(t_qry_img_feats):
                    tmp_s_qry_img_feats = F.normalize(s_qry_img_feats[i], p=2, dim=-1)
                    tmp_t_qry_img_feats = self.distiller.t2s_img_align(t_qry_img_feats[i])

                    tmp_t_qry_image_features = F.normalize(tmp_t_qry_img_feats, p=2, dim=-1)

                    img_align_loss += self.alignment_loss_mmd(tmp_t_qry_image_features, tmp_s_qry_img_feats)
                    cur_idx_qry_img += 1

            if s_pos_img_feats is not None and t_pos_img_feats is not None:
                if cur_idx_pos_img < len(s_pos_img_feats) and cur_idx_pos_img < len(t_pos_img_feats):
                    tmp_s_pos_img_feats = F.normalize(s_pos_img_feats[i], p=2, dim=-1)
                    tmp_t_pos_img_feats = self.distiller.t2s_img_align(t_pos_img_feats[i])
                    
                    tmp_t_pos_image_features = F.normalize(tmp_t_pos_img_feats, p=2, dim=-1)

                    img_align_loss += self.alignment_loss_mmd(tmp_t_pos_image_features, tmp_s_pos_img_feats)
                    cur_idx_pos_img += 1

        img_align_loss = img_align_loss / batch_size

        ## data-points rkd 
        # s_qry_reps = F.normalize(s_qry_reps, p=2, dim=-1)
        # s_pos_reps = F.normalize(s_pos_reps, p=2, dim=-1)
        # t_qry_reps = F.normalize(t_qry_reps, p=2, dim=-1)
        # t_pos_reps = F.normalize(t_pos_reps, p=2, dim=-1)

        # qry_distance_loss = self.compute_distance_loss(s_qry_reps, t_qry_reps)
        # pos_distance_loss = self.compute_distance_loss(s_pos_reps, t_pos_reps)
        # distance_loss = 0.5 * qry_distance_loss + 0.5 * pos_distance_loss

        # qry_angle_loss = self.compute_angle_loss(s_qry_reps, t_qry_reps)
        # pos_angle_loss = self.compute_angle_loss(s_pos_reps, t_pos_reps)
        # angle_loss = 0.5 * qry_angle_loss + 0.5 * pos_angle_loss

        # rkd_loss = (0.5 * distance_loss + 0.5 * angle_loss)
        rkd_loss = torch.tensor(0.0)

        ## simple kd
        simple_kd_loss = self.simple_kd_logit_loss(s_qry_reps, s_pos_reps, t_qry_reps, t_pos_reps)

        ## intra rkd
        intra_rkd_loss = self.intra_rkd(t_qry_layers_embeds, t_pos_layers_embeds,
                                        s_qry_layers_embeds, s_pos_layers_embeds)
        
        ## cross modal kd
        # num_s_img_tokens, num_t_img_tokens = s_qry_img_feats.size(1), t_qry_img_feats.size(1)
        qry_cross_modal_kd_loss = self.cross_modal_kd_loss(s_qry_hidden_states,
                                                       t_qry_hidden_states,
                                                       s_qry_img_feats,
                                                       t_qry_img_feats)

        # num_s_img_tokens, num_t_img_tokens = s_pos_img_feats.size(1), t_pos_img_feats.size(1)
        pos_cross_modal_kd_loss = self.cross_modal_kd_loss(s_pos_hidden_states,
                                                       t_pos_hidden_states,
                                                       s_pos_img_feats,
                                                       t_pos_img_feats)
        
        cross_modal_kd_loss = 0.5 * qry_cross_modal_kd_loss + 0.5 * pos_cross_modal_kd_loss

        ## optimal transport loss
        ot_loss = self.compute_ot(s_qry_hidden_states, s_qry_attention,
                                  t_qry_hidden_states, t_qry_attention)
        ot_loss += self.compute_ot(s_pos_hidden_states, s_pos_attention,
                                  t_pos_hidden_states, t_pos_attention)
        ot_loss = ot_loss / 2.0

        total_loss = contrastive_loss + \
                     self.rkd_loss_weight * rkd_loss + \
                     self.simple_kd_weight * simple_kd_loss + \
                     self.intra_rkd_weight * intra_rkd_loss + \
                     self.cross_modal_kd_weight * cross_modal_kd_loss + \
                     self.ot_loss_weight * ot_loss + \
                     self.img_align_loss_weight * img_align_loss
        
        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "rkd_loss": rkd_loss,
            "simple_kd_loss": simple_kd_loss,
            "intra_rkd_loss": intra_rkd_loss,
            "cross_modal_kd_loss": cross_modal_kd_loss,
            "ot_loss": ot_loss,
            "img_align_loss": img_align_loss
        }

    def gaussian_kernel(self, x, y, sigma=2.0):
        """
        Computes the RBF (Gaussian) kernel between two sets of vectors.
        k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        
        Args:
            x (torch.Tensor): Shape (n, dim)
            y (torch.Tensor): Shape (m, dim)
            sigma (float): Kernel bandwidth.
        """
        beta = 1.0 / (2.0 * (sigma ** 2))
        # (n, m) matrix of squared pairwise distances
        dist_sq = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).pow(2)
        return torch.exp(-beta * dist_sq)

    def alignment_loss_mmd(self, t_feats, s_feats, sigma=2.0):
        """
        Computes the Maximum Mean Discrepancy (MMD) loss using a Gaussian kernel.

        Args:
            x_teacher (torch.Tensor): Teacher features, shape (n, dim)
            x_student (torch.Tensor): Student features, shape (m, dim)
            sigma (float): Kernel bandwidth.
        """
        
        # Compute kernel matrices
        k_tt = self.gaussian_kernel(t_feats, t_feats, sigma) # (n, n)
        k_ss = self.gaussian_kernel(s_feats, s_feats, sigma) # (m, m)
        k_ts = self.gaussian_kernel(t_feats, s_feats, sigma) # (n, m)
        
        # This is the (biased) MMD^2 statistic
        # E[k(t, t')] + E[k(s, s')] - 2 * E[k(t, s)]
        mmd_loss = k_tt.mean() + k_ss.mean() - 2 * k_ts.mean()
        
        return mmd_loss
    
    def compute_ot(self, s_hidden_states, s_attn, t_hidden_states, t_attn):
        
        loss = 0.0
        num_student_layers = len(s_hidden_states)
        num_teacher_layers = len(t_hidden_states)
        scale = num_teacher_layers // num_student_layers
        start_layer = num_student_layers - self.distiller.num_chosen_hidden_states

        for l in range(start_layer, num_student_layers):
            s_dist = F.softmax(s_attn[l - 1].mean(dim=1)[:, -1], dim=-1) # (b, n)
            t_dist = F.softmax(t_attn[l - 1].mean(dim=1)[:, -1], dim=-1) # (b, m)

            s_hidden_state = s_hidden_states[l] # (b, n, emb_dim)
            proj_t_hidden_state = self.distiller.t2s[l - start_layer](t_hidden_states[scale * l]) # (b, m, emb_dim)

            for b in range(s_dist.size(0)):
                cost_matrix = torch.cdist(s_hidden_state[b].unsqueeze(0), 
                                          proj_t_hidden_state[b].unsqueeze(0)).squeeze(0)
                cost_matrix = cost_matrix / cost_matrix.mean()

                transport = self.sinkhorn(s_dist[b], t_dist[b], cost_matrix) 
                loss += torch.sum(transport * cost_matrix)

        return loss
    
    def sinkhorn(self, a, b, cost_matrix, reg=0.1, num_iters=100, eps=1e-9, stopThr = 1e-7):
        """
        a: (m,) or (m,1) torch tensor (source weights)
        b: (n,) or (n,1) torch tensor (target weights)
        cost_matrix: (m, n) torch tensor
        reg: regularization (>=0) -- larger reg -> smoother K = exp(-C/reg)
        num_iters: number of Sinkhorn iterations
        """
        device = cost_matrix.device
        dtype = cost_matrix.dtype

        a = a.view(-1, 1)
        b = b.view(-1, 1)
        C = cost_matrix

        m, n = C.shape
        if m == 0 or n == 0:
            return torch.zeros((m, n), device=device, dtype=dtype)

        # ensure shapes
        if a.shape[0] != m:
            a = torch.ones((m, 1), device=device, dtype=dtype) / m
        if b.shape[0] != n:
            b = torch.ones((n, 1), device=device, dtype=dtype) / n

        suma = a.sum()
        sumb = b.sum()
        if suma <= eps or sumb <= eps:
            a = torch.ones((m, 1), device=device, dtype=dtype) / m
            b = torch.ones((n, 1), device=device, dtype=dtype) / n
        else:
            a = a / suma
            b = b / sumb

        K = torch.exp(-C / (reg + 1e-12))

        u = torch.ones((m, 1), device=device, dtype=dtype)
        v = torch.ones((n, 1), device=device, dtype=dtype)

        for i in range(num_iters):
            u_prev = u.clone()
            KTv = (K.t() @ u)  # shape (n,1)
            v = b / (KTv + eps)
            Kv = (K @ v)       # shape (m,1)
            u = a / (Kv + eps)

            err = torch.max(torch.abs(u - u_prev))
            if err.item() < stopThr:
                break

        # transport plan
        U = torch.diag_embed(u.squeeze())   # (m,m) diag(u)
        V = torch.diag_embed(v.squeeze())   # (n,n) diag(v)
        P = U @ K @ V                       # (m,n)
        return P
    
    def cross_modal_kd_loss(self, s_hidden_states, t_hidden_states, s_img_feats, t_img_feats):
        """
            hidden_states: list of (n_layers, b, n, dim)
            img_feats: (b, n_img_tokens, dim)
        """

        loss = 0.0
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0
        num_student_layers = len(s_hidden_states)
        num_teacher_layers = len(t_hidden_states)
        scale = num_teacher_layers // num_student_layers
        batch_size = s_hidden_states[0].size(0)
        start_layer = num_student_layers - self.distiller.num_chosen_hidden_states

        if s_img_feats is None or t_img_feats is None:
            return loss
        
        for b in range(batch_size):
            if s_img_feats[b] is not None and t_img_feats[b] is not None:

                for l in range(start_layer, num_student_layers):

                    num_s_img_tokens = s_img_feats[b].size(0)
                    num_t_img_tokens = t_img_feats[b].size(0)

                    s_img_hidden_states = F.normalize(s_hidden_states[l][b][:num_s_img_tokens])
                    s_text_hidden_states = F.normalize(s_hidden_states[l][b][num_s_img_tokens:])

                    proj_t_img_hidden_states = F.normalize(self.distiller.t2s[l - start_layer](t_hidden_states[scale * l][b][:num_t_img_tokens]))
                    proj_t_text_hidden_states = F.normalize(self.distiller.t2s[l - start_layer](t_hidden_states[scale * l][b][num_t_img_tokens:]))
                    
                    loss += 0.5 * self.sdtw(s_img_hidden_states.unsqueeze(0), proj_t_text_hidden_states.unsqueeze(0)).mean()
                    loss += 0.5 * self.sdtw(s_text_hidden_states.unsqueeze(0), proj_t_img_hidden_states.unsqueeze(0)).mean()

        return loss / (batch_size * self.distiller.num_chosen_hidden_states)
    
    def simple_kd_logit_loss(self, student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps):
            projector_teacher_qry_reps = self.distiller.last_layer_projector(teacher_qry_reps)
            projector_teacher_pos_reps = self.distiller.last_layer_projector(teacher_pos_reps)

            loss = (
                    self.mse_loss(student_qry_reps, projector_teacher_qry_reps) +  
                    self.mse_loss(student_pos_reps, projector_teacher_pos_reps)
                   ) / 2.0
            return loss
    
    def intra_rkd(self, 
                  teacher_qry_layers_embeds, # (b, n_layers, dim), 
                  teacher_pos_layers_embeds,
                  student_qry_layers_embeds,
                  student_pos_layers_embeds):
        
        loss = 0.0
        batch_size = student_pos_layers_embeds.size(0)

        for b in range(batch_size):

            qry_dist_loss = self.compute_distance_loss(student_qry_layers_embeds[b], teacher_qry_layers_embeds[b])
            pos_dist_loss = self.compute_distance_loss(student_pos_layers_embeds[b], teacher_pos_layers_embeds[b])
            dist_loss = 0.5 * qry_dist_loss + 0.5 * pos_dist_loss
            
            qry_angle_loss = self.compute_angle_loss(student_qry_layers_embeds[b], teacher_qry_layers_embeds[b])
            pos_angle_loss = self.compute_angle_loss(student_pos_layers_embeds[b], teacher_pos_layers_embeds[b])
            angle_loss = 0.5 * qry_angle_loss + 0.5 * pos_angle_loss

            loss += 0.5 * dist_loss + 0.5 * angle_loss

        return loss / batch_size

    def pairwise_distance(self, x):
        norm = (x**2).sum(dim=1, keepdim=True)
        dist = norm + norm.t() - 2.0 * torch.mm(x, x.t())
        return dist
    
    def compute_distance_loss(self, student_repr, teacher_repr):
        
        num_student_layers = student_repr.size(0)
        num_teacher_layers = teacher_repr.size(0)
        scale = num_teacher_layers // num_student_layers

        teacher_repr = teacher_repr[
            torch.tensor([i * scale for i in range(num_student_layers)], device=teacher_repr.device)
        ]

        dist_student = self.pairwise_distance(student_repr)
        dist_teacher = self.pairwise_distance(teacher_repr)
        
        mask = torch.triu(torch.ones_like(dist_student), diagonal=1).bool()
        dist_student = dist_student[mask]
        dist_teacher = dist_teacher[mask]
        
        mean_td = dist_teacher.mean().detach() + 1e-8
        mean_sd = dist_student.mean().detach() + 1e-8
        
        dist_student = dist_student / mean_sd
        dist_teacher = dist_teacher / mean_td
        
        diff = dist_student - dist_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5
        
        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        loss = loss.mean()
        return loss
    
    def angle_potentials(self, x):
        n = x.size(0)
        diffs = x.unsqueeze(0) - x.unsqueeze(1)
        norms = torch.norm(diffs, dim=-1, keepdim=True) + 1e-8
        e = diffs / norms
        
        cos_angles = torch.einsum('ijd,kjd->ijk', e, e)
        return cos_angles
    
    def compute_angle_loss(self, student_repr, teacher_repr):
        
        num_student_layers = student_repr.size(0)
        num_teacher_layers = teacher_repr.size(0)
        scale = num_teacher_layers // num_student_layers

        teacher_repr = teacher_repr[
            torch.tensor([i * scale for i in range(num_student_layers)], device=teacher_repr.device)
        ]

        psi_student = self.angle_potentials(student_repr)
        psi_teacher = self.angle_potentials(teacher_repr)
        
        n = psi_student.size(0)
        mask = torch.ones((n, n, n), dtype=torch.bool, device=psi_student.device)
        idx = torch.arange(n, device=psi_student.device)
        mask[idx, idx, :] = 0
        mask[idx, :, idx] = 0
        mask[:, idx, idx] = 0
        
        psi_teacher = psi_teacher[mask]
        psi_student = psi_student[mask]
        
        diff = psi_student - psi_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5
        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        loss = loss.mean()
        return loss
        