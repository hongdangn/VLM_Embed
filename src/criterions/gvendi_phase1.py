import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict
from .gvendi_utils.rademacher_projection import RademacherProjection
from .gvendi_utils.utils import teacher_per_sample_signal, per_sample_grads, detach_hook, \
    get_sample_grads_from_input


from prepare_one_model_training import OneModelTrainer

class GvendiPhase1(nn.Module):
    def __init__(self, args, model_trainer):
        super().__init__()
        self.args = args
        self.model_trainer = model_trainer
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0


    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.zeros_like(t) for _ in range(self.world_size)] 
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    

    def forward(self, input_data: Dict[str, Tensor]):
        model = self.model_trainer.model

        target_layers = self.model_trainer.target_layers
        target_params = self.model_trainer.target_params
        target_named_params = self.model_trainer.target_named_params

        sample_grad = get_sample_grads_from_input(input_data, model, target_layers, target_params, target_named_params)
        
        # print(sample_grad.shape)  # (batch_size, total_gradient_dim)
        # print(sample_grad[0][:10])  # In ra 10 giá trị đầu tiên của gradient vector của mẫu đầu tiên
        # print(sample_grad[1][:10])  # In ra 10 giá trị đầu tiên của gradient vector của mẫu thứ hai (nếu có)
        exit(0)
        return {
            # "loss": contrastive_loss, 
            # "extracted_gradients": final_grad_vector
        }

        