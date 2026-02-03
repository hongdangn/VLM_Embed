import json
from src.distiller import Distiller, DistillationCollator, DistillationDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
from src import model
from src.utils import print_rank, print_master
from src.criterions import build_criterion
import time 
import os
import sys
from tqdm import tqdm 
import math

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import wandb 
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig
import logging

# Set the logging level for Numba's CUDA driver
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)

# You may also want to set the general Numba logger level
logging.getLogger('numba').setLevel(logging.WARNING)
# Todo
# wandb.login(key="f5a118efa8813fb4edc7f6b8a7ab5c9c5f9e1ece")

def get_optimizer_params(model, training_args):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters

def get_optimizer(model, training_args):
    while isinstance(model, DDP):
        model = model.module
    optimizer_grouped_parameters = get_optimizer_params(model, training_args)
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    return optimizer

def prepare_dataset(data_args, model_args):
    dataset = DistillationDataset(data_args, model_args)
    return dataset

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def to_device(obj, device):
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [to_device(v, device) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    else:
        if hasattr(obj, 'to') and callable(obj.to):
            return obj.to(device)
        return obj

def ddp_setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(self, distiller, train_data, optimizer, lr_scheduler, criterion, model_args, training_args):
        print_rank("Initializing Trainer...")
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        # self.gpu_id = int(training_args.gpu_id)
        # self.gpu_id = 0
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.distiller = distiller.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.model_args = model_args
        self.training_args = training_args
        
        self.distiller = DDP(self.distiller, 
                             device_ids=[self.gpu_id],
                             find_unused_parameters=True)
    
    def _debug_batch_devices(self, obj, prefix=""):
        if obj is None:
            print(f"{prefix}Value: None")
            return
        
        try:
            if isinstance(obj, torch.Tensor):
                print(f"{prefix}Tensor device: {obj.device}, shape: {obj.shape}")
            elif isinstance(obj, dict):
                if len(obj) == 0:
                    print(f"{prefix}Empty dict")
                for k, v in obj.items():
                    self._debug_batch_devices(v, prefix=f"{prefix}{k}.")
            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    print(f"{prefix}Empty {type(obj).__name__}")
                for i, v in enumerate(obj):
                    self._debug_batch_devices(v, prefix=f"{prefix}[{i}].")
            else:
                print(f"{prefix}Type: {type(obj).__name__}, Value: {obj}")
        except Exception as e:
            print(f"{prefix}ERROR: {e}")
        
    def run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        # total_losses, contrastive_losses, rkd_losses, simple_kd_losses = [], [], [], []
        # intra_rkd_losses, cross_modal_kd_losses, ot_losses, img_align_losses = [], [], [], []
        losses = {}
        
        progress_bar = tqdm(total=len(self.train_data.dataset) // self.training_args.per_device_train_batch_size // self.training_args.gradient_accumulation_steps // dist.get_world_size(), 
                            desc=f"Epoch {epoch}",
                            disable=not dist.get_rank() == 0)
        for batch_idx, batch in enumerate(self.train_data):
            try:
                batch = to_device(batch, self.device)
                loss_dict = self.distiller(self.criterion, batch)
                del batch
                torch.cuda.empty_cache()
            
                if batch_idx == 0:
                    losses = {loss_type: [] for loss_type in loss_dict}
                
                for loss_type in losses:
                    batch_loss = loss_dict.get(loss_type, torch.tensor(0.0))
                    losses[loss_type].append(batch_loss.detach().item())

                total_loss = loss_dict['loss'] / self.training_args.gradient_accumulation_steps
                total_loss.backward()
            except RuntimeError as e:
                print(e)
                torch.cuda.empty_cache()
                continue
            
            if (batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            if is_main_process():
                postfix_batch_loss = {
                    loss_type: f"{(sum(losses[loss_type]) / len(losses[loss_type])):.4f}"
                    for loss_type in losses
                }
                progress_bar.set_postfix(postfix_batch_loss)
                progress_bar.update(1)
                # if "wandb" in self.training_args.report_to:
                #     wandb.log({
                #         "train/loss": batch_loss,
                #         "train/contrastive_loss": batch_contrastive_loss,
                #         "train/rkd_loss": batch_rkd_loss,
                #         "train/simple_kd_loss": batch_simple_kd_loss,
                #         "train/intra_rkd_loss": batch_kd_rkd_loss,
                #         "train/cross_modal_kd_loss": batch_kd_dtw_loss,
                #         "train/ot_loss": batch_ot_loss,
                #         "train/img_align_loss": batch_img_align_loss,
                #     }, step=epoch * (len(self.train_data.dataset) // self.training_args.per_device_train_batch_size // dist.get_world_size()) + batch_idx // self.training_args.gradient_accumulation_steps)
            torch.cuda.empty_cache()
                    
        progress_bar.close()
        
    def train(self):
        # if "wandb" in self.training_args.report_to and self.gpu_id == 0:
        #     wandb.init(
        #         project=self.training_args.output_dir.split("/")[-1],
        #         name=self.model_args.model_backbone + "_distillation",
        #         config={
        #             "learning_rate": self.training_args.learning_rate,
        #             "batch_size": self.training_args.per_device_train_batch_size,
        #             "epochs": self.training_args.num_train_epochs,
        #             "gradient_accumulation_steps": self.training_args.gradient_accumulation_steps,
        #         }
        #     )
        for epoch in range(self.training_args.num_train_epochs):
            self.run_epoch(epoch)
            if self.training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(self.training_args.output_dir, f"checkpoint-epoch-{epoch}")
                projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                student = self.distiller.module.student
                student.encoder.save_pretrained(ckpt_dir)
                
                if self.model_args.model_backbone in ["llava_onevision", "llava_two_vision"]:
                    torch.save(student.encoder.model.multi_modal_projector.state_dict(), projector_dir)
                else:
                    torch.save(student.encoder.model.model.mm_projector.state_dict(), projector_dir)

                student_config = AutoConfig.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                if student_config:
                    student_config.save_pretrained(ckpt_dir)
                if tokenizer:
                    tokenizer.save_pretrained(ckpt_dir)
                try:
                    processor = AutoProcessor.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                    if processor:
                        processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning: Could not save processor: {e}")
                print_rank(f"Saved checkpoint to {ckpt_dir}")

                ## for evaluation
                print(f"Start evaluating student at epoch {epoch}")
                

                
def main():
    for arg in sys.argv:
        if arg.startswith("--local_rank"):
            local_rank = int(arg.split("=")[-1])
            sys.argv.remove(arg)
            sys.argv.append(f"--local_rank")
            sys.argv.append(f"{local_rank}")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    
    distiller = Distiller(model_args, training_args)
    num_trainable_projector = 0

    for name, param in distiller.student.named_parameters():
        if "mm_projector" in name:
            param.requires_grad = True
            num_trainable_projector += param.numel()

    print("Num trainable parameters: ", num_trainable_projector)

    train_dataset = prepare_dataset(data_args, model_args)
    dist_sampler = DistributedSampler(train_dataset, shuffle=True)
    for n, p in distiller.student.named_parameters():
        if p.requires_grad:  # thường chỉ là LoRA
            p.data = p.data.to(torch.bfloat16)
    
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=dist_sampler,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
    )
    params_to_optimize = list(distiller.student.parameters()) + \
                         list(distiller.t2s_img_align.parameters()) + \
                         list(distiller.last_layer_projector.parameters())

    num_trainable_vision = 0

    for n, p in distiller.student.named_parameters():
            
        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True
            
        if "lm_head" in n:
            p.requires_grad = False

        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
            num_trainable_vision += p.numel()
    print_rank(f"Number of trainable vision parameters: {num_trainable_vision}")

    optimizer = AdamW(
        params_to_optimize,
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )

    print(f"Len of train dataset: {len(train_dataloader.dataset)}")
    total_steps = (len(train_dataloader.dataset) // (training_args.per_device_train_batch_size * dist.get_world_size()) // training_args.gradient_accumulation_steps) * training_args.num_train_epochs
    if model_args.projector_config_path is not None:
        optimizer = distiller.add_optimizer_param_group(optimizer)
        
    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    else:
        from transformers import get_constant_schedule_with_warmup
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
        )
    criterion = build_criterion(training_args)
    trainer = Trainer(distiller, train_dataloader, optimizer, lr_scheduler, criterion, model_args, training_args)
    trainer.train()
    
if __name__ == "__main__":
    ddp_setup()
    main()
    destroy_process_group()