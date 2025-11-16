import json
from prepare_one_model_training import TrainOneModelCollator, OneModelTrainer, TrainOneModelDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
from src import model
from src.utils import print_rank, print_master
from src.criterions import build_criterion
import time 
import os
import sys
from tqdm import tqdm 

import torch
import torch.nn as nn 
import torch.nn.functional as F
# import torch.distributed as dist # Removed, accelerate handles this
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler # DistributedSampler is no longer needed
from torch.optim import AdamW

# import deepspeed # Removed, accelerate can manage this if configured
import wandb 
from accelerate import Accelerator # Import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig
from huggingface_hub import login
from dotenv import load_dotenv

# token=os.getenv("HF_TOKEN")
# login(token=token)

# wandb.login(key="810a71e03133ccddd00133f1fe9d2cd0f8001b4e")

def push_to_hub(repo_name=None, token=None, commit_message="Upload model", 
                local_dir="./temp_model", private=False):
    # This function is fine as-is, but should only be called by the main process.
    try:
        if not repo_name:
            raise ValueError("must specify a repo name to push to hub")
        
        if not os.path.exists(local_dir):
            raise ValueError(f"local_dir {local_dir} does not exist")
        
        print_rank(f"Pushing model to the hub at {repo_name}...")
        api = HfApi()
        create_repo(repo_name, token=token, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_name, 
            token=token, 
            commit_message=commit_message
        )

        print_rank(f"Model has been pushed to the hub at: {repo_name}")
        return True
        
    except Exception as e:
        print_rank(f"Error pushing to hub: {str(e)}")
        return False

# This function is no longer needed with accelerate.prepare(dataloader)
# def batch_to_device(batch, device):
#     _batch = {}
#     for key, value in batch.items():
#         if isinstance(value, torch.Tensor):
#             _batch[key] = value.to(device)
#         else:
#             _batch[key] = value
#     return _batch

def finetune(
    model_args: ModelArguments, 
    data_args: DataArguments,
    training_args: TrainingArguments,
    model_trainer: OneModelTrainer, 
    train_dataset: TrainOneModelDataset, # train_dataset is passed, but dataloader is prepared in main
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    collator: TrainOneModelCollator,
    criterion: nn.Module,
    accelerator: Accelerator, # Pass in the accelerator
    train_dataloader: DataLoader, # Pass in the prepared dataloader
):
    print_rank("Start finetuning...")
    start_time = time.time()
    
    # Get the prepared model from the trainer
    # The model, optimizer, and dataloader are already prepared by accelerator in main
    model = model_trainer.model 
    
    # Casting to bfloat16 is handled by accelerate if mixed precision (e.g., "bf16") is configured
    # You can remove this loop if you use accelerate's mixed precision
    for n, p in model.named_parameters():
        if p.requires_grad:
            try:
                p.data = p.data.to(torch.bfloat16)
            except Exception:
                pass
    
    model.train()
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank(f"Number of parameters in student model: {num_trainable}")
    
    # Wandb init should only happen on the main process
    # if accelerator.is_main_process:
    #     wandb.init(
    #         project="dang_sft",
    #         # name=f"finetune-{model_args.model_backbone}-{data_args.subset_name}",
    #         # config={**vars(model_args), **vars(data_args), **vars(training_args)}, 
    #         # Pass accelerator config to wandb
    #         # config=accelerator.state,
    #     )
    
    step = 0
    logging_output = {
        'epoch': 0, 
        'global_step': 0, 
        'loss': [],
        'contrastive_loss': [],
        'micro_step_time': [],
        'step_time': []
    }
    
    for epoch in range(training_args.num_train_epochs):
        logging_output['epoch'] = epoch + 1
        print_rank("Start iteration of epoch {}".format(epoch + 1))
        
        epoch_loss, epoch_contrastive_loss, epoch_kd_loss = 0, 0, 0
        losses, contrastive_losses, kd_losses = [], [], []
        
        # No need to set epoch on sampler, accelerate's dataloader handles it
        
        progress_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}",
            # Disable progress bar on non-main processes
            disable=not accelerator.is_main_process,
        )
        
        # device = accelerator.device # Get device from accelerator
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Batches are automatically moved to the correct device by accelerate
            # No need for batch_to_device or device_batch
            
            # Apply bfloat16 conversion manually if not using accelerate's mixed precision
            batch['qry']['images'] = batch['qry']["images"].to(dtype=torch.bfloat16)
            # batch['pos']['images'] = batch['pos']["images"].to(dtype=torch.bfloat16)
            
            st_time = time.time()
            
            # Use the model directly (it's already wrapped)
            loss_dict = model_trainer(criterion, batch)
            
            loss = loss_dict['loss']
            
            # Use accelerator.backward() instead of loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            
            contrastive_loss = loss_dict['contrastive_loss']
            # kd_loss = loss_dict['kd_loss']
            
            # Gather losses across all processes for accurate logging
            avg_loss = accelerator.gather(loss).mean().item()
            avg_contrastive_loss = accelerator.gather(contrastive_loss).mean().item()
            
            losses.append(avg_loss)
            contrastive_losses.append(avg_contrastive_loss)
            # kd_losses.append(kd_loss.detach().item()) # Handle kd_loss similarly if needed
            
            logging_output['micro_step_time'].append(time.time() - st_time)
            
            step += 1
            logging_output['global_step'] = step
            
            # No need for torch.cuda.synchronize()
            
            batch_loss = sum(losses)/len(losses)
            batch_contrastive_loss = sum(contrastive_losses)/len(contrastive_losses)
            # batch_kd_loss = sum(kd_losses)/len(kd_losses)
            
            progress_bar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "contrastive_loss": f"{batch_contrastive_loss:.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
            })
            progress_bar.update(1)
            
            # Log only on the main process
            # if accelerator.is_main_process and "wandb" in training_args.report_to:
            #     wandb.log({
            #         "train/loss": batch_loss,
            #         "train/contrastive_loss": batch_contrastive_loss,
            #         # "train/kd_loss": batch_kd_loss,
            #         "train/lr": lr_scheduler.get_last_lr()[0],
            #         "train/epoch": epoch + 1,
            #         "train/global_step": step,
            #     })

        # End of epoch
        # Calculate average epoch loss across all processes
        avg_epoch_loss = sum(losses) / len(losses)
        avg_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
        
        print_rank(
            f"Epoch {epoch + 1} completed. Avg Loss: {avg_epoch_loss:.4f} | "
            f"Avg Contrastive Loss: {avg_contrastive_loss:.4f} "
        )
            
        # Log epoch-level stats only on the main process
        # if accelerator.is_main_process and "wandb" in training_args.report_to:
        #     wandb.log({
        #         "epoch/avg_loss": avg_epoch_loss,
        #         "epoch/avg_contrastive_loss": avg_contrastive_loss,
        #         # "epoch/avg_kd_loss": avg_kd_loss,
        #         "epoch/epoch": epoch + 1,
        #     })
        
        # Save checkpoint only on the main process
        if training_args.save_strategy == "epoch":
            accelerator.wait_for_everyone() # Ensure all processes are done
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)

                # Unwrap the model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                
                if hasattr(unwrapped_model, 'peft_config'):
                    unwrapped_model.peft_config.save_pretrained(ckpt_dir)
                    unwrapped_model.save_pretrained(ckpt_dir)
                    print_rank("Saved LoRA adapter model.")
                else:
                    unwrapped_model.encoder.save_pretrained(ckpt_dir)
                    print_rank("Saved full student model.")
                
                # This seems redundant if the above saves correctly
                # unwrapped_model.encoder.save_pretrained(ckpt_dir)          
                print_rank(f"Checkpoint saved at {ckpt_dir}")
                
                model_config = AutoConfig.from_pretrained(model_args.model_name) if model_args.model_name else None
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_name) if model_args.model_name else None

                if model_config:
                    model_config.save_pretrained(ckpt_dir)
                if tokenizer:
                    tokenizer.save_pretrained(ckpt_dir)

                try:
                    processor = AutoProcessor.from_pretrained("dangnguyens1/sft-fastvlm-1e")
                    processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Error saving processor: {str(e)}")
                
                # push_to_hub(
                #     repo_name="dangnguyens1/sft-fastvlm-1e",
                #     token=token,
                #     local_dir=ckpt_dir,
                # )

                print_rank(f"Epoch {epoch + 1} finished.")

    total_time = time.time() - start_time
    print_rank(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save final model, only on main process
    if training_args.save_strategy == "epoch":
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-final")
            os.makedirs(final_ckpt_dir, exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)

            if hasattr(unwrapped_model, 'peft_config'):
                unwrapped_model.peft_config.save_pretrained(final_ckpt_dir)
                unwrapped_model.save_pretrained(final_ckpt_dir)
                print_rank("Saved LoRA adapter model.")
            else:
                unwrapped_model.encoder.save_pretrained(final_ckpt_dir)
                print_rank("Saved full student model.")
            
            if model_args.model_name:
                model_config = AutoConfig.from_pretrained(model_args.model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
                
                model_config.save_pretrained(final_ckpt_dir)
                tokenizer.save_pretrained(final_ckpt_dir)

                try:
                    processor = AutoProcessor.from_pretrained("dangnguyens1/sft-fastvlm-final_e")
                    processor.save_pretrained(final_ckpt_dir) # Corrected path
                except Exception as e:
                    print_rank(f"Error saving processor: {str(e)}")
                
                # push_to_hub(
                #     repo_name="dangnguyens1/sft-fastvlm-final_e",
                #     token=token,
                #     local_dir=final_ckpt_dir, # Corrected path
                # )

    # if accelerator.is_main_process and "wandb" in training_args.report_to:
    #     wandb.finish()

    return logging_output

def main():
    
    # Initialize accelerator
    accelerator = Accelerator()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(data_args.subset_name)
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    train_dataset = TrainOneModelDataset(data_args, model_args)
    print_rank(f"Number of training samples: {len(train_dataset)}")
    
    # Pass accelerator's device to the trainer
    model_trainer = OneModelTrainer(model_args, 
                                    training_args, 
                                    device="cuda") # Use accelerator's device
    collator = TrainOneModelCollator(
        processor=model_trainer.get_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    
    # Create Dataloader here to be prepared by accelerator
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        shuffle=True, # Sampler is handled by accelerate
        drop_last=True,
        pin_memory=True,
    ) 
    
    optimizer = AdamW(
        model_trainer.model.parameters(),
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    
    # Use accelerator.num_processes instead of world_size
    steps_per_epoch = len(train_dataloader) // (
        training_args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * training_args.num_train_epochs
        
    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps ,
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
        from transformers import get_constant_schedule
        lr_scheduler = get_constant_schedule(optimizer)
        
    criterion = build_criterion(training_args)
    print(model_args.model_backbone)
    
    # Prepare everything with accelerator
    model_trainer.model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model_trainer.model, optimizer, train_dataloader, lr_scheduler
    )
    
    logging_output = finetune(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        model_trainer=model_trainer,
        train_dataset=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        collator=collator,
        criterion=criterion,
        accelerator=accelerator, # Pass accelerator
        train_dataloader=train_dataloader # Pass prepared dataloader
    )
    
    print_rank("Training completed successfully!")
    return logging_output

if __name__ == "__main__":
    main()