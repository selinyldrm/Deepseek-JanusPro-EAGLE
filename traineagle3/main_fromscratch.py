import argparse
import os
import sys
import json
import torch
import wandb

# Add parent directory to Python path to find models module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from models.configs.configs import EConfig
from cnets_lumina_mgpt import Eagle3Model
from cnets_anole import Eagle3AnoleModel
# from cnets_llamagen import Model
from entrypoints.train_drafter.data_utils import (
    list_files,
    AddGaussianNoise,
    AddUniformNoise,
    CustomDataset,
    CoupledDataset,
    DataCollatorWithPadding,
    DataCollatorWithPaddingForCoupled,
)

torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description='Training Eagle 3 drafter for image generation')
    
    # paths and directories
    parser.add_argument("--model", type=str, default="lumina_mgpt")
    parser.add_argument('--base_path', type=str, default='ckpts/lumina_mgpt/Lumina-mGPT-7B-768')
    parser.add_argument('--config_path', type=str, default='data/configs/lumina_mgpt_config.json')
    parser.add_argument('--data_dir', type=str, default='/home/server44/sihwan_workspace/ssd/lumina_mgpt_eagle_mscoco2017train')
    parser.add_argument('--save_dir', type=str, default='ckpts/lumina_mgpt/trained_drafters_eagle3')
    
    # dataset arguments
    parser.add_argument('--coupled', action='store_true', default=False)
    parser.add_argument('--train_data_ratio', type=float, default=0.95)
    parser.add_argument('--data_noise', type=str, default='uniform')
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=0.2)
    
    # training arguments
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--warmup_steps_ratio', type=float, default=0.03)
    parser.add_argument('--is_warmup', action='store_true', default=True)
    
    # Eagle 3 specific arguments
    parser.add_argument('--length', type=int, default=7, help='Eagle 3 multi-step training length')
    parser.add_argument('--eagle3_weight_decay', type=float, default=0.8, help='Weight decay for Eagle 3 multi-step loss')
    
    parser.add_argument('--p_w', type=float, default=0.1)
    parser.add_argument('--cfg_loss', action='store_true', default=False)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--embed_upscale', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--wandb', action='store_true', default=True)

    return parser

@torch.no_grad()
def top_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    _, pred = output.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res

def update_metrics(plosses, acces, loss_mask, top_3acc):
    """Update metrics for Eagle 3 multi-step training"""
    total = loss_mask.sum().item()
    
    # Use the last step's accuracy as the primary metric
    if len(acces) > 0:
        correct = acces[-1] * total
    else:
        correct = 0
    
    # For top-k accuracy, we'll use a simplified approach
    # In a real implementation, you'd want to compute this properly for each step
    if len(top_3acc) == 3:
        for i in range(3):
            top_3acc[i] += correct

    return correct, total

def log_metrics(optimizer, plosses, vloss, acces, loss_weights, correct, total, top_3acc, phase, wandb_instance):
    """Log metrics for Eagle 3 training"""
    
    # Compute weighted loss
    weighted_loss = sum([loss_weights[i] * plosses[i] for i in range(len(plosses))])
    
    logdict = {
        f"{phase}/lr": optimizer.param_groups[0]["lr"] if phase == "train" else None,
        f"{phase}/vloss": vloss.item() if vloss is not None else None,
        f"{phase}/weighted_loss": weighted_loss.item(),
        f"{phase}/acc": correct / total if total > 0 else 0
    }
    
    # Log individual step losses and accuracies
    for i, (ploss, acc) in enumerate(zip(plosses, acces)):
        logdict[f'{phase}/ploss_step_{i}'] = ploss.item()
        logdict[f'{phase}/acc_step_{i}'] = acc

    # Log top-k accuracies
    for id, i in enumerate(top_3acc):
        logdict[f'{phase}/top_{id + 1}_acc'] = i / total if total > 0 else 0
        
    if wandb_instance:
        wandb_instance.log(logdict)

def run_epoch(args, model, data_loader, optimizer, scheduler, criterion, accelerator, is_warmup,  wandb_instance, train_mode=True):
    """Run one epoch of Eagle 3 training"""
    model.train() if train_mode else model.eval()
    
    top_3acc = [0 for _ in range(3)]
    correct, total = 0, 0
    epoch_loss = 0
    num_batches = 0
    
    # Eagle 3 loss weights (exponential decay)
    loss_weights = [args.eagle3_weight_decay ** i for i in range(args.length)]

    for data in tqdm(data_loader):
        with torch.set_grad_enabled(train_mode):
            if train_mode:
                optimizer.zero_grad()
                
            # Eagle 3 forward pass returns multiple losses and accuracies


            plosses, vlosses, acces = model(
                cond_idx=data["cond_idx"], 
                input_ids=data["input_ids"], 
                attention_mask=data["attention_mask"],
                loss_mask=data["loss_mask"]
            )
            # Compute weighted loss for Eagle 3
            weighted_loss = sum([loss_weights[i] * plosses[i] for i in range(len(plosses))])

            loss = weighted_loss
            
            if train_mode:
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                if is_warmup:
                    scheduler.step()

        with torch.no_grad():
            # Update metrics using Eagle 3 approach
            correct_batch, total_batch = update_metrics(
                plosses, acces, data["loss_mask"], top_3acc
            )
            correct += correct_batch
            total += total_batch

        if accelerator.is_main_process and data["loss_mask"].sum().item() != 0 and train_mode:
            log_metrics(
                optimizer, plosses, None, acces, loss_weights, 
                correct, total, top_3acc, "train", wandb_instance
            )

        epoch_loss += loss.item()
        num_batches += 1

    # Aggregate metrics across processes
    correct = torch.tensor(correct, dtype=torch.float32).to(accelerator.device)
    total = torch.tensor(total, dtype=torch.float32).to(accelerator.device)
    epoch_loss = torch.tensor(epoch_loss, dtype=torch.float32).to(accelerator.device)

    correct, total, epoch_loss = accelerator.gather_for_metrics((correct, total, epoch_loss))
    correct = correct.sum().item()
    total = total.sum().item()
    epoch_loss = epoch_loss.mean()

    top_3acc = [accelerator.gather_for_metrics(torch.tensor(acc, dtype=torch.float32).to(accelerator.device)).sum() for acc in top_3acc]

    return epoch_loss / num_batches, correct, total, top_3acc

class Eagle3TrainingConfig:
    """Configuration for Eagle 3 training"""
    def __init__(self, args):
        self.gradient_checkpointing = False
        self.max_len = args.max_len
        self.length = args.length

def run_train_drafter(args):
    """Main training function for Eagle 3 drafter"""
    if args.cfg_loss and not args.coupled:
        raise ValueError("--cfg_loss can not be activated without --coupled.")

    # Enable anomaly detection to catch gradient computation issues early
    torch.autograd.set_detect_anomaly(True)

    set_seed(0)
    accelerator = Accelerator(
                    mixed_precision='bf16',
                    gradient_accumulation_steps=args.gradient_accumulation_steps,)
    
    wandb_instance = None
    if accelerator.is_main_process:
        wandb.login(key="46d0f4a8c52a34c94859af9091c680cd79990fd6")

        run_name = f"{args.model}_lr{args.lr}_p_w{args.p_w}_bsz{args.bs}_gradacc_{args.gradient_accumulation_steps}"
        run_name += f"_epochs{args.num_epochs}_length{args.length}"
        if args.coupled:
            run_name += "_coupled"
        if args.cfg_loss:
            run_name += f"_cfgloss_cfgscale_{args.cfg_scale}"
        if args.embed_upscale > 1.0:
            run_name += f"_embed_upscale_{args.embed_upscale}"
        run_name += "_mscoco2017train30k"
        wandb_instance = wandb.init(project="eagle3-llamagen2", name=run_name, config=args)

    # Load configuration and model
    if args.model == "lumina_mgpt":
        from models.configs.configuration_lumina_mgpt import ChameleonConfig
        base_config = ChameleonConfig.from_pretrained(args.base_path)
        ModelClass = Eagle3Model
    elif args.model == "anole":
        from models.configs.configuration_anole import ChameleonConfig
        base_config = ChameleonConfig.from_pretrained(args.base_path)
        ModelClass = Eagle3AnoleModel
    elif "llamagen" in args.model:
        from transformers import AutoConfig
        base_config = AutoConfig.from_pretrained(args.base_path)
        from traineagle3.cnets_llamagen import Model
        ModelClass = Model
    else:
        raise ValueError("Invalid model name. Supported: lumina_mgpt, anole, llamagen, llamagen2")

    # Data augmentation
    if args.data_noise == "uniform":
        aug = AddUniformNoise(std=args.std)
    elif args.data_noise == "gaussian":
        aug = AddGaussianNoise(mean=args.mean, std=args.std)
    else:
        aug = None

    data_path = list_files(args.data_dir)
    train_data_path = data_path[:int(len(data_path) * args.train_data_ratio)]
    test_data_path = data_path[int(len(data_path) * args.train_data_ratio):]

    # [SY] Dataset difference:
    # Eagle 2 data preprocessing saves gtp in data['target'], therefore draft training does not need base model fwd pass during training, only calls lm_head on gtp.
    # In eagle 3 draft training, we similarly saved data['target'] as 3k- concat features. but we still use base model fwd due to some test-time paddings.
    # Therefore, base model needs to be passed with cond_idx that customdataset automatically discards for eagle2.
    # To fix, returning cond_idxs in customdataset's get_item call.

    # Create datasets
    if args.coupled:
        if args.model != "lumina_mgpt":
            raise ValueError("--coupled can only be used with lumina_mgpt model.")
        train_dataset = CoupledDataset(train_data_path, max_len=args.max_len, transform=aug)
        test_dataset = CoupledDataset(test_data_path, max_len=args.max_len)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                collate_fn=DataCollatorWithPaddingForCoupled(), num_workers=0,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                collate_fn=DataCollatorWithPaddingForCoupled(), num_workers=0, pin_memory=True)
        
    else:
        
        train_dataset = CustomDataset(train_data_path, max_len=args.max_len, transform=aug, model=args.model)
        test_dataset = CustomDataset(test_data_path, max_len=args.max_len, model=args.model)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                collate_fn=DataCollatorWithPadding(), num_workers=0,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                collate_fn=DataCollatorWithPadding(), num_workers=0, pin_memory=True)
    
    if accelerator.is_main_process:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # Create Eagle 3 model
    config = EConfig.from_pretrained(args.config_path)
    training_config = Eagle3TrainingConfig(args)
   
    model = ModelClass(config,training_config, load_emb=True, path=args.base_path) # only drafter. no base model needs to be loaded since we have gtp saved in dataset
    

    # ckpt_path = "/work1/deming/shared/llamagen/eagle3-drafters/llamagen2-eagle3-lossscaled/llamagen2_lr0.0001_p_w0.1_bsz8_gradacc_1_epochs20_length7_mscoco2017train30k/state_8/model.safetensors"
    # from safetensors.torch import load_file
    # state_dict = load_file(ckpt_path)
    # state_dict["embed_tokens.weight"] = model.target_model.model.embed_tokens.weight
    # # state_dict["target_model.lm_head.weight"] = model.target_model.lm_head.weight
    # model.load_state_dict(state_dict, strict=True)

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    if args.is_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps_ratio * len(train_loader),
                                                    num_training_steps=args.num_epochs * len(train_loader))

        model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, test_loader, scheduler
        )
    else:
        model,  optimizer, train_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, test_loader
        )

    # Training loop
    for epoch in range(0, args.num_epochs):
        epoch_loss, epoch_correct, epoch_total, epoch_top3 = run_epoch(
            args, model, train_loader, optimizer, scheduler, criterion, accelerator, args.is_warmup, wandb_instance, train_mode=True
        )

        if accelerator.is_main_process and wandb_instance is not None:
            log_metrics(
                optimizer, [torch.tensor(epoch_loss)], None, [epoch_correct/epoch_total if epoch_total > 0 else 0], 
                [1.0], epoch_correct, epoch_total, epoch_top3, "epoch", wandb_instance
            )
        
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.num_epochs:
            test_loss, test_correct, test_total, test_top3 = run_epoch(
                args, model, test_loader, optimizer, scheduler, criterion, accelerator, args.is_warmup, wandb_instance, train_mode=False
            )
            
            if accelerator.is_main_process and wandb_instance is not None:
                log_metrics(
                    optimizer, [torch.tensor(test_loss)], None, [test_correct/test_total if test_total > 0 else 0],
                    [1.0], test_correct, test_total, test_top3, "test", wandb_instance
                )

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.num_epochs:
            if accelerator.is_main_process:
                accelerator.save_state(output_dir=f"{args.save_dir}/{run_name}/state_{epoch + 1}")

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    
    run_train_drafter(args)
