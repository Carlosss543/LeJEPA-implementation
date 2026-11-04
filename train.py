import argparse
import torch
from data import get_data_loaders
from models import ViTEncoder
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from engine import train_one_epoch, evaluate
from sigreg import SIGReg
from tqdm import tqdm
import wandb
import os


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")

    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-3) # 2e-3

    parser.add_argument("--batch_size", type=int, default=128) # 256
    parser.add_argument("--num_epochs", type=int, default=300)

    parser.add_argument("--lamb", type=float, default=0.02)
    parser.add_argument("--V", type=int, default=4)
    parser.add_argument("--proj_dim", type=int, default=16)

    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--wandb_run_id", type=str, default=None)

    return parser


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a compatible NVIDIA GPU."
    args.device = torch.device("cuda")

    # --- load data ---
    train_loader, val_loader = get_data_loaders(args, batch_size=args.batch_size)

    # --- initialize models ---
    base_model = ViTEncoder(proj_dim=args.proj_dim).to(args.device)
    model = base_model #torch.compile(base_model)    for some reason compilation seems to slow down training, we are cpu bound anyway
    args.num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    probe = nn.Sequential(
        nn.LayerNorm(512),
        nn.Linear(512, args.num_classes)
    ).to(args.device)

    # --- optimizer ---
    g1 = {"params": model.parameters(), "lr": args.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    optimizer = torch.optim.AdamW([g1, g2], fused=True)

    sigreg = SIGReg().to(args.device)

    # --- lr scheduler ---
    warmup_steps = 1
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - warmup_steps, eta_min=0.5 * args.lr)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # --- if resuming from checkpoint ---
    start_epoch = 1

    if args.resume_from_checkpoint:
        assert args.wandb_run_id is not None, "wandb_run_id must be specified when resuming from checkpoint."
        assert os.path.exists(f"{args.checkpoints_dir}/{args.wandb_run_id}.pt"), f"Checkpoint path '{args.checkpoints_dir}/{args.wandb_run_id}.pt' does not exist."

        checkpoint = torch.load(f"{args.checkpoints_dir}/{args.wandb_run_id}.pt", map_location=args.device)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        probe.load_state_dict(checkpoint["probe_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        del checkpoint # free up memory

        print(f"Resumed training from checkpoint: {args.checkpoints_dir}/{args.wandb_run_id}.pt")

    # --- initialize wandb ---
    run = wandb.init(
        project="LeJEPA",
        config=vars(args),
        mode="online",
        id=args.wandb_run_id if args.resume_from_checkpoint else None,
        resume="must" if args.resume_from_checkpoint else None,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    # --- training loop ---
    for epoch in tqdm(range(start_epoch, args.num_epochs + 1), desc="Epochs"):
        train_one_epoch(args, model, probe, train_loader, optimizer, sigreg, epoch)
        lr_scheduler.step()
        evaluate(args, model, probe, val_loader, epoch)

        if epoch % 10 == 0:
            model_to_save = model.module if hasattr(model, "module") else model # Unwrap the model if it's wrapped by DDP
            model_to_save = model_to_save._orig_mod if hasattr(model_to_save, "_orig_mod") else model_to_save # Unwrap the model if it's wrapped by torch.compile
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "probe_state_dict": probe.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "args": vars(args)
            }
            os.makedirs(args.checkpoints_dir, exist_ok=True)
            torch.save(checkpoint, f"{args.checkpoints_dir}/{run.id}.pt")

    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
