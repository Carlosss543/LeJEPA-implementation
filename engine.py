import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast
import wandb


def train_one_epoch(args, model, probe, train_loader, optimizer, sigreg, epoch):
    model.train(), probe.train()

    for views, labels in tqdm(train_loader, total=len(train_loader)):
        views, labels = views.to(args.device), labels.to(args.device)

        with autocast("cuda", dtype=torch.bfloat16):
            emb, proj = model(views)

            # LeJEPA loss
            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = sigreg(proj)
            lejepa_loss = args.lamb * sigreg_loss + (1 - args.lamb) * inv_loss

            # linear probe
            y_hat = probe(emb.detach())
            y_rep = labels.repeat_interleave(args.V)
            probe_loss = F.cross_entropy(y_hat, y_rep)

            loss = lejepa_loss + probe_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log(
            {
                "epoch": epoch,
                "train/inv": inv_loss.item(),
                "train/sigreg": sigreg_loss.item(),
                "train/lejepa": lejepa_loss.item(),
                "train/probe": probe_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            }
        )


@torch.inference_mode()
def evaluate(args, model, probe, val_loader, epoch):
    model.eval(), probe.eval()

    total_correct = 0

    for views, labels in val_loader:
        views, labels = views.to(args.device), labels.to(args.device)

        with autocast("cuda", dtype=torch.bfloat16):
            emb, _ = model(views)
            y_hat = probe(emb)

            total_correct += (y_hat.argmax(1) == labels).sum().item()

    val_acc = total_correct / len(val_loader.dataset)

    wandb.log(
        {
            "epoch": epoch,
            "val/acc": val_acc, 
        }
    )
