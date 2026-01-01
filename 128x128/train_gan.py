# ===============================
# train_gan.py (stable-resume)
# - Resume compatible avec tes checkpoints existants (G/D inchangés)
# - Corrige: AdamW->Adam + weight_decay=0, AMP optionnel (OFF par défaut),
#           d_updates_per_batch, grad_clip, label_fake=0, R1 lazy, EMA, nan-guard
# ===============================

import os
import re
import copy
import math
import json
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

from torchvision import datasets, transforms, utils as vutils

from gan_discriminator import Discriminator


# -----------------------------
# Generator (INCHANGÉ)
# -----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, base_ch: int = 64, img_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_ch = base_ch
        self.img_size = img_size

        if img_size == 64:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, base_ch * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(base_ch * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_ch * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_ch * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        elif img_size == 128:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, base_ch * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(base_ch * 16),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_ch * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_ch * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_ch * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(True),

                nn.ConvTranspose2d(base_ch, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        else:
            raise ValueError("img_size doit être 64 ou 128 (dans ce script).")

    def forward(self, z):
        return self.main(z)


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_last_ckpt(out_dir: str) -> str:
    if not os.path.isdir(out_dir):
        return ""
    files = [f for f in os.listdir(out_dir) if re.match(r"ckpt_\d+\.pth", f)]
    if not files:
        return ""
    files.sort()
    return os.path.join(out_dir, files[-1])


def evaluate_val(G, D, val_loader, device, criterion, latent_dim, label_real: float = 0.9, label_fake: float = 0.0):
    G.eval()
    D.eval()
    loss_d_total = 0.0
    loss_g_total = 0.0
    n = 0

    with torch.no_grad():
        for real_imgs, _ in val_loader:
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)

            labels_real = torch.full((bsz,), label_real, device=device)
            labels_fake = torch.full((bsz,), label_fake, device=device)

            logits_real = D(real_imgs)
            loss_real = criterion(logits_real, labels_real)

            z = torch.randn(bsz, latent_dim, 1, 1, device=device)
            fake_imgs = G(z)
            logits_fake = D(fake_imgs.detach())
            loss_fake = criterion(logits_fake, labels_fake)

            loss_d = loss_real + loss_fake

            logits_for_g = D(fake_imgs)
            loss_g = criterion(logits_for_g, labels_real)

            loss_d_total += loss_d.item() * bsz
            loss_g_total += loss_g.item() * bsz
            n += bsz

    return loss_d_total / max(n, 1), loss_g_total / max(n, 1)


def make_loaders(data_root: str, img_size: int, batch_size: int, num_workers: int, val_ratio: float, no_colorjitter: bool = False):
    tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if not no_colorjitter:
        tfms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    tfm = transforms.Compose(tfms)

    ds = datasets.ImageFolder(root=data_root, transform=tfm)
    n_total = len(ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)

    p.add_argument("--g_updates_per_batch", type=int, default=1)

    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--samples_dir", type=str, default="samples")
    p.add_argument("--plots_dir", type=str, default="plots")
    p.add_argument("--plot_every", type=int, default=10)

    p.add_argument("--resume", type=str, default="")
    # ---- Ajouts stabilité ----
    p.add_argument("--d_updates_per_batch", type=int, default=1,
                   help="Nombre de updates Discriminator par batch (1 = standard)")
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="Clip global L2 des gradients (0 = désactivé)")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Weight decay (0 = désactivé). IMPORTANT: éviter AdamW par défaut")
    p.add_argument("--amp", action="store_true",
                   help="Active l'AMP (mixed precision). Si tu as des NaN, désactive (par défaut OFF).")
    p.add_argument("--reset_optim", action="store_true",
                   help="Ignore l'état des optimiseurs du checkpoint (reprend les poids seulement).")
    p.add_argument("--label_real", type=float, default=0.9,
                   help="Label pour réels (one-sided smoothing recommandé: 0.9)")
    p.add_argument("--label_fake", type=float, default=0.0,
                   help="Label pour faux (met 0.0, éviter 0.1)")
    p.add_argument("--no_colorjitter", action="store_true",
                   help="Désactive ColorJitter (utile si ça dégrade / instabilise)")
    p.add_argument("--r1_gamma", type=float, default=0.0,
                   help="R1 penalty (0 = off). Exemple: 2.0 à 10.0")
    p.add_argument("--r1_every", type=int, default=16,
                   help="Fréquence du R1 (lazy regularization).")
    p.add_argument("--ema", action="store_true",
                   help="Active EMA du générateur (échantillons plus stables).")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="Decay EMA (0.999 ou 0.9995)")
    p.add_argument("--instance_noise", type=float, default=0.0,
                   help="Bruit gaussien ajouté aux images réelles/fakes pour stabiliser D (0 = off). Ex: 0.05")
    return p.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device} | img_size={args.img_size} | latent_dim={args.latent_dim} | base_ch={args.base_ch}")

    ensure_dir(args.out_dir)
    ensure_dir(args.samples_dir)
    ensure_dir(args.plots_dir)

    train_loader, val_loader = make_loaders(args.data_root, args.img_size, args.batch_size,
                                            args.num_workers, args.val_ratio, args.no_colorjitter)

    G = Generator(latent_dim=args.latent_dim, base_ch=args.base_ch, img_size=args.img_size).to(device)
    D = Discriminator(img_size=args.img_size, base_ch=args.base_ch).to(device)

    # EMA (optionnel)
    if args.ema:
        G_ema = copy.deepcopy(G).to(device)
        G_ema.eval()
        for p in G_ema.parameters():
            p.requires_grad_(False)
    else:
        G_ema = None

    criterion = nn.BCEWithLogitsLoss()

    # Adam (pas AdamW par défaut)
    opt_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    opt_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # AMP (OFF par défaut)
    use_amp = bool(args.amp) and (device == "cuda")
    scaler_g = GradScaler(device="cuda", enabled=use_amp)
    scaler_d = GradScaler(device="cuda", enabled=use_amp)

    start_epoch = 0

    # Resume
    if args.resume:
        resume_path = args.resume
        if resume_path.lower() == "auto":
            resume_path = find_last_ckpt(args.out_dir)
        if resume_path and os.path.isfile(resume_path):
            print(f"✅ Resume: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            G.load_state_dict(ckpt["G"], strict=True)
            D.load_state_dict(ckpt["D"], strict=True)

            if not args.reset_optim:
                if "opt_g" in ckpt: opt_g.load_state_dict(ckpt["opt_g"])
                if "opt_d" in ckpt: opt_d.load_state_dict(ckpt["opt_d"])
            else:
                print("   └─ reset_optim: état des optimiseurs ignoré (poids seulement)")

            start_epoch = ckpt["epoch"]

            # EMA
            if args.ema:
                if "G_ema" in ckpt and G_ema is not None:
                    G_ema.load_state_dict(ckpt["G_ema"])
                elif G_ema is not None:
                    G_ema.load_state_dict(G.state_dict())
        else:
            print(f"⚠️ resume introuvable: {args.resume}")

    fixed_noise = torch.randn(16, args.latent_dim, 1, 1, device=device)

    history = {"epoch": [], "train_d": [], "train_g": [], "val_d": [], "val_g": []}

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()

        running_d = 0.0
        running_g = 0.0
        seen = 0

        for step, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)

            # ========== D ==========
            loss_d = None
            for _d in range(args.d_updates_per_batch):
                D.zero_grad(set_to_none=True)
                real_labels = torch.full((bsz,), args.label_real, device=device)
                fake_labels = torch.full((bsz,), args.label_fake, device=device)

                real_in = real_imgs
                if args.instance_noise > 0:
                    real_in = (real_imgs + torch.randn_like(real_imgs) * args.instance_noise).clamp(-1, 1)

                with autocast(device_type="cuda", enabled=use_amp):
                    logits_real = D(real_in)
                    loss_real = criterion(logits_real, real_labels)

                    z = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
                    fake_imgs = G(z).detach()
                    if args.instance_noise > 0:
                        fake_imgs = (fake_imgs + torch.randn_like(fake_imgs) * args.instance_noise).clamp(-1, 1)

                    logits_fake = D(fake_imgs)
                    loss_fake = criterion(logits_fake, fake_labels)

                    loss_d = loss_real + loss_fake

                    # R1 penalty (lazy)
                    if args.r1_gamma > 0 and (global_step % args.r1_every == 0):
                        real_r1 = real_imgs.detach().requires_grad_(True)
                        logits_r1 = D(real_r1)
                        grad = torch.autograd.grad(outputs=logits_r1.sum(), inputs=real_r1, create_graph=True)[0]
                        r1_pen = (grad.view(bsz, -1).pow(2).sum(1)).mean() * (args.r1_gamma / 2.0)
                        loss_d = loss_d + r1_pen

                if not torch.isfinite(loss_d.detach()):
                    raise FloatingPointError(f"Non-finite loss_d at epoch={epoch+1} step={step} (loss_d={loss_d.item()})")

                scaler_d.scale(loss_d).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                scaler_d.step(opt_d)
                scaler_d.update()

            # ========== G ==========
            loss_g = None
            for _g in range(args.g_updates_per_batch):
                G.zero_grad(set_to_none=True)
                z = torch.randn(bsz, args.latent_dim, 1, 1, device=device)

                with autocast(device_type="cuda", enabled=use_amp):
                    gen_imgs = G(z)
                    logits = D(gen_imgs)
                    loss_g = criterion(logits, real_labels)

                if not torch.isfinite(loss_g.detach()):
                    raise FloatingPointError(f"Non-finite loss_g at epoch={epoch+1} step={step} (loss_g={loss_g.item()})")

                scaler_g.scale(loss_g).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler_g.unscale_(opt_g)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
                scaler_g.step(opt_g)
                scaler_g.update()

                # EMA update
                if args.ema and G_ema is not None:
                    with torch.no_grad():
                        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                            p_ema.mul_(args.ema_decay).add_(p, alpha=1.0 - args.ema_decay)

            running_d += float(loss_d.item())
            running_g += float(loss_g.item())
            seen += bsz
            global_step += 1

            if step % 50 == 0:
                print(f"[Epoch {epoch+1:03d}/{args.epochs}] Step {step:04d}/{len(train_loader)} | D={loss_d.item():.4f} G={loss_g.item():.4f}")

        train_loss_d = running_d / max(len(train_loader), 1)
        train_loss_g = running_g / max(len(train_loader), 1)

        val_loss_d, val_loss_g = evaluate_val(G, D, val_loader, device, criterion, args.latent_dim, args.label_real, args.label_fake)

        history["epoch"].append(epoch + 1)
        history["train_d"].append(train_loss_d)
        history["train_g"].append(train_loss_g)
        history["val_d"].append(val_loss_d)
        history["val_g"].append(val_loss_g)

        print(f"[Epoch {epoch+1:03d}] Train D={train_loss_d:.4f} | G={train_loss_g:.4f}  ||  Val D={val_loss_d:.4f} | G={val_loss_g:.4f}")

        # Samples visuels (EMA + normal)
        with torch.no_grad():
            if args.ema and G_ema is not None:
                G_ema.eval()
                samples = G_ema(fixed_noise).detach().cpu()
                vutils.save_image(samples, f"{args.samples_dir}/epoch_{epoch+1:03d}_EMA.png",
                                  nrow=4, normalize=True, value_range=(-1, 1))

            G.eval()
            samples = G(fixed_noise).detach().cpu()
            vutils.save_image(samples, f"{args.samples_dir}/epoch_{epoch+1:03d}.png",
                              nrow=4, normalize=True, value_range=(-1, 1))

        # Save ckpt
        ckpt_obj = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch + 1,
            "meta": {
                "img_size": args.img_size,
                "latent_dim": args.latent_dim,
                "base_ch": args.base_ch,
            }
        }
        if args.ema and G_ema is not None:
            ckpt_obj["G_ema"] = G_ema.state_dict()

        torch.save(ckpt_obj, f"{args.out_dir}/ckpt_{epoch+1:03d}.pth")

        # Save history
        with open(os.path.join(args.plots_dir, "loss_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # Save final
    torch.save(G.state_dict(), os.path.join(args.out_dir, "generator_final.pth"))
    torch.save(D.state_dict(), os.path.join(args.out_dir, "discriminator_final.pth"))
    if args.ema and G_ema is not None:
        torch.save(G_ema.state_dict(), os.path.join(args.out_dir, "generator_ema_final.pth"))


if __name__ == "__main__":
    main()
