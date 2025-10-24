# train_gan.py
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils as vutils

# --- pour sauvegarder les graphes sans affichage ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gan_discriminator import Discriminator

# -----------------------
# Générateur DCGAN (64x64)
# -----------------------
class Generator(nn.Module):
    """
    DCGAN-like Generator (64x64, 3 canaux).
    Entrée: bruit z ~ N(0,1) de taille LATENT_DIM.
    """
    def __init__(self, latent_dim: int = 100, img_channels: int = 3, base_ch: int = 64):
        super().__init__()
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

            nn.ConvTranspose2d(base_ch, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # [-1,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)

# -------------
# Entrée / Args
# -------------
def get_args():
    p = argparse.ArgumentParser(description="DCGAN 64x64 - entraînement (AMP, BCEWithLogits) + courbes")
    p.add_argument("--data_root", type=str, default="data/cars/train",
                   help="Racine ImageFolder (met toutes tes images réelles dans un sous-dossier, ex: data/cars/train/real/*.jpg)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--g_updates_per_batch", type=int, default=1)
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="part du dataset pour la validation")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--samples_dir", type=str, default="generated_samples")
    p.add_argument("--plots_dir", type=str, default="plots")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()

# ---------------
# Prépare dataset
# ---------------
def make_loaders(data_root: str, img_size: int, batch_size: int, num_workers: int, val_ratio: float):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = datasets.ImageFolder(root=data_root, transform=tfm)

    n_total = len(ds)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return train_loader, val_loader

# -----------------------
# Évaluation sur le "val"
# -----------------------
@torch.no_grad()
def evaluate_val(G, D, val_loader, device, criterion, latent_dim):
    D.eval(); G.eval()
    running_d, running_g, count = 0.0, 0.0, 0
    for real_imgs, _ in val_loader:
        real_imgs = real_imgs.to(device, non_blocking=True)
        bsz = real_imgs.size(0)

        # Discriminateur: réel vs faux (avec faux générés pour matcher la taille)
        logits_real = D(real_imgs)
        labels_real = torch.full((bsz,), 0.9, device=device)
        loss_real = criterion(logits_real, labels_real)

        z = torch.randn(bsz, latent_dim, 1, 1, device=device)
        fake_imgs = G(z)
        logits_fake = D(fake_imgs)
        labels_fake = torch.full((bsz,), 0.1, device=device)
        loss_fake = criterion(logits_fake, labels_fake)

        loss_d = loss_real + loss_fake

        # Générateur: veut tromper D -> labels "réel"
        logits_for_g = D(fake_imgs)
        loss_g = criterion(logits_for_g, labels_real)

        running_d += loss_d.item() * bsz
        running_g += loss_g.item() * bsz
        count += bsz

    return running_d / count, running_g / count

# -------
# Train
# -------
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Data
    train_loader, val_loader = make_loaders(args.data_root, args.img_size, args.batch_size, args.num_workers, args.val_ratio)

    # Modèles
    G = Generator(args.latent_dim, 3, 64).to(device)
    D = Discriminator(3, 64).to(device)

    # Optim & loss (logits -> BCEWithLogits)
    opt_g = optim.AdamW(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = optim.AdamW(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    criterion = nn.BCEWithLogitsLoss()

    # AMP
    scaler_g = GradScaler(device="cuda")
    scaler_d = GradScaler(device="cuda")

    # Bruit fixe pour suivi visuel
    torch.manual_seed(1337)
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    # Reprise éventuelle (CORRECTION ICI)
    start_epoch = 0  # Valeur par défaut pour nouvel entraînement
    
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location="cpu")
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"]
        print(f"✔ [Resume] Chargé: {args.resume}, reprend à epoch {start_epoch}")
    else:
        print(f"✔ [Nouveau entraînement] Démarre à epoch 0")

    # Logs pour les courbes
    train_loss_d_hist, train_loss_g_hist = [], []
    val_loss_d_hist,   val_loss_g_hist   = [], []

    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        running_d, running_g, seen = 0.0, 0.0, 0

        for step, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device, non_blocking=True)
            bsz = real_imgs.size(0)

            # ========== D ==========
            D.zero_grad(set_to_none=True)
            real_labels = torch.full((bsz,), 0.9, device=device)
            fake_labels = torch.full((bsz,), 0.1, device=device)

            with autocast("cuda"):
                logits_real = D(real_imgs)
                loss_real = criterion(logits_real, real_labels)

                z = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
                fake_imgs = G(z).detach()
                logits_fake = D(fake_imgs)
                loss_fake = criterion(logits_fake, fake_labels)

                loss_d = loss_real + loss_fake

                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
           
            # ========== G ==========
            for _ in range(args.g_updates_per_batch):
                G.zero_grad(set_to_none=True)
                z = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
                with autocast("cuda"):
                    gen_imgs = G(z)
                    logits = D(gen_imgs)
                    loss_g = criterion(logits, real_labels)

                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g)
                scaler_g.update()

            # accumulate train losses (moyenne pondérée par bsz)
            running_d += loss_d.item() * bsz
            running_g += loss_g.item() * bsz
            seen += bsz

            if step % 50 == 0:
                print(f"[Epoch {epoch+1:03d}/{args.epochs}] Step {step:04d}/{len(train_loader)} | "
                      f"Loss_D={loss_d.item():.4f}  Loss_G={loss_g.item():.4f}")

        # moyenne par epoch (train)
        train_loss_d = running_d / seen
        train_loss_g = running_g / seen
        train_loss_d_hist.append(train_loss_d)
        train_loss_g_hist.append(train_loss_g)

        # --- validation ---
        val_loss_d, val_loss_g = evaluate_val(G, D, val_loader, device, criterion, args.latent_dim)
        val_loss_d_hist.append(val_loss_d)
        val_loss_g_hist.append(val_loss_g)

        print(f"[Epoch {epoch+1:03d}] Train D={train_loss_d:.4f} | G={train_loss_g:.4f}  ||  "
              f"Val D={val_loss_d:.4f} | G={val_loss_g:.4f}")

        # Samples visuels
        with torch.no_grad():
            G.eval()
            samples = G(fixed_noise).detach().cpu()
        vutils.save_image(samples, f"{args.samples_dir}/epoch_{epoch+1:03d}.png",
                          nrow=8, normalize=True, value_range=(-1, 1))

        # Checkpoint léger
        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch + 1
        }, f"{args.out_dir}/ckpt_{epoch+1:03d}.pth")

    # Poids finaux
    torch.save(G.state_dict(), f"{args.out_dir}/generator_final.pth")
    torch.save(D.state_dict(), f"{args.out_dir}/discriminator_final.pth")
    print("✔ Entraînement terminé. Poids enregistrés dans", args.out_dir)

    # -----------------
    # GRAPHE 1 : Discri
    # -----------------
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_loss_d_hist, label="train D")
    plt.plot(range(1, args.epochs + 1), val_loss_d_hist,   label="val D")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Discriminator loss (train vs val)")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
    d_path = f"{args.plots_dir}/loss_discriminator.png"
    plt.savefig(d_path, dpi=150, bbox_inches="tight")
    print(f"✔ Graphe Discriminator -> {d_path}")

    # -----------------
    # GRAPHE 2 : Générateur
    # -----------------
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_loss_g_hist, label="train G")
    plt.plot(range(1, args.epochs + 1), val_loss_g_hist,   label="val G")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Generator loss (train vs val)")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
    g_path = f"{args.plots_dir}/loss_generator.png"
    plt.savefig(g_path, dpi=150, bbox_inches="tight")
    print(f"✔ Graphe Generator -> {g_path}")

if __name__ == "__main__":
    main()