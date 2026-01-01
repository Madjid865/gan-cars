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
import glob
import re

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

# -----------------------
# Trouver le dernier checkpoint
# -----------------------
def find_latest_checkpoint(out_dir: str) -> str:
    """
    Cherche le dernier checkpoint dans out_dir.
    Retourne le chemin ou "" si aucun checkpoint trouvé.
    """
    pattern = os.path.join(out_dir, "ckpt_*.pth")
    ckpts = glob.glob(pattern)
    
    if not ckpts:
        return ""
    
    # Extraire les numéros d'epoch et trouver le max
    def extract_epoch(path):
        match = re.search(r'ckpt_(\d+)\.pth', path)
        return int(match.group(1)) if match else 0
    
    latest = max(ckpts, key=extract_epoch)
    return latest

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
    p.add_argument("--resume", type=str, default="",
                   help="Chemin vers checkpoint. Si vide, cherche automatiquement le dernier dans --out_dir")
    p.add_argument("--force_restart", action="store_true",
                   help="Force à redémarrer de zéro même s'il existe des checkpoints")
    p.add_argument("--plot_every", type=int, default=10,
                   help="Fréquence de génération des graphiques (en epochs)")
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

# -----------------------
# Sauvegarde des historiques de loss
# -----------------------
def save_loss_history(out_dir: str, train_d, train_g, val_d, val_g):
    """Sauvegarde l'historique des loss dans un fichier JSON."""
    import json
    history = {
        "train_loss_d": train_d,
        "train_loss_g": train_g,
        "val_loss_d": val_d,
        "val_loss_g": val_g
    }
    history_path = os.path.join(out_dir, "loss_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)

def load_loss_history(out_dir: str):
    """Charge l'historique des loss depuis le fichier JSON."""
    import json
    history_path = os.path.join(out_dir, "loss_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        return (history["train_loss_d"], history["train_loss_g"],
                history["val_loss_d"], history["val_loss_g"])
    return [], [], [], []

# -----------------------
# Plot des courbes
# -----------------------

"""def plot_losses(plots_dir: str, start_epoch: int, current_epoch: int, 
                train_d, train_g, val_d, val_g):
    
    epochs = range(start_epoch + 1, current_epoch + 1)
    
    # Discriminator
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_d, label="train D", linewidth=2)
    plt.plot(epochs, val_d, label="val D", linewidth=2)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.title(f"Discriminator loss (epoch {start_epoch+1}-{current_epoch})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    d_path = f"{plots_dir}/loss_discriminator.png"
    plt.savefig(d_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Generator
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_g, label="train G", linewidth=2)
    plt.plot(epochs, val_g, label="val G", linewidth=2)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.title(f"Generator loss (epoch {start_epoch+1}-{current_epoch})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    g_path = f"{plots_dir}/loss_generator.png"
    plt.savefig(g_path, dpi=150, bbox_inches="tight")
    plt.close()
"""
def plot_losses(out_dir, start_epoch, end_epoch, train_d, train_g, val_d, val_g):
    import matplotlib.pyplot as plt
    import os

    # Construire l’axe x à partir de la longueur des listes
    n = max(len(train_d), len(train_g), len(val_d), len(val_g))
    epochs = list(range(1, n + 1))

    # Tronquer/aligner proprement (au cas où)
    train_d = train_d[:n]
    train_g = train_g[:n]
    val_d   = val_d[:n]
    val_g   = val_g[:n]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_d, label="train D", linewidth=2)
    plt.plot(epochs, val_d,   label="val D",   linewidth=2)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Discriminator")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_discriminator.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_g, label="train G", linewidth=2)
    plt.plot(epochs, val_g,   label="val G",   linewidth=2)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Generator")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_generator.png"), dpi=150, bbox_inches="tight")
    plt.close()

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
    fixed_noise = torch.randn(16, args.latent_dim, 1, 1, device=device)

    # AUTO-REPRISE : Cherche automatiquement le dernier checkpoint
    start_epoch = 0
    resume_path = args.resume
    
    if args.force_restart:
        print("🔄 [Force Restart] Redémarre de zéro (--force_restart activé)")
        train_loss_d_hist, train_loss_g_hist = [], []
        val_loss_d_hist, val_loss_g_hist = [], []
    else:
        # Si pas de --resume fourni, cherche automatiquement
        if not resume_path:
            resume_path = find_latest_checkpoint(args.out_dir)
        
        # Charge le checkpoint s'il existe
        if resume_path and Path(resume_path).is_file():
            try:
                ckpt = torch.load(resume_path, map_location="cpu")
                G.load_state_dict(ckpt["G"])
                D.load_state_dict(ckpt["D"])
                opt_g.load_state_dict(ckpt["opt_g"])
                opt_d.load_state_dict(ckpt["opt_d"])
                start_epoch = ckpt["epoch"]
                
                # Charge aussi l'historique des loss
                train_loss_d_hist, train_loss_g_hist, val_loss_d_hist, val_loss_g_hist = load_loss_history(args.out_dir)
                
                print(f"✅ [Auto-Resume] Chargé: {resume_path}")
                print(f"   └─ Reprend à epoch {start_epoch}/{args.epochs}")
                print(f"   └─ Historique de loss restauré ({len(train_loss_d_hist)} epochs)")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {resume_path}: {e}")
                print("   └─ Démarre un nouvel entraînement")
                start_epoch = 0
                train_loss_d_hist, train_loss_g_hist = [], []
                val_loss_d_hist, val_loss_g_hist = [], []
        else:
            print("✨ [Nouveau entraînement] Aucun checkpoint trouvé, démarre à epoch 0")
            train_loss_d_hist, train_loss_g_hist = [], []
            val_loss_d_hist, val_loss_g_hist = [], []

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
                          nrow=4, normalize=True, value_range=(-1, 1))

        # Checkpoint léger
        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch + 1
        }, f"{args.out_dir}/ckpt_{epoch+1:03d}.pth")
        
        # Sauvegarde l'historique des loss à chaque epoch (pour reprise)
        save_loss_history(args.out_dir, train_loss_d_hist, train_loss_g_hist, 
                         val_loss_d_hist, val_loss_g_hist)
        
        # Plot des courbes selon la fréquence définie ET au dernier epoch
        if (epoch + 1) % args.plot_every == 0 or (epoch + 1) == args.epochs:
            plot_losses(args.plots_dir, start_epoch, epoch + 1,
                       train_loss_d_hist, train_loss_g_hist,
                       val_loss_d_hist, val_loss_g_hist)
            print(f"   📊 Graphiques mis à jour -> {args.plots_dir}/")

    # Poids finaux
    torch.save(G.state_dict(), f"{args.out_dir}/generator_final.pth")
    torch.save(D.state_dict(), f"{args.out_dir}/discriminator_final.pth")
    print("✔ Entraînement terminé. Poids enregistrés dans", args.out_dir)
    print(f"✔ Graphiques finaux dans {args.plots_dir}/")

if __name__ == "__main__":
    main()