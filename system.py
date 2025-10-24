# system.py
import argparse
from pathlib import Path
import torch
from torchvision import transforms, utils as vutils
from PIL import Image

from train_gan import Generator
from gan_discriminator import Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_generator(weights: str, latent_dim: int = 100) -> Generator:
    G = Generator(latent_dim=latent_dim, img_channels=3, base_ch=64).to(device)
    sd = torch.load(weights, map_location="cpu")
    # accepter checkpoint (dict) ou poids bruts (state_dict)
    if isinstance(sd, dict) and "G" in sd:
        sd = sd["G"]
    G.load_state_dict(sd, strict=True)
    G.eval()
    return G

def load_discriminator(weights: str) -> Discriminator:
    D = Discriminator(img_channels=3, base_ch=64).to(device)
    sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "D" in sd:
        sd = sd["D"]
    D.load_state_dict(sd, strict=True)
    D.eval()
    return D

@torch.no_grad()
def cmd_generate(args):
    G = load_generator(args.gen, latent_dim=args.latent_dim)
    torch.manual_seed(args.seed)
    z = torch.randn(args.n, args.latent_dim, 1, 1, device=device)
    imgs = G(z).cpu()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(imgs, args.out, nrow=args.nrow, normalize=True, value_range=(-1, 1))
    print(f"✔ Images générées -> {args.out}")

@torch.no_grad()
def cmd_score(args):
    D = load_discriminator(args.disc)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = D(x)                 # logits
    prob = torch.sigmoid(logits)  # proba [0,1] juste pour affichage
    print(f"Logit: {logits.item():.4f} | Score(sigmoid): {prob.item():.4f}")

def get_args():
    p = argparse.ArgumentParser(description="Outils génération/score pour GAN voitures")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Générer une grille d'images")
    g.add_argument("--gen", type=str, default="checkpoints/generator_final.pth",
                   help="poids du générateur")
    g.add_argument("--out", type=str, default="generated_samples/samples.png")
    g.add_argument("--n", type=int, default=64, help="nombre d'images")
    g.add_argument("--nrow", type=int, default=8, help="images par ligne")
    g.add_argument("--latent_dim", type=int, default=100)
    g.add_argument("--seed", type=int, default=123)

    s = sub.add_parser("score", help="Évaluer une image par le Discriminateur")
    s.add_argument("--disc", type=str, default="checkpoints/discriminator_final.pth",
                   help="poids du discriminateur")
    s.add_argument("--image", type=str, required=True, help="chemin de l'image à évaluer")
    s.add_argument("--img_size", type=int, default=64)

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "score":
        cmd_score(args)
