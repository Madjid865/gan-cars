import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    """
    DCGAN-like Discriminator (64x64 ou 128x128, 3 canaux)
    qui renvoie des LOGITS (pas de Sigmoid).
    Spectral Normalization pour stabiliser l'entraînement.
    """
    def __init__(self, img_channels: int = 3, base_ch: int = 64, img_size: int = 64):
        super().__init__()

        if img_size == 64:
            # Archi d'origine 64x64
            self.main = nn.Sequential(
                # 64x64 -> 32x32
                spectral_norm(nn.Conv2d(img_channels, base_ch, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),

                # 32x32 -> 16x16
                spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(base_ch * 2),
                nn.LeakyReLU(0.2, inplace=True),

                # 16x16 -> 8x8
                spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(base_ch * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # 8x8 -> 4x4
                spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(base_ch * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # 4x4 -> 1x1 (logit)
                spectral_norm(nn.Conv2d(base_ch * 8, 1, 4, 1, 0, bias=False)),
            )
        elif img_size == 128:
            # Nouvelle archi 128x128 -> ... -> 1x1
            self.main = nn.Sequential(
                # 128x128 -> 64x64
                spectral_norm(nn.Conv2d(img_channels, base_ch, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),

                # 64x64 -> 32x32
                spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(base_ch * 2),
                nn.LeakyReLU(0.2, inplace=True),

                # 32x32 -> 16x16
                spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(base_ch * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # 16x16 -> 8x8
                spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(base_ch * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # 8x8 -> 4x4
                spectral_norm(nn.Conv2d(base_ch * 8, base_ch * 16, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(base_ch * 16),
                nn.LeakyReLU(0.2, inplace=True),

                # 4x4 -> 1x1 (logit)
                spectral_norm(nn.Conv2d(base_ch * 16, 1, 4, 1, 0, bias=False)),
            )
        else:
            raise ValueError(f"img_size must be 64 or 128, got {img_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sortie: [batch] (logits)
        return self.main(x).view(-1)
