import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration OPTIMISÉE et ÉQUILIBRÉE
IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 32              # Réduit de 64 à 32 pour plus de stabilité
LEARNING_RATE_G = 0.0002     # Learning rate du Générateur
LEARNING_RATE_D = 0.00005    # Learning rate du Discriminateur (4x plus lent!)
EPOCHS = 150                 # Augmenté de 100 à 150
MAX_IMAGES = 5000            # Augmenté de 3000 à 5000

class Generator(nn.Module):
    """Générateur du GAN - crée des images de voitures à partir de bruit aléatoire"""
    def __init__(self, latent_dim=100, img_channels=3):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 8 x 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 16 x 16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 32 x 32
            
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 64 x 64
        )
    
    def forward(self, z):
        return self.model(z)

class CarDataset(Dataset):
    """Dataset personnalisé pour charger les images de voitures depuis plusieurs dossiers"""
    def __init__(self, root_dirs, transform=None, max_images=5000):
        """
        Args:
            root_dirs: Peut être un string (1 dossier) ou une liste (plusieurs dossiers)
            transform: Transformations à appliquer
            max_images: Nombre maximum d'images à charger (défaut: 5000)
        """
        self.transform = transform
        self.images = []
        
        # Si c'est un seul chemin, le convertir en liste
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        
        # Parcourir tous les dossiers RÉCURSIVEMENT
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                print(f"ATTENTION: {root_dir} n'existe pas!")
                continue
            
            print(f"🔍 Recherche d'images dans: {root_dir}")
            
            # Parcourir récursivement tous les sous-dossiers avec os.walk()
            for dirpath, dirnames, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        full_path = os.path.join(dirpath, filename)
                        self.images.append(full_path)
                        
                        # Limiter le nombre d'images
                        if len(self.images) >= max_images:
                            print(f"⚠️  Limite de {max_images} images atteinte, arrêt du chargement")
                            break
                
                if len(self.images) >= max_images:
                    break
            
            if len(self.images) >= max_images:
                break
        
        if len(self.images) == 0:
            raise ValueError("Aucune image trouvée dans les dossiers spécifiés!")
        
        print(f"✓ {len(self.images)} images chargées (max: {max_images})")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"Erreur lors du chargement de {img_path}: {e}")
            # Retourner une image noire en cas d'erreur
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)

def train_generator(dataset_path, save_dir='models'):
    """Entraîne le générateur du GAN avec configuration équilibrée"""
    
    # Créer le dossier de sauvegarde
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('generated_samples', exist_ok=True)
    
    # Définir le device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Utilisation du device: {device}")
    
    # Transformations pour les images
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Charger le dataset avec limite d'images
    dataset = CarDataset(dataset_path, transform=transform, max_images=MAX_IMAGES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"✓ Dataset chargé: {len(dataset)} images")
    print(f"✓ Nombre de batches: {len(dataloader)}")
    
    # Initialiser le générateur et le discriminateur
    from gan_discriminator import Discriminator
    
    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    # Initialisation des poids
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print("\n" + "="*70)
    print("CONFIGURATION D'ENTRAÎNEMENT OPTIMISÉE")
    print("="*70)
    print(f"Images: {MAX_IMAGES}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning Rate Générateur: {LEARNING_RATE_G}")
    print(f"Learning Rate Discriminateur: {LEARNING_RATE_D} (4x plus lent!)")
    print(f"Label smoothing: Activé (0.9/0.1)")
    print(f"Ratio G/D: 2:1 (générateur entraîné 2x plus)")
    print("="*70 + "\n")
    
    # Optimiseurs avec learning rates DIFFÉRENTS
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
    
    # Fonction de perte
    criterion = nn.BCELoss()
    
    # Labels avec SMOOTHING pour rendre le discriminateur moins sûr de lui
    real_label = 0.9  # Au lieu de 1.0
    fake_label = 0.1  # Au lieu de 0.0
    
    # Vecteur fixe pour visualiser la progression
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    
    print("🚀 Début de l'entraînement...\n")
    
    for epoch in range(EPOCHS):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # ============================================================
            # ENTRAÎNER LE DISCRIMINATEUR (1 fois par batch)
            # ============================================================
            discriminator.zero_grad()
            
            # Images réelles
            labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_images).view(-1)
            loss_d_real = criterion(output, labels)
            loss_d_real.backward()
            
            # Images fausses
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            labels.fill_(fake_label)
            output = discriminator(fake_images.detach()).view(-1)
            loss_d_fake = criterion(output, labels)
            loss_d_fake.backward()
            
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()
            
            # ============================================================
            # ENTRAÎNER LE GÉNÉRATEUR (2 fois par batch pour compenser)
            # ============================================================
            for _ in range(2):  # Entraîner 2 fois le générateur
                generator.zero_grad()
                
                # Générer de nouvelles images
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                fake_images = generator(noise)
                
                # Le générateur veut tromper le discriminateur
                labels.fill_(real_label)
                output = discriminator(fake_images).view(-1)
                loss_g = criterion(output, labels)
                loss_g.backward()
                optimizer_g.step()
            
            # Afficher les statistiques
            if i % 20 == 0:  # Afficher plus souvent
                print(f'Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{len(dataloader)}] '
                      f'Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}')
                
                # Avertissement si déséquilibre détecté
                if loss_d.item() < 0.3:
                    print("Discriminateur trop fort!")
                if loss_g.item() > 4.0:
                    print("Générateur en difficulté!")
        
        # Sauvegarder des exemples générés
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                
                # Sauvegarder une grille d'images
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title(f"Images générées - Epoch {epoch+1}")
                plt.imshow(np.transpose(
                    torch.cat([fake[i] for i in range(min(16, len(fake)))], dim=2),
                    (1, 2, 0)
                ) * 0.5 + 0.5)
                plt.savefig(f'generated_samples/epoch_{epoch+1}.png')
                plt.close()
            
            # Sauvegarder les modèles
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, f'{save_dir}/gan_checkpoint_epoch_{epoch+1}.pth')
            
            print(f"✓ Checkpoint sauvegardé à l'epoch {epoch+1}")
    
    # Sauvegarder les modèles finaux
    torch.save(generator.state_dict(), f'{save_dir}/generator_final.pth')
    torch.save(discriminator.state_dict(), f'{save_dir}/discriminator_final.pth')
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ!")
    print("="*70)
    print(f"Modèles sauvegardés dans: {save_dir}/")
    print(f"Échantillons dans: generated_samples/")
    print("="*70)
    
    return generator, discriminator

if __name__ == "__main__":
    # ============================================================
    # CONFIGURATION DES CHEMINS VERS VOS DATASETS
    # ============================================================
    
    # OPTION 1 : Un seul dossier (cherchera dans tous les sous-dossiers)
    DATASET_PATH = "/home/madjid/Documents/gan_cars_project/training/train"
    
    # OPTION 2 : Plusieurs dossiers différents (décommentez si besoin)
    # DATASET_PATH = [
    #     "C:/Users/naimi/Documents/archive/train",
    #     "C:/Users/naimi/Documents/autre_dossier"
    # ]
    
    # ============================================================
    # Vérification et lancement de l'entraînement
    # ============================================================
    
    print("="*70)
    print("🚗 ENTRAÎNEMENT GAN POUR GÉNÉRATION DE VOITURES")
    print("="*70 + "\n")
    
    # Vérifier si au moins un dossier existe
    if isinstance(DATASET_PATH, str):
        if not os.path.exists(DATASET_PATH):
            print(f"ERREUR: Le dossier {DATASET_PATH} n'existe pas!")
            print("Veuillez modifier DATASET_PATH avec le chemin vers vos images de voitures.")
        else:
            print(f"✓ Dossier trouvé: {DATASET_PATH}\n")
            generator, discriminator = train_generator(DATASET_PATH)
    else:
        # C'est une liste de chemins
        existing_paths = [p for p in DATASET_PATH if os.path.exists(p)]
        if not existing_paths:
            print("ERREUR: Aucun des dossiers spécifiés n'existe!")
            print("Dossiers spécifiés:")
            for path in DATASET_PATH:
                print(f"  - {path}")
        else:
            print(f"✓ {len(existing_paths)} dossier(s) trouvé(s)\n")
            generator, discriminator = train_generator(DATASET_PATH)