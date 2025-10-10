import torch
from train_gan import Generator
from gan_discriminator import Discriminator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import os

class SystemeGANComplet:
    def __init__(self):
        """Initialise le système GAN complet"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger les modèles
        print("📥 Chargement des modèles...")
        self.generator = Generator(latent_dim=100).to(self.device)
        self.generator.load_state_dict(torch.load('models/generator_final.pth', map_location=self.device))
        self.generator.eval()
        
        self.discriminator = Discriminator().to(self.device)
        self.discriminator.load_state_dict(torch.load('models/discriminator_final.pth', map_location=self.device))
        self.discriminator.eval()
        
        print("✅ Modèles chargés avec succès!\n")
        
        # Transformation pour les images
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def generation_progressive_complete(self):
        """Génère une voiture avec TOUTES les étapes intermédiaires visibles en temps réel"""
        print("\n🎬 Génération Progressive Complète")
        print("="*70)
        print("Vous allez voir l'évolution COMPLÈTE du bruit → voiture")
        print("avec le score du discriminateur à chaque étape\n")
        
        # Créer une figure
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5])
        
        ax_image = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])
        ax_info.axis('off')
        
        plt.ion()
        
        # Bruit de départ
        noise = torch.randn(1, 100, 1, 1, device=self.device)
        
        print("🎨 Génération en cours...\n")
        
        # Extraire TOUTES les sorties intermédiaires
        def get_all_outputs(model, input_tensor):
            outputs = []
            layer_names = []
            x = input_tensor
            
            for i, layer in enumerate(model.model):
                x = layer(x)
                # Capturer après chaque couche significative
                if isinstance(layer, (torch.nn.ConvTranspose2d, torch.nn.Conv2d, torch.nn.Tanh, torch.nn.Sigmoid)):
                    outputs.append(x.detach())
                    layer_names.append(f"Couche {i+1}: {layer.__class__.__name__}")
            
            return outputs, layer_names
        
        with torch.no_grad():
            outputs, layer_names = get_all_outputs(self.generator, noise)
        
        print(f"✓ {len(outputs)} étapes capturées\n")
        
        # Afficher TOUTES les étapes progressivement
        for idx, (output, layer_name) in enumerate(zip(outputs, layer_names), 1):
            print(f"  ▸ Étape {idx}/{len(outputs)}: {layer_name}")
            
            # Préparer l'image pour affichage
            img_tensor = output[0]
            
            if img_tensor.shape[0] == 3:
                img = img_tensor.permute(1, 2, 0).cpu().numpy()
                img = (img * 0.5 + 0.5)
            else:
                if img_tensor.shape[0] >= 3:
                    img = img_tensor[:3].permute(1, 2, 0).cpu().numpy()
                else:
                    img = img_tensor[0].cpu().numpy()
                    img = np.stack([img, img, img], axis=-1)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Redimensionner pour le discriminateur si nécessaire
            if output.shape[-1] != 64:
                output_resized = torch.nn.functional.interpolate(
                    output, size=(64, 64), mode='bilinear', align_corners=False
                )
            else:
                output_resized = output
            
            # Score du discriminateur — uniquement pour la dernière couche (3 canaux)
            if output.shape[1] == 3:
                score = self.discriminator(output_resized).item()
            else:
                score = 0.0  # Ne pas évaluer les couches intermédiaires
            
            # Afficher l'image
            ax_image.clear()
            ax_image.imshow(np.clip(img, 0, 1))
            ax_image.set_title(f'Étape {idx}/{len(outputs)}: {layer_name}', 
                               fontsize=14, fontweight='bold', pad=20)
            ax_image.axis('off')
            
            # Afficher les infos
            ax_info.clear()
            ax_info.axis('off')
            
            size_text = f"📐 Taille: {output.shape[-2]}x{output.shape[-1]} pixels\n"
            score_text = f"📊 Score Discriminateur: {score:.4f} "
            
            # Barre de progression
            bar_length = 30
            filled = int(bar_length * score)
            bar = "█" * filled + "░" * (bar_length - filled)
            score_text += f"[{bar}]\n"
            
            # Verdict
            if score > 0.7:
                verdict = "✅ Très probablement une VOITURE"
                color = 'lightgreen'
            elif score > 0.5:
                verdict = "✅ Probablement une voiture"
                color = 'lightyellow'
            elif score > 0.3:
                verdict = "⚠️ Incertain..."
                color = 'lightyellow'
            else:
                verdict = "❌ Pas encore une voiture"
                color = 'lightcoral'
            
            info_full = size_text + score_text + f"\n{verdict}"
            
            ax_info.text(0.5, 0.5, info_full, ha='center', va='center', 
                         fontsize=11, bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))
            
            plt.draw()
            
            # Pause progressive (plus courte au début, plus longue à la fin)
            if idx == len(outputs):
                print(f"     → Score FINAL: {score:.4f} - {verdict}")
                plt.pause(3)
            else:
                print(f"     → Score: {score:.4f}")
                plt.pause(1.5)
        
        print("\n" + "="*70)
        print("✅ Génération terminée!")
        print("="*70)
        
        plt.ioff()
        plt.savefig('generation_complete.png', dpi=150, bbox_inches='tight')
        print("\n📄 Image sauvegardée: generation_complete.png")
        print("\nFermez la fenêtre pour retourner au menu.")
        plt.show()
    
    def classifier_image(self):
        """Teste si une image est une voiture ou pas"""
        print("\n🔍 Classification d'Image")
        print("="*70)
        
        # Demander le chemin de l'image
        image_path = input("\n👉 Entrez le chemin de l'image à tester: ").strip().strip('"')
        
        if not os.path.exists(image_path):
            print(f"❌ ERREUR: Le fichier {image_path} n'existe pas!")
            return
        
        print(f"\n📸 Analyse de: {image_path}")
        
        try:
            # Charger et préparer l'image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            print(f"✓ Image chargée (taille originale: {original_size})")
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Prédiction
            with torch.no_grad():
                score = self.discriminator(image_tensor).item()
            
            # Afficher l'image et le résultat
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Image originale
            ax1.imshow(image)
            ax1.set_title('Image Originale', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Résultat
            ax2.axis('off')
            
            # Verdict
            print("\n" + "="*70)
            print("📊 RÉSULTAT DE L'ANALYSE")
            print("="*70)
            print(f"Score: {score:.4f} (0 = pas voiture, 1 = voiture)")
            
            bar_length = 40
            filled = int(bar_length * score)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"[{bar}] {score*100:.1f}%")
            print("="*70 + "\n")
            
            if score > 0.7:
                verdict = "✅ C'est une VOITURE!"
                color = 'green'
                emoji = "🚗"
            elif score > 0.5:
                verdict = "✅ Probablement une voiture"
                color = 'yellowgreen'
                emoji = "🚙"
            elif score > 0.3:
                verdict = "⚠️ Incertain"
                color = 'orange'
                emoji = "❓"
            else:
                verdict = "❌ Ce n'est PAS une voiture"
                color = 'red'
                emoji = "🚫"
            
            print(f"{emoji} {verdict}")
            print()
            
            # Afficher le résultat graphiquement
            result_text = f"{emoji}\n\n{verdict}\n\nScore: {score:.4f}\n({score*100:.1f}%)"
            ax2.text(0.5, 0.5, result_text, ha='center', va='center',
                     fontsize=16, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=1', facecolor=color, alpha=0.3))
            
            plt.tight_layout()
            plt.savefig('classification_result.png', dpi=150, bbox_inches='tight')
            print("📄 Résultat sauvegardé: classification_result.png")
            plt.show()
            
        except Exception as e:
            print(f"❌ ERREUR lors de l'analyse: {e}")
    
    def menu_couleurs(self):
        """Menu interactif pour changer la couleur d'une voiture générée"""
        print("\n🎨 Changement de Couleur")
        print("="*70)
        print("1. Générer d'abord une nouvelle voiture")
        print("2. Utiliser la dernière voiture générée")
        print("0. Retour au menu principal")
        print("="*70)
        
        choix = input("\n👉 Votre choix: ").strip()
        
        if choix == '0':
            return
        elif choix == '1':
            # Générer une nouvelle voiture
            print("\n🚗 Génération d'une nouvelle voiture...")
            noise = torch.randn(1, 100, 1, 1, device=self.device)
            with torch.no_grad():
                voiture = self.generator(noise)[0].cpu()
        elif choix == '2':
            # Utiliser la dernière générée
            if not os.path.exists('generation_complete.png'):
                print("❌ Aucune voiture générée précédemment. Génération d'une nouvelle...")
                noise = torch.randn(1, 100, 1, 1, device=self.device)
                with torch.no_grad():
                    voiture = self.generator(noise)[0].cpu()
            else:
                # Charger la dernière
                noise = torch.randn(1, 100, 1, 1, device=self.device)
                with torch.no_grad():
                    voiture = self.generator(noise)[0].cpu()
        else:
            print("❌ Choix invalide!")
            return
        
        # Convertir en image
        img_np = (voiture.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        img_np = img_np.astype(np.uint8)
        
        # Menu de couleurs
        while True:
            print("\n" + "="*70)
            print("COULEURS DISPONIBLES")
            print("="*70)
            couleurs = {
                '1': ('Rouge', (0, 1.5, 1.1)),
                '2': ('Bleu', (120, 1.4, 1.0)),
                '3': ('Vert', (60, 1.4, 1.1)),
                '4': ('Jaune', (30, 1.6, 1.15)),
                '5': ('Violet', (150, 1.5, 0.95)),
                '6': ('Orange', (15, 1.6, 1.1)),
                '7': ('Rose', (-30, 1.3, 1.1)),
                '8': ('Cyan', (90, 1.5, 1.05)),
                '9': ('Noir', (0, 0.4, 0.4)),
                '10': ('Blanc', (0, 0.2, 1.6)),
                '11': ('Gris', (0, 0.2, 1.0)),
                '0': ('Quitter', None)
            }
            
            for key, (nom, _) in couleurs.items():
                if key != '0':
                    print(f"  {key}. {nom}")
            print(f"  0. Retour au menu principal")
            print("="*70)
            
            choix_couleur = input("\n👉 Choisissez une couleur: ").strip()
            
            if choix_couleur == '0':
                break
            
            if choix_couleur in couleurs and choix_couleur != '0':
                nom_couleur, params = couleurs[choix_couleur]
                
                # Appliquer le changement de couleur
                print(f"\n🎨 Application de la couleur {nom_couleur}...")
                
                img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
                
                hue_shift, sat_scale, val_scale = params
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * sat_scale, 0, 255)
                img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * val_scale, 0, 255)
                
                img_colored = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                
                # Afficher
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                ax1.imshow(img_np)
                ax1.set_title('Originale', fontsize=12, fontweight='bold')
                ax1.axis('off')
                
                ax2.imshow(img_colored)
                ax2.set_title(f'Couleur: {nom_couleur}', fontsize=12, fontweight='bold')
                ax2.axis('off')
                
                plt.tight_layout()
                filename = f'voiture_{nom_couleur.lower()}.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"✓ Image sauvegardée: {filename}")
                plt.show()
                
                # Mettre à jour l'image pour les prochains changements
                img_np = img_colored
            else:
                print("❌ Choix invalide!")
    
    def menu_principal(self):
        """Menu principal du système"""
        while True:
            print("\n" + "="*70)
            print("🚗 SYSTÈME GAN COMPLET - GÉNÉRATION DE VOITURES")
            print("="*70)
            print("1. 🎬 Génération Progressive Complète (avec toutes les étapes)")
            print("2. 🔍 Classifier une Image (voiture ou pas)")
            print("3. 🎨 Changer la Couleur d'une Voiture")
            print("0. 🚪 Quitter")
            print("="*70)
            
            choix = input("\n👉 Votre choix: ").strip()
            
            if choix == '1':
                self.generation_progressive_complete()
            elif choix == '2':
                self.classifier_image()
            elif choix == '3':
                self.menu_couleurs()
            elif choix == '0':
                print("\n👋 Au revoir!")
                break
            else:
                print("❌ Choix invalide! Veuillez réessayer.")

if __name__ == "__main__":
    try:
        systeme = SystemeGANComplet()
        systeme.menu_principal()
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        print("\nAssurez-vous que:")
        print("  - Les modèles sont dans models/generator_final.pth et models/discriminator_final.pth")
        print("  - Les fichiers train_gan.py et gan_discriminator.py existent")