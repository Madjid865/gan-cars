# 🚗 GAN - Générateur de Voitures avec Changement de Couleur

Ce projet implémente un GAN (Generative Adversarial Network) pour générer des images de voitures avec la possibilité de modifier leurs couleurs de manière interactive.

## 📋 Prérequis

- Python 3.8 ou supérieur
- CUDA (optionnel, pour l'accélération GPU)
- Un dataset d'images de voitures

## 🔧 Installation

1. **Cloner ou créer le projet**
   ```bash
   mkdir gan_voitures
   cd gan_voitures
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Préparer votre dataset**
   - Créez un dossier pour vos images de voitures (ex: `car_dataset/`)
   - Placez toutes vos images de voitures dans ce dossier
   - Formats supportés: `.jpg`, `.jpeg`, `.png`

## 📁 Structure du projet

```
gan_voitures/
│
├── gan_generator.py          # Code du générateur et entraînement
├── gan_discriminator.py      # Code du discriminateur
├── gan_color_menu.py         # Menu interactif pour changer les couleurs
├── requirements.txt          # Dépendances Python
│
├── car_dataset/              # Vos images d'entraînement (à créer)
├── models/                   # Modèles sauvegardés (créé automatiquement)
├── generated_samples/        # Échantillons pendant l'entraînement
└── output/                   # Images générées finales
```

## 🚀 Utilisation

### Étape 1: Entraîner le GAN

1. **Modifiez le chemin du dataset** dans `gan_generator.py`:
   ```python
   DATASET_PATH = "path/to/your/car/dataset"  # Ligne 205
   ```

2. **Lancez l'entraînement**:
   ```bash
   python gan_generator.py
   ```

   L'entraînement va:
   - Entraîner le générateur et le discriminateur simultanément
   - Sauvegarder des checkpoints tous les 10 epochs
   - Créer des échantillons d'images dans `generated_samples/`
   - Durée: variable selon votre GPU/CPU (plusieurs heures recommandées)

### Étape 2: Tester le discriminateur

Pour tester si une image représente une voiture:

```bash
python gan_discriminator.py path/to/image.jpg
```

**Exemple de sortie:**
```
Score de réalité: 0.8523
✓ Très probablement une vraie image de voiture
```

**Évaluer un dossier d'images:**
```python
from gan_discriminator import evaluate_batch
results = evaluate_batch('test_images/')
```

### Étape 3: Générer des voitures avec le menu interactif

```bash
python gan_color_menu.py
```

**Menu disponible:**

```
====================================
MENU PRINCIPAL
====================================
1. Générer une voiture (couleur originale)
2. Générer une voiture avec changement de couleur
3. Générer plusieurs voitures
4. Liste des couleurs disponibles
5. Mode personnalisé (HSV)
0. Quitter
====================================
```

## 🎨 Couleurs disponibles

Le système offre 11 couleurs prédéfinies:

- 🔴 **Rouge**
- 🔵 **Bleu**
- 🟢 **Vert**
- 🟡 **Jaune**
- 🟣 **Violet**
- 🟠 **Orange**
- 🌸 **Rose**
- 🩵 **Cyan**
- ⚫ **Noir**
- ⚪ **Blanc**
- ⬜ **Gris**

## 🎛️ Mode personnalisé (HSV)

Le mode personnalisé vous permet de contrôler finement les couleurs:

- **Teinte (Hue)**: -180 à 180 (décalage de la couleur)
- **Saturation**: 0.0 à 2.0 (intensité de la couleur)
- **Luminosité (Value)**: 0.0 à 2.0 (clarté de l'image)

**Exemples:**
- Voiture rouge vif: `hue=0, saturation=1.5, value=1.0`
- Voiture bleue sombre: `hue=120, saturation=1.3, value=0.7`
- Voiture pastel: `hue=30, saturation=0.5, value=1.2`

## ⚙️ Configuration de l'entraînement

Vous pouvez modifier les hyperparamètres dans `gan_generator.py`:

```python
IMG_SIZE = 64          # Taille des images (64x64)
LATENT_DIM = 100       # Dimension du vecteur de bruit
BATCH_SIZE = 32        # Taille des batches
LEARNING_RATE = 0.0002 # Taux d'apprentissage
EPOCHS = 100           # Nombre d'époques
```

**Recommandations:**
- Plus d'epochs = meilleure qualité (100-200 epochs recommandés)
- Plus de données = meilleurs résultats (minimum 500 images)
- GPU fortement recommandé pour l'entraînement

## 📊 Suivre l'entraînement

Pendant l'entraînement, vous verrez:

```
Epoch [10/100] Batch [50/156] Loss_D: 0.6234 Loss_G: 2.1456
```

- **Loss_D**: Perte du discriminateur (devrait osciller autour de 0.5-0.7)
- **Loss_G**: Perte du générateur (devrait diminuer progressivement)

Les images générées sont sauvegardées dans `generated_samples/` tous les 10 epochs.

## 🐛 Résolution des problèmes

### Le modèle n'existe pas
```
⚠️ ATTENTION: Le modèle models/generator_final.pth n'existe pas!
```
**Solution**: Entraînez d'abord le GAN avec `python gan_generator.py`

### Out of Memory (GPU)
**Solution**: Réduisez `BATCH_SIZE` dans `gan_generator.py` (essayez 16 ou 8)

### Images floues
**Solution**: 
- Augmentez le nombre d'epochs
- Vérifiez la qualité de votre dataset
- Ajustez le learning rate

### Le discriminateur est trop fort
```
Loss_D: 0.01 Loss_G: 8.5
```
**Solution**: Le discriminateur domine. Essayez:
- Réduire le learning rate du discriminateur
- Augmenter le learning rate du générateur

## 📝 Utilisation avancée

### Générer une seule image en Python

```python
import torch
from gan_generator import Generator
from gan_color_menu import generate_and_save

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
generator.load_state_dict(torch.load('models/generator_final.pth'))

# Générer une voiture rouge
generate_and_save(generator, device, num_images=1, color='rouge')
```

### Batch de voitures avec différentes couleurs

```python
colors = ['rouge', 'bleu', 'vert', 'jaune']
for color in colors:
    generate_and_save(generator, device, num_images=5, color=color)
```

### Modifier une image existante

```python
from gan_color_menu import ColorChanger
import torch
from PIL import Image
from torchvision import transforms

# Charger une image
img = Image.open('car.jpg')
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
img_tensor = transform(img)

# Changer en bleu
img_blue = ColorChanger.apply_color_filter(img_tensor, 'bleu')
Image.fromarray(img_blue.astype('uint8')).save('car_blue.jpg')
```

## 🎓 Architecture du GAN

### Générateur
- Input: Vecteur aléatoire de dimension 100
- Architecture: 5 couches ConvTranspose2d
- Output: Image RGB 64x64
- Activation finale: Tanh

### Discriminateur
- Input: Image RGB 64x64
- Architecture: 5 couches Conv2d
- Output: Probabilité (0-1)
- Activation finale: Sigmoid

## 📚 Ressources

- [Documentation PyTorch](https://pytorch.org/docs/)
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## 🤝 Contribution

N'hésitez pas à améliorer ce projet:
- Ajouter de nouvelles couleurs prédéfinies
- Améliorer l'architecture du GAN
- Augmenter la résolution des images
- Ajouter des filtres supplémentaires

## 📄 Licence

Projet éducatif - Libre d'utilisation

## ✨ Astuces

1. **Qualité des données**: Utilisez des images de voitures de bonne qualité et bien cadrées
2. **Patience**: L'entraînement prend du temps, laissez tourner plusieurs heures
3. **Sauvegarde**: Les checkpoints permettent de reprendre l'entraînement si interrompu
4. **Expérimentation**: Testez différentes couleurs et paramètres HSV pour des résultats créatifs!

---

Bon entraînement! 🚗💨