# 🚗 GAN Cars Project

Ce projet utilise un **GAN (Generative Adversarial Network)** de type DCGAN pour générer des images réalistes de voitures à partir de bruit aléatoire. Le projet inclut deux versions : 64×64 (utilisée pour le rapport avec des courbes de loss plus parlantes) et 128×128 (pour la génération d'images de meilleure qualité).

---

## 📁 Structure du Projet

```
gan-cars/
├── 128x128/                          # Version 128×128 avec interface web
│   ├── checkpoints/                  # Poids du modèle 128×128
│   ├── plots_128_resume/             # Graphiques des courbes de loss
│   ├── samples_128_resume/           # Échantillons générés pendant l'entraînement
│   ├── __pycache__/                  # Cache Python
│   ├── advanced_features.py          # Fonctionnalités avancées (variations, interpolations, GIF)
│   ├── app_enhanced.py               # Interface web Gradio complète
│   ├── fix_all_img_size.py           # Script de correction des paramètres
│   ├── gan_discriminator.py          # Discriminateur 128×128
│   ├── inference.py                  # Génération d'images avec le modèle entraîné
│   ├── launch_enhanced.bat           # Script de lancement Windows
│   ├── README.md                     # Documentation 128×128
│   ├── requirements_interface.txt    # Dépendances de l'interface
│   ├── system.py                     # Utilitaires (génération, scoring)
│   └── train_gan.py                  # Entraînement 128×128 (avec features de stabilité)
│
├── checkpoints/                      # Poids du modèle 64×64
├── generated_samples/                # Images générées 64×64
├── plots/                            # Graphiques 64×64
├── .gitignore                        # Fichiers à ignorer par Git
├── gan_discriminator.py              # Discriminateur 64×64
├── README.md                         # Ce fichier
├── requirements.txt                  # Dépendances du projet complet
├── system.py                         # Utilitaires 64×64
└── train_gan.py                      # Entraînement 64×64
```

---

## 📝 Description des Fichiers

### Racine du Projet (Version 64×64)

- **`train_gan.py`** : Script principal d'entraînement pour la version 64×64. Gère l'entraînement, la validation, les checkpoints et la génération d'échantillons.

- **`gan_discriminator.py`** : Discriminateur DCGAN avec Spectral Normalization pour stabiliser l'entraînement. Sortie en logits (sans Sigmoid).

- **`system.py`** : Utilitaires en ligne de commande pour générer des images et scorer des images avec le discriminateur.

- **`.gitignore`** : Liste des fichiers/dossiers à exclure du versioning Git.

### Dossier 128x128/ (Version Haute Résolution + Interface)

- **`train_gan.py`** : Version avancée de l'entraînement avec features de stabilité :
  - Gradient clipping
  - R1 regularization (lazy)
  - EMA (Exponential Moving Average) pour le générateur
  - Instance noise
  - Label smoothing
  - Support AMP (Automatic Mixed Precision)

- **`gan_discriminator.py`** : Discriminateur adapté pour 64×64 et 128×128 avec architecture plus profonde pour 128×128.

- **`system.py`** : Utilitaires compatibles avec les deux résolutions.

- **`inference.py`** : Module de génération d'images utilisant un modèle pré-entraîné. Inclut :
  - Génération d'images uniques ou en batch
  - Grilles d'images
  - Interpolation entre deux seeds

- **`advanced_features.py`** : Fonctionnalités avancées pour démonstrations :
  - `generate_variations()` : Variations autour d'un seed
  - `generate_random_walk()` : Marche aléatoire dans l'espace latent
  - `generate_latent_grid()` : Exploration systématique de l'espace latent
  - `generate_mega_showcase()` : Grille massive de voitures
  - `create_gif_from_images()` : Export en GIF animé

- **`app_enhanced.py`** : Interface web Gradio professionnelle avec 5 onglets :
  - 🎨 Single Car : Génération d'une voiture à la fois
  - 🎯 Batch Generation : Grilles de plusieurs voitures
  - 🔄 Interpolation : Transitions fluides + export GIF
  - 🎲 Variations : Exploration autour d'un design
  - 🎆 Mega Showcase : Génération massive (64-100 voitures)

- **`launch_enhanced.bat`** : Script de lancement rapide pour Windows.

- **`fix_all_img_size.py`** : Script de correction pour retirer les références obsolètes à `img_size`.

- **`requirements_interface.txt`** : Dépendances spécifiques à l'interface web.

### Dossiers de Données

- **`checkpoints/`** : Sauvegarde des poids du modèle (Generator et Discriminator) à chaque epoch.

- **`generated_samples/` ou `samples_128_resume/`** : Échantillons visuels générés pendant l'entraînement pour suivre la progression.

- **`plots/` ou `plots_128_resume/`** : Graphiques des courbes de loss (Discriminator et Generator) et historique JSON.

---

## 🛠️ Installation

### Prérequis

- Python 3.8+
- CUDA (recommandé pour l'entraînement GPU)

### Installation des Dépendances

```bash
# À la racine du projet
pip install -r requirements.txt

# Pour l'interface web seulement (dossier 128x128)
cd 128x128
pip install -r requirements_interface.txt
```

---

## 🚀 Utilisation

### 1. Entraînement (Version 64×64)

```bash
python train_gan.py \
  --data_root data/cars/train \
  --epochs 50 \
  --batch_size 32 \
  --img_size 64 \
  --lr_g 2e-4 \
  --lr_d 2e-4
```

### 2. Entraînement (Version 128×128 avec stabilité)

```bash
cd 128x128

python train_gan.py \
  --data_root "path/to/dataset" \
  --img_size 128 \
  --batch_size 32 \
  --epochs 120 \
  --lr_g 1e-4 \
  --lr_d 5e-5 \
  --grad_clip 1.0 \
  --r1_gamma 2.0 \
  --r1_every 16 \
  --ema --ema_decay 0.999
```

**Options de stabilité importantes :**
- `--grad_clip` : Clip les gradients (recommandé : 1.0)
- `--r1_gamma` : Régularisation R1 (recommandé : 2.0-10.0)
- `--ema` : Active l'EMA pour des échantillons plus stables
- `--label_real 0.9 --label_fake 0.0` : Label smoothing
- `--resume auto` : Reprend automatiquement le dernier checkpoint
- `--reset_optim` : Réinitialise les optimiseurs (utile en cas d'instabilité)

### 3. Génération d'Images (CLI)

```bash
# Générer une grille de 16 images
python system.py generate --gen checkpoints/generator_final.pth --n 16 --seed 42

# Scorer une image avec le discriminateur
python system.py score --disc checkpoints/discriminator_final.pth --image test.jpg
```

### 4. Interface Web (128×128)

```bash
cd 128x128

# Windows
launch_enhanced.bat

# Linux/Mac
python app_enhanced.py
```

L'interface sera accessible sur `http://localhost:7863`

**Fonctionnalités de l'interface :**
- ✨ Génération d'images uniques ou en batch
- 🔄 Interpolation fluide entre deux designs
- 🎬 Export en GIF animé
- 🎲 Exploration de variations
- 🎆 Showcase massif (64-100 voitures)

---

## 👥 Équipe

| Nom     | Groupe | Branche Git |
|---------|--------|-------------|
| Madjid  | B      | `madjid`    |
| Nassim  | B      | `nassim`    |
| Hazem   | C      | `hazem`     |
| Kim     | A      | `kim`       |

---

## 🔀 Workflow Git

### Branches Personnelles
```bash
# Cloner le repo
git clone https://github.com/Madjid865/gan-cars.git
cd gan-cars

# Basculer sur votre branche
git checkout votre-nom

# Après chaque session de travail
git add .
git commit -m "Description claire des changements"
git push origin votre-nom
```

### Règles de Collaboration
- ✅ Tout le travail se fait sur les **branches personnelles**
- 🔁 Les merges vers `main` se font **après tests**
- ❌ Ne jamais commit directement sur `main`
- 🔒 Une seule personne merge à la fois

---

## 📊 Architecture Technique

### Générateur (Generator)
- Architecture DCGAN avec ConvTranspose2d
- Input : Vecteur latent (100D) de bruit gaussien
- Output : Image RGB normalisée en [-1, 1]
- BatchNorm + ReLU dans les couches intermédiaires
- Tanh en sortie

### Discriminateur (Discriminator)
- Architecture convolutionnelle DCGAN
- Spectral Normalization pour la stabilité
- Sortie : Logits (pas de Sigmoid)
- LeakyReLU (0.2) + BatchNorm

### Entraînement
- Loss : BCEWithLogitsLoss (stable numériquement)
- Optimiseur : Adam (β₁=0.0, β₂=0.9)
- Techniques de stabilisation :
  - Label smoothing (real=0.9, fake=0.0)
  - Gradient clipping
  - R1 regularization (lazy)
  - EMA sur le générateur

---

## 📝 Conventions de Code

- Code et commentaires en **anglais**
- Naming : `snake_case` pour variables/fonctions
- Indentation : 4 espaces (PEP8)
- Éditeur recommandé : **VS Code**
- Commits clairs et atomiques

---

## 💡 Tips

- **Seeds** : Même seed = même voiture générée
- **Version 64×64** : Utilisée pour le rapport (courbes de loss)
- **Version 128×128** : Meilleure qualité visuelle + interface
- Les checkpoints sont sauvegardés automatiquement
- Les courbes de loss GAN oscillent naturellement

---

## 📚 Références

- Dataset : CompCars
- Architecture : DCGAN (Deep Convolutional GAN)
- Framework : PyTorch
- Interface : Gradio

---