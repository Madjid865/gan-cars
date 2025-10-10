# 🚗 GAN Cars Project

This project aims to generate realistic images of cars using a Generative Adversarial Network (GAN).  
The goal is to train a model capable of generating synthetic car images similar to real ones.

---

## 📁 Project Structure

```
gan-cars-project/
├── data/                # Raw and processed image datasets
│   ├── raw/
│   └── processed/
├── models/              # Generator and Discriminator model definitions
│   ├── generator.py
│   └── discriminator.py
├── results/             # Generated images, checkpoints, loss plots
│   ├── generated_images/
│   ├── checkpoints/
│   └── loss_curves/
├── training/            # Training scripts
│   ├── train_discriminator.py
│   └── train_gan.py
├── utils/               # Helper functions (image transforms, dataloaders)
│   ├── dataset_loader.py
│   └── image_utils.py
├── README.md
└── requirements.txt
```

---

## 👥 Team

| Name      | Group | Git Branch |
|-----------|--------|-------------|
| Madjid    | B      | `madjid`    |
| Nassim    | C      | `nassim`    |
| Hazem     | C      | `hazem`     |
| Kim       | A      | `kim`       |

---

## 🔧 Git Workflow

### 1. Clone the repository (once):

```bash
git clone https://github.com/Madjid865/gan-cars.git
cd gan-cars
```

### 2. Switch to your personal branch:

```bash
git checkout your-name
```

### 3. After each work session:

```bash
git add .
git commit -m "Your message here"
git push origin your-name
```

---

## 🔀 Merging Strategy

- ✅ All work is done on **personal branches**
- 🔁 Merges to `main` happen **only after testing**
- ❌ Never commit directly to `main`
- 🔒 Only one person merges at a time to avoid conflicts

---

## 📝 Conventions

- Code and comments in **English**
- Use consistent naming (`snake_case`)
- Use **VS Code** (recommended)
- Follow **PEP8** (indentation, spacing, etc.)
- Save generated images in `results/generated_images/`

---

## 🤝 Collaboration Tips

- Communicate before modifying shared files
- Commit small and clear changes
- Don’t push broken or untested code
- Regularly pull from `main` and sync your branch

---
