"""
dataset.py — Personne 2 : Prétraitement & Data Augmentation
Projet : Détection de Pneumonie par CNN (Chest X-Ray)
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# 1. CONSTANTES & CHEMINS
# ─────────────────────────────────────────────

IMG_SIZE    = 224          # Résolution cible (224×224)
MEAN        = [0.485]      # Moyenne ImageNet (adapté grayscale)
STD         = [0.229]      # Écart-type ImageNet
BATCH_SIZE  = 32
NUM_WORKERS = 2
DATA_ROOT   = Path("data")  # Adapter si besoin


# ─────────────────────────────────────────────
# 2. TRANSFORMATIONS
# ─────────────────────────────────────────────

# --- Train : preprocessing + augmentation ---
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # radiographies en niveaux de gris
    transforms.Resize((IMG_SIZE, IMG_SIZE)),        # redimensionnement 224×224
    transforms.RandomRotation(degrees=10),          # rotation légère ±10°
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),                     # translation ±5%
        scale=(0.95, 1.05),                         # zoom léger
    ),
    transforms.ColorJitter(brightness=0.2),         # variation de luminosité ±20%
    transforms.ToTensor(),                          # conversion PIL → Tensor [0,1]
    transforms.Normalize(mean=MEAN, std=STD),       # normalisation
])

# --- Val / Test : preprocessing uniquement (pas d'augmentation) ---
eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ─────────────────────────────────────────────
# 3. CHARGEMENT DES DATASETS
# ─────────────────────────────────────────────

def get_datasets(data_root: Path = DATA_ROOT):
    """
    Retourne les trois ImageFolders (train, val, test)
    en appliquant les transformations appropriées.

    Structure attendue :
        data/
          train/  NORMAL/  PNEUMONIA/
          val/    NORMAL/  PNEUMONIA/
          test/   NORMAL/  PNEUMONIA/
    """
    train_dir = data_root / "train"
    val_dir   = data_root / "val"
    test_dir  = data_root / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(
                f"Dossier introuvable : {d}\n"
                "Vérifiez la structure du dataset (cf. Personne 1 — 01_eda.ipynb)."
            )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(str(test_dir),  transform=eval_transforms)

    return train_dataset, val_dataset, test_dataset


# ─────────────────────────────────────────────
# 4. DATALOADERS
# ─────────────────────────────────────────────

def get_dataloaders(
    data_root:   Path = DATA_ROOT,
    batch_size:  int  = BATCH_SIZE,
    num_workers: int  = NUM_WORKERS,
):
    """
    Retourne les DataLoaders prêts pour l'entraînement.

    Args:
        data_root   : racine du dataset
        batch_size  : taille de batch (défaut 32)
        num_workers : workers parallèles pour le chargement
    Returns:
        dict avec clés 'train', 'val', 'test'
    """
    train_ds, val_ds, test_ds = get_datasets(data_root)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    print(f"[dataset.py] Classes détectées : {train_ds.classes}")
    print(f"  Train  : {len(train_ds):>5} images")
    print(f"  Val    : {len(val_ds):>5} images")
    print(f"  Test   : {len(test_ds):>5} images")

    return loaders


# ─────────────────────────────────────────────
# 5. UTILITAIRE — DÉNORMALISATION POUR AFFICHAGE
# ─────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Dénormalise un tensor image (C, H, W) et retourne
    un array numpy (H, W) prêt pour plt.imshow.
    """
    mean = torch.tensor(MEAN).view(-1, 1, 1)
    std  = torch.tensor(STD).view(-1, 1, 1)
    img  = tensor * std + mean
    img  = img.squeeze(0).clamp(0, 1).numpy()
    return img


# ─────────────────────────────────────────────
# 6. VISUALISATION — BATCH AUGMENTÉ
# ─────────────────────────────────────────────

def show_augmented_batch(data_root: Path = DATA_ROOT, n: int = 12):
    """
    Affiche n images augmentées issues du loader d'entraînement.
    Permet de vérifier visuellement que l'augmentation est correcte.
    Sauvegarde la figure dans outputs/figures/.
    """
    os.makedirs("outputs/figures", exist_ok=True)
    loaders     = get_dataloaders(data_root, batch_size=n)
    images, labels = next(iter(loaders["train"]))
    class_names = loaders["train"].dataset.classes

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Batch augmenté — Train set", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < n:
            img_np = denormalize(images[i])
            ax.imshow(img_np, cmap="gray")
            ax.set_title(class_names[labels[i].item()], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = "outputs/figures/augmented_batch.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Figure sauvegardée → {out_path}")


# ─────────────────────────────────────────────
# 7. POINT D'ENTRÉE (test rapide)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    loaders = get_dataloaders()
    imgs, lbls = next(iter(loaders["train"]))
    print(f"\nShape d'un batch : {imgs.shape}")   # (32, 1, 224, 224)
    print(f"Labels (8 premiers) : {lbls[:8].tolist()}")
    show_augmented_batch()
