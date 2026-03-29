# Projet Deep Learning — Détection de Pneumonie (Chest X-Ray)

## Dataset
[Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Structure du projet
```
Projet_Deep_Learning/
├── src/
│   └── dataset.py
├── notebooks/
│   └── 02_preprocessing.ipynb
└── README.md
```

## Prétraitement & Data Augmentation

### Description
Pipeline de prétraitement et data augmentation pour la détection de pneumonie par CNN.

### Fichiers
- `src/dataset.py` : pipeline complet (transformations, datasets, dataloaders)
- `notebooks/02_preprocessing.ipynb` : notebook avec visualisations

### Prétraitements appliqués
- Conversion en niveaux de gris
- Resize 224×224
- Normalisation (mean=0.485, std=0.229)

### Data Augmentation (train uniquement)
- Rotation ±10°
- Translation ±5%
- Zoom 95–105%
- Variation de luminosité ±20%

### Technologies
- Python 3.10
- PyTorch
- Torchvision
- Matplotlib

### Utilisation
```python
from src.dataset import get_dataloaders

loaders = get_dataloaders(data_root="data/", batch_size=32)
# loaders['train'], loaders['val'], loaders['test']
```
