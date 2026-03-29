# Projet_Deep_Learning

## Branche de la partie Dataset (Personne 1)
Les livrables de la partie Dataset/EDA sont sur la branche `Partie_Dataset`.

## Pourquoi le dataset n'est pas dans Git
Le dataset Chest X-Ray est volumineux. Il n'est pas versionné dans Git pour éviter un dépôt trop lourd.

## Télécharger le dataset Kaggle (recommandé)
1. Installer Kaggle:
   - `pip install kaggle`
2. Créer les credentials API Kaggle:
   - Télécharger `kaggle.json` depuis ton compte Kaggle
   - Placer le fichier dans `~/.kaggle/kaggle.json`
3. Lancer le script:
   - `python scripts/download_kaggle_dataset.py --output data`

Après exécution, la structure attendue est en général:
- `data/chest_xray/train`
- `data/chest_xray/val`
- `data/chest_xray/test`

## Option Git LFS (uniquement si vous décidez de versionner des données)
Par défaut, ne versionnez pas les images du dataset.
Si l'équipe choisit quand même de stocker des fichiers lourds:
1. Installer Git LFS
2. Initialiser: `git lfs install`
3. Suivre les gros fichiers: `git lfs track "*.jpeg"`
4. Committer `.gitattributes`

## Livrables Personne 1
- `notebooks/01_eda.ipynb`
- `report/personne1_dataset_eda.md`

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
