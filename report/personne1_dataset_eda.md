# Personne 1 — Dataset & Exploration des données (EDA)

## 1) Dataset
- **Source** : Chest X-Ray Images (Pneumonia) — Kaggle.
- **Type de tâche** : classification binaire (`NORMAL` vs `PNEUMONIA`).
- **Structure vérifiée** : `train/`, `val/`, `test/` avec sous-dossiers `NORMAL/` et `PNEUMONIA/`.
- **Racine utilisée pour l'analyse** : `chest_xray/chest_xray/`.

## 2) Exploration des données
Résultats obtenus après exécution de `notebooks/01_eda.ipynb` :

### Répartition par split et par classe
- **Train** : NORMAL = 1342, PNEUMONIA = 3876
- **Validation** : NORMAL = 9, PNEUMONIA = 9
- **Test** : NORMAL = 234, PNEUMONIA = 390

### Totaux par classe
- **NORMAL** : 1585
- **PNEUMONIA** : 4275
- **Ratio de déséquilibre** (classe majoritaire / minoritaire) : **2.70**

### Dimensions d'images (sur 5856 images lisibles)
- **Largeur** : moyenne = 1327.88, min = 384, max = 2916
- **Hauteur** : moyenne = 970.69, min = 127, max = 2713

### Formats
- **`.jpeg`** : 5856 fichiers
- **Sans extension (`<no_ext>`)** : 4 fichiers

### Figures à insérer
- `outputs/figures/eda_class_counts.png`
- `outputs/figures/eda_samples_normal_vs_pneumonia.png`
- `outputs/figures/eda_dimensions_hist.png`

## 3) Qualité et limites du dataset
Observations principales :
- **Images corrompues/non lisibles** : 4 fichiers détectés (impossible à ouvrir avec PIL).
- **Déséquilibre de classes** : la classe `PNEUMONIA` est ~2.7 fois plus représentée que `NORMAL`.
- **Hétérogénéité des résolutions** : forte variabilité des dimensions, ce qui justifie un `resize` systématique au prétraitement.
- **Split de validation très petit** : seulement 18 fichiers au total, ce qui peut rendre l'estimation des performances instable.

## 4) Conclusion courte (pour transition vers Personne 2)
Le dataset est exploitable pour une classification binaire, mais nécessite un prétraitement rigoureux (resize/normalisation) et une attention au déséquilibre de classes. Les 4 fichiers non lisibles doivent être ignorés/filtrés dans le pipeline de la Personne 2 pour éviter des erreurs d'entraînement.
