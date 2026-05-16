# Projet_Deep_Learning

## Classification d'images medicales

Cas d'etude: detection de pneumonie sur radiographies thoraciques (classification binaire NORMAL vs PNEUMONIA).

Ce projet applique un pipeline complet de Deep Learning pour analyser des radiographies thoraciques avec un modele CNN, en visant un usage d'aide a la decision (et non un remplacement de l'expertise medicale).

## 1. Contexte et problematique

L'imagerie medicale genere un volume important de donnees (radiographie, IRM, scanner, etc.). Les CNN (Convolutional Neural Networks) sont bien adaptes a l'analyse automatique d'images.

Question principale:

Peut-on entrainer un modele CNN capable de distinguer des radiographies thoraciques normales de celles presentant une pneumonie, avec des performances robustes et une interpretation visuelle minimale?

Contraintes reelles:

- Qualite variable des images.
- Desequilibre possible entre classes.
- Risque de sur-apprentissage (overfitting).
- Exigence d'explicabilite pour renforcer la confiance clinique.

## 2. Objectifs du projet

1. Construire un pipeline reproductible de traitement d'images medicales.
2. Concevoir et entrainer un CNN pour la classification binaire.
3. Evaluer le modele avec des metriques adaptees (Accuracy, Precision, Recall, F1, AUC).
4. Visualiser les erreurs via une matrice de confusion.
5. Ajouter une interpretation visuelle (Grad-CAM) en extension.

Livrables attendus:

- Notebook(s) Python reproductible(s).
- Modele entraine.
- Rapport ou slides de presentation.
- Discussion des limites et pistes d'amelioration.

## 3. Dataset retenu (Kaggle)

Dataset: Chest X-Ray Images (Pneumonia)

- Type: radiographies thoraciques.
- Tache: classification binaire (NORMAL, PNEUMONIA).
- Interet pedagogique: cas reel de vision medicale, pipeline CNN accessible, comparaison baseline vs transfer learning.

Points de vigilance methodologiques:

- Verifier la structure des dossiers.
- Verifier la repartition train/validation/test.
- Controler l'equilibre des classes.
- Verifier la qualite des labels.
- Identifier les risques de fuite de donnees (data leakage).

## 4. Structure des donnees

Structure attendue:

```text
data/
   chest_xray/
      train/
         NORMAL/
         PNEUMONIA/
      val/
         NORMAL/
         PNEUMONIA/
      test/
         NORMAL/
         PNEUMONIA/
```

Verifications conseillees avant entrainement:

- Nombre d'images par classe et par split.
- Dimensions d'images et formats (JPEG/PNG).
- Presence d'images corrompues.

## 5. Pipeline global

1. Telechargement du dataset Kaggle.
2. Inspection des donnees.
3. Pretraitement et augmentation.
4. Entrainement d'un CNN baseline.
5. Evaluation et analyse des erreurs.
6. Interpretabilite (Grad-CAM).
7. Extension possible avec transfer learning.

## 6. Pretraitement et augmentation

Pretraitement recommande:

- Chargement en tenseurs (PyTorch).
- Redimensionnement (exemple: 224 x 224).
- Normalisation des intensites.
- Conversion eventuelle en 3 canaux (compatibilite modeles pre-entraines ImageNet).

Augmentation (uniquement sur train):

- Rotation legere (environ +/- 5 deg a +/- 10 deg).
- Translation faible.
- Zoom leger.
- Variation moderee contraste/luminosite.

Important: eviter les transformations qui alterent le sens medical (rotations extremes, inversions non justifiees, etc.).

## 7. Architecture CNN baseline

Architecture proposee:

- Bloc 1: Conv(32) + ReLU + MaxPool
- Bloc 2: Conv(64) + ReLU + MaxPool
- Bloc 3: Conv(128) + ReLU + MaxPool
- Flatten
- Dense(128) + Dropout
- Dense(1) + Sigmoid (ou logit + BCEWithLogitsLoss)

Pourquoi commencer simple:

- Poser une baseline robuste.
- Diagnostiquer les problemes de donnees.
- Comprendre l'impact de chaque choix (taille image, augmentation, seuil).

## 8. Fonction de cout et optimisation

Pour une classification binaire:

$$
L_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\Big(y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\Big)
$$

Choix pratiques:

- Optimiseur: Adam (lr initiale 1e-3), ou AdamW.
- Regularisation: Dropout, Early Stopping, L2/weight decay.
- Eventuellement class weights en cas de desequilibre.

## 9. Protocole experimental (suggestion)

- Taille image: 224 x 224.
- Batch size: 16 ou 32.
- Nombre d'epoques: 20 a 40 avec early stopping.
- Optimiseur: Adam.
- Learning rate initiale: 1e-3.
- Loss: Binary Cross-Entropy.

Suivi a chaque epoque:

- Train loss / Val loss.
- Train accuracy / Val accuracy.
- Eventuellement AUC validation.

## 10. Gestion des risques experimentaux

- Overfitting: ecart train/val important.
- Data leakage: verifier la separation des patients selon les splits (si information disponible).
- Desequilibre: l'accuracy seule peut etre trompeuse.
- Seuil de decision: 0.5 n'est pas toujours optimal.

En contexte medical, la sensibilite (Recall) peut etre prioritaire selon le besoin clinique.

## 11. Metriques d'evaluation

A partir de la matrice de confusion (TP, TN, FP, FN):

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
Specificite = \frac{TN}{TN + FP}
$$

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

Complements utiles:

- Courbe ROC et AUC.
- PR curve en cas de classes desequilibrees.

## 12. Interpretabilite (Grad-CAM)

Objectif:

- Comprendre quelles regions de l'image influencent la prediction.
- Verifier que le modele regarde des zones pulmonaires pertinentes.

Important: une heatmap explicative n'est pas une preuve clinique. Elle aide surtout a auditer le comportement du modele.

## 13. Planification type (6 semaines)

- S1: Choix dataset, telechargement, exploration, nettoyage, statistiques descriptives.
- S2: Pretraitement, visualisation, pipeline de chargement.
- S3: Implementation CNN baseline, premier entrainement.
- S4: Ajustement hyperparametres, regularisation, augmentation.
- S5: Evaluation complete (metriques, matrice, courbes) + Grad-CAM.
- S6: Rapport final, slides, demonstration.

## 14. Livrables du groupe

1. Code source (notebooks et scripts).
2. Modele entraine (.pt ou .h5).
3. Rapport technique (probleme, methode, resultats, limites).
4. Presentation finale.
5. Annexe de reproductibilite (versions librairies, seed, protocole d'evaluation).

## 15. Extensions possibles

- Transfer learning (ResNet18, DenseNet121, EfficientNet).
- Comparaison baseline CNN vs modele pre-entraine.
- Optimisation du seuil selon la sensibilite visee.
- Explicabilite avancee (Grad-CAM++, Integrated Gradients).
- Validation plus rigoureuse (K-fold si applicable).
- Prototype de deploiement local (interface Streamlit).

## 16. Contenu actuel du depot

Arborescence principale:

```text
README.md
data/
   chest_xray/
notebooks/
   chest-x-ray-images-pneumonia.ipynb
   chest_xray_cnn.ipynb
scripts/
   download_kaggle_dataset.py
```

Notebook principal ajoute pour le groupe:

- notebooks/chest_xray_cnn.ipynb

## 17. Environnement Python (recommande)

- Python >= 3.10
- torch, torchvision
- numpy, pandas
- matplotlib, scikit-learn
- pillow
- opencv-python (optionnel)

Exemple d'installation:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow
```

## 18. Reproductibilite

Bonnes pratiques a appliquer dans les notebooks/scripts:

- Fixer la seed aleatoire.
- Logger les hyperparametres et versions de librairies.
- Conserver les courbes d'entrainement et la matrice de confusion.
- Sauvegarder les poids du meilleur modele selon validation.

## 19. Limites et conclusion

Ce projet permet une montee en competence rapide sur un pipeline complet de Deep Learning applique a un cas medical realiste.

La qualite finale depend autant:

- de la rigueur methodologique (split, metriques, risques experimentaux),
- que du choix du modele.

L'ajout d'explicabilite et d'une discussion critique des limites est indispensable pour un usage responsable en contexte medical.

## 20. References

- Kaggle: Chest X-Ray Images (Pneumonia), dataset public.


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
