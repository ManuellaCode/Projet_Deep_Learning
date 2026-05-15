# Projet_Deep_Learning
# Classification NORMAL vs PNEUMONIA (radiographies thoraciques)

**Objectif :** Entraîner un petit CNN pour distinguer des radiographies normales de celles avec pneumonie, puis évaluer le modèle et visualiser les résultats.

**Dataset :** Chest X-Ray Images (Pneumonia) sur Kaggle.  
Structure attendue des dossiers :
- `chest_xray/train/NORMAL/` et `chest_xray/train/PNEUMONIA/`
- `chest_xray/val/` et `chest_xray/test/` avec les mêmes sous-dossiers

Adapter la variable `data_dir` plus bas si les données sont ailleurs.
### Pour Google Colab

1. **Transférer le notebook** : Fichier → Téléverser le notebook → choisis `chest_xray_cnn.ipynb`.
2. **Données** : soit tu téléverses le dossier `chest_xray` (train/val/test) dans le panneau Fichiers (📁) de Colab, soit tu le mets sur Google Drive et tu exécutes la cellule « Montage Google Drive » ci-dessous, puis tu adaptes `data_dir` vers ton chemin (ex. `Path("/content/drive/MyDrive/chest_xray")`).
3. **GPU** : Exécution → Changer le type d’exécution → Accélérateur matériel : GPU (T4).
4. Exécute les cellules dans l’ordre. Sur Colab, `num_workers` est déjà réglé à 0 pour éviter les erreurs.
