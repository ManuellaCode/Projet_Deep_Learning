import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# Configuration par défaut
# ─────────────────────────────────────────────
SEED        = 42
IMG_SIZE    = 224
BATCH_SIZE  = 32
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# Architecture CNN (identique à src/model.py — Personne 3)
# ─────────────────────────────────────────────
class CNNBaseline(nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────
def get_test_loader(data_dir):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Test set : {len(dataset)} images | Classes : {dataset.classes}")
    return loader

# ─────────────────────────────────────────────
# Inférence
# ─────────────────────────────────────────────
def run_inference(model, loader):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            probs = model(imgs).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────
# Calcul des métriques
# ─────────────────────────────────────────────
def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'threshold'   : threshold,
        'accuracy'    : round(accuracy_score(labels, preds), 4),
        'precision'   : round(precision_score(labels, preds, zero_division=0), 4),
        'recall'      : round(recall_score(labels, preds, zero_division=0), 4),
        'specificity' : round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        'f1_score'    : round(f1_score(labels, preds, zero_division=0), 4),
        'auc_roc'     : round(roc_auc_score(labels, probs), 4),
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        'false_negatives_count': int(fn),
        'false_positives_count': int(fp),
        'total_test_images'    : int(len(labels))
    }
    return metrics, preds


def print_metrics(metrics):
    print('\n' + '=' * 52)
    print('       MÉTRIQUES D\'ÉVALUATION — SET DE TEST')
    print('=' * 52)
    print(f"  Seuil utilisé : {metrics['threshold']}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}   ← priorité clinique")
    print(f"  Spécificité   : {metrics['specificity']:.4f}")
    print(f"  F1-Score      : {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC       : {metrics['auc_roc']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"\n  Matrice de confusion :")
    print(f"    TN={cm['TN']}  FP={cm['FP']}")
    print(f"    FN={cm['FN']}  TP={cm['TP']}  ← FN = pneumonie manquée ⚠️")
    print('=' * 52)


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, fig_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Prédiction', fontweight='bold')
    ax.set_ylabel('Réalité', fontweight='bold')
    ax.set_title('Matrice de Confusion', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Sauvegardée : {path}')


def plot_roc_curve(labels, probs, fig_dir):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_t   = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='royalblue', lw=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'gray', linestyle='--', lw=1.5, label='Aléatoire')
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5,
               label=f'Seuil optimal = {optimal_t:.3f}')
    ax.set_xlabel('FPR (1 - Spécificité)')
    ax.set_ylabel('TPR (Recall / Sensibilité)')
    ax.set_title('Courbe ROC', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, 'roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Sauvegardée : {path}')
    print(f'   Seuil optimal ROC : {optimal_t:.4f}')
    return optimal_t


def plot_probability_distribution(labels, probs, threshold, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(probs[labels == 0], bins=40, alpha=0.7, color='steelblue', label='NORMAL')
    axes[0].hist(probs[labels == 1], bins=40, alpha=0.7, color='tomato',    label='PNEUMONIA')
    axes[0].axvline(x=threshold, color='black', linestyle='--', lw=2, label=f'Seuil={threshold}')
    axes[0].set_xlabel('P(PNEUMONIA)')
    axes[0].set_ylabel("Nombre d'images")
    axes[0].set_title('Distribution des probabilités', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot([probs[labels == 0], probs[labels == 1]],
                    labels=CLASS_NAMES, patch_artist=True,
                    boxprops=dict(facecolor='steelblue'))
    axes[1].axhline(y=threshold, color='black', linestyle='--', lw=2, label=f'Seuil={threshold}')
    axes[1].set_ylabel('P(PNEUMONIA)')
    axes[1].set_title('Boxplot des probabilités', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('Calibration du Modèle', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'probability_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Sauvegardée : {path}')


def plot_threshold_analysis(labels, probs, fig_dir):
    thresholds_range = np.arange(0.1, 0.95, 0.05)
    rows = []
    for t in thresholds_range:
        p = (probs >= t).astype(int)
        rows.append({
            't'         : t,
            'accuracy'  : accuracy_score(labels, p),
            'precision' : precision_score(labels, p, zero_division=0),
            'recall'    : recall_score(labels, p, zero_division=0),
            'f1'        : f1_score(labels, p, zero_division=0),
        })

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([r['t'] for r in rows], [r['accuracy']  for r in rows], label='Accuracy',  marker='o', markersize=4)
    ax.plot([r['t'] for r in rows], [r['precision'] for r in rows], label='Precision', marker='s', markersize=4)
    ax.plot([r['t'] for r in rows], [r['recall']    for r in rows], label='Recall',    marker='^', markersize=4, color='red')
    ax.plot([r['t'] for r in rows], [r['f1']        for r in rows], label='F1-Score',  marker='D', markersize=4)
    ax.axvline(x=0.5, color='gray', linestyle='--', lw=1.5, label='Seuil par défaut')
    ax.set_xlabel('Seuil de décision')
    ax.set_ylabel('Score')
    ax.set_title('Impact du Seuil sur les Métriques', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, 'threshold_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Sauvegardée : {path}')


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    fig_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Chargement
    loader = get_test_loader(args.data_dir)

    model = CNNBaseline().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    print(f'✅ Modèle chargé : {args.model}')

    # Inférence
    labels, probs = run_inference(model, loader)
    preds = (probs >= args.threshold).astype(int)

    # Métriques
    metrics, preds = compute_metrics(labels, probs, threshold=args.threshold)
    print_metrics(metrics)
    print('\n--- Rapport détaillé ---')
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    # Figures
    plot_confusion_matrix(labels, preds, fig_dir)
    plot_roc_curve(labels, probs, fig_dir)
    plot_probability_distribution(labels, probs, args.threshold, fig_dir)
    plot_threshold_analysis(labels, probs, fig_dir)

    # Export JSON
    results_path = os.path.join(fig_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'\n✅ Résultats exportés : {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Évaluation du modèle CNN — Personne 4')
    parser.add_argument('--data_dir',   type=str, default='data/chest_xray',       help='Dossier dataset')
    parser.add_argument('--model',      type=str, default='outputs/models/model.pt', help='Chemin du modèle')
    parser.add_argument('--output_dir', type=str, default='outputs',                help='Dossier de sortie')
    parser.add_argument('--threshold',  type=float, default=0.5,                   help='Seuil de décision')
    args = parser.parse_args()
    main(args)