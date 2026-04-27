"""
evaluate.py
-----------
Loads all three trained models, runs evaluation on test set,
generates confusion matrices, classification reports, and comparison charts.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.models import build_model


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, ncols=70):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Get only classes that appear in test set
    present = sorted(set(all_labels.tolist()))
    present_names = [class_names[i] for i in present]

    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds, labels=present)
    report    = classification_report(
        all_labels, all_preds,
        labels=present,
        target_names=present_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "report": report,
        "present_names": present_names,
        "preds": all_preds,
        "labels": all_labels,
    }


def plot_confusion_matrix(cm, class_names, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_comparison(results, save_dir):
    model_names = list(results.keys())
    metrics = {
        "Accuracy":  [results[m]["accuracy"]  * 100 for m in model_names],
        "Precision": [results[m]["precision"] * 100 for m in model_names],
        "Recall":    [results[m]["recall"]    * 100 for m in model_names],
        "F1 Score":  [results[m]["f1"]        * 100 for m in model_names],
    }

    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#E24B4A", "#378ADD", "#1D9E75", "#F5A623"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace("resnet50", "ResNet-50").title() for m in model_names], fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Comparison — Accuracy, Precision, Recall, F1", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_curves(log_dirs, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"baseline": "#E24B4A", "improved": "#378ADD", "resnet50": "#1D9E75"}

    for model_name, log_dir in log_dirs.items():
        log_path = os.path.join(log_dir, "training_log.csv")
        if not os.path.exists(log_path):
            continue
        df = pd.read_csv(log_path)
        label = model_name.replace("resnet50", "ResNet-50").title()
        color = colors.get(model_name, "gray")

        axes[0].plot(df["epoch"], df["train_acc"], color=color, label=f"{label} train")
        axes[0].plot(df["epoch"], df["val_acc"],   color=color, label=f"{label} val", linestyle="--")
        axes[1].plot(df["epoch"], df["train_loss"], color=color, label=f"{label} train")
        axes[1].plot(df["epoch"], df["val_loss"],   color=color, label=f"{label} val", linestyle="--")

    for ax, title, ylabel in zip(
        axes,
        ["Accuracy over epochs", "Loss over epochs"],
        ["Accuracy", "Cross-entropy loss"]
    ):
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(results):
    print("\n" + "="*65)
    print(f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("="*65)
    for model_name, r in results.items():
        print(f"{model_name:<12} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% {r['f1']*100:>9.2f}%")
    print("="*65)

    # Find best model
    best = max(results, key=lambda m: results[m]["f1"])
    print(f"\n  Best model: {best.upper()} (F1: {results[best]['f1']*100:.2f}%)")