"""
label_severity.py
-----------------
Generates pseudo severity labels (mild / moderate / severe) for AIDER images
using domain-informed visual heuristics.

Why pseudo-labels?
  AIDER provides disaster *type* labels only. True severity ground-truth
  requires expert annotation. In practice, ML teams use proxy signals from
  image features when ground-truth is unavailable — this is standard
  industry practice (see: weak supervision, programmatic labeling).

Heuristics by disaster type:
  FLOOD            — dark/brown water coverage (HSV saturation + value), edge density
  FIRE             — warm pixel ratio (high R, low B), smoke darkness
  EARTHQUAKE       — edge density (rubble = many edges), dark pixel ratio
  TRAFFIC_INCIDENT — edge density (debris, vehicle damage) + dark pixel ratio
  NON-DISASTER     — always "mild" (baseline normality)

Folder mapping (AIDER folder name → class name used in code):
  collapsed_building → earthquake
  fire               → fire
  flooded_areas      → flood
  normal             → non_disaster
  traffic_incident   → traffic_incident

Output:
  data/labels/severity_labels.csv  with columns:
    image_path, disaster_type, severity, severity_idx, label_idx
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# Class names used throughout the code
DISASTER_TYPES = ["flood", "fire", "earthquake", "traffic_incident", "non_disaster"]

# Maps class name → actual folder name in AIDER dataset
FOLDER_MAP = {
    "flood":            "flooded_areas",
    "fire":             "fire",
    "earthquake":       "collapsed_building",
    "traffic_incident": "traffic_incident",
    "non_disaster":     "normal",
}
SEVERITY_LEVELS = ["mild", "moderate", "severe"]

DISASTER_IDX = {d: i for i, d in enumerate(DISASTER_TYPES)}
SEVERITY_IDX = {s: i for i, s in enumerate(SEVERITY_LEVELS)}


# ── Heuristic functions ────────────────────────────────────────────────────────

def _water_coverage(img_bgr: np.ndarray) -> float:
    """Fraction of pixels that are dark/murky brown-blue (flood water proxy)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Brown-murky water: low saturation OR dark value OR blue-ish hue
    low_sat = hsv[:, :, 1] < 80
    dark_val = hsv[:, :, 2] < 120
    blue_hue = (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 140)
    water_mask = low_sat | dark_val | blue_hue
    return float(water_mask.mean())


def _edge_density(img_bgr: np.ndarray) -> float:
    """Fraction of edge pixels (Canny). High = rubble/debris/chaos."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return float((edges > 0).mean())


def _warm_pixel_ratio(img_bgr: np.ndarray) -> float:
    """Fraction of pixels with high red and low blue (fire/flame proxy)."""
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    warm = (r.astype(int) - b.astype(int) > 60) & (r > 150)
    return float(warm.mean())


def _dark_pixel_ratio(img_bgr: np.ndarray) -> float:
    """Fraction of very dark pixels (smoke, ash, storm clouds proxy)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float((gray < 60).mean())


def _texture_variance(img_bgr: np.ndarray) -> float:
    """Local standard deviation — high = chaotic texture (cyclone debris)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    mean, sq_mean = cv2.blur(gray, (9, 9)), cv2.blur(gray ** 2, (9, 9))
    variance_map = sq_mean - mean ** 2
    return float(variance_map.mean())


# ── Per-class severity scoring ─────────────────────────────────────────────────

def score_flood(img: np.ndarray) -> float:
    return 0.6 * _water_coverage(img) + 0.4 * _edge_density(img)

def score_fire(img: np.ndarray) -> float:
    return 0.7 * _warm_pixel_ratio(img) + 0.3 * _dark_pixel_ratio(img)

def score_earthquake(img: np.ndarray) -> float:
    return 0.5 * _edge_density(img) + 0.5 * _dark_pixel_ratio(img)

def score_traffic_incident(img: np.ndarray) -> float:
    # Traffic incidents: vehicle damage, debris, road markings = high edges
    # Emergency lights, fire = warm pixels; nighttime scenes = dark pixels
    return 0.6 * _edge_density(img) + 0.4 * _dark_pixel_ratio(img)

def score_non_disaster(_: np.ndarray) -> float:
    return 0.0  # always mild


SCORERS = {
    "flood":            score_flood,
    "fire":             score_fire,
    "earthquake":       score_earthquake,
    "traffic_incident": score_traffic_incident,
    "non_disaster":     score_non_disaster,
}

# Thresholds tuned so ~33% of disaster images land in each severity bucket
THRESHOLDS = {
    "flood":            (0.6, 0.67),
    "fire":             (0.08, 0.18),
    "earthquake":       (0.12, 0.25),
    "traffic_incident": (0.10, 0.22),
    "non_disaster":     (1.0,  2.0),   # always mild
}


def assign_severity(score: float, disaster_type: str) -> str:
    low, high = THRESHOLDS[disaster_type]
    if score < low:
        return "mild"
    elif score < high:
        return "moderate"
    else:
        return "severe"


# ── Main ───────────────────────────────────────────────────────────────────────

def label_dataset(aider_root: str, output_csv: str) -> pd.DataFrame:
    """
    Walk AIDER directory structure and generate severity labels.

    Actual AIDER folder layout:
      AIDER/
        collapsed_building/  → earthquake
        fire/                → fire
        flooded_areas/       → flood
        normal/              → non_disaster
        traffic_incident/    → traffic_incident
    """
    aider_root = Path(aider_root)
    records = []

    for disaster_type in DISASTER_TYPES:
        folder_name = FOLDER_MAP[disaster_type]
        class_dir = aider_root / folder_name
        if not class_dir.exists():
            print(f"  [!] Missing folder: {class_dir}  (for class: {disaster_type})")
            continue

        image_paths = list(class_dir.glob("*.jpg")) + \
                      list(class_dir.glob("*.jpeg")) + \
                      list(class_dir.glob("*.png"))

        print(f"\n  {disaster_type} ({folder_name}): {len(image_paths)} images")
        if disaster_type == "non_disaster" and len(image_paths) > 600:
            import random
            random.seed(42)
            image_paths = random.sample(image_paths, 600)
            print(f"  → sampled down to 600")

        for img_path in tqdm(image_paths, desc=f"    {disaster_type}", ncols=70):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))

            score = SCORERS[disaster_type](img)
            severity = assign_severity(score, disaster_type)

            records.append({
                "image_path": str(img_path),
                "disaster_type": disaster_type,
                "disaster_idx": DISASTER_IDX[disaster_type],
                "severity": severity,
                "severity_idx": SEVERITY_IDX[severity],
                "score": round(score, 4),
                # Combined label for single-head classification (15 classes)
                "combined_label": f"{disaster_type}_{severity}",
                "combined_idx": DISASTER_IDX[disaster_type] * 3 + SEVERITY_IDX[severity],
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n  Saved {len(df)} labels → {output_csv}")
    return df


def print_distribution(df: pd.DataFrame):
    print("\n── Severity distribution by disaster type ──")
    pivot = pd.crosstab(df["disaster_type"], df["severity"])
    print(pivot)
    print(f"\n  Total images: {len(df)}")
    print(f"  Combined classes (type × severity): {df['combined_label'].nunique()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--aider_root", default="data/raw/AIDER",
                        help="Path to AIDER dataset root directory")
    parser.add_argument("--output_csv", default="data/labels/severity_labels.csv",
                        help="Where to save the generated label CSV")
    args = parser.parse_args()

    print("Generating severity labels...")
    df = label_dataset(args.aider_root, args.output_csv)
    print_distribution(df)