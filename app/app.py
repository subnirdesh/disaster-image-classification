"""
app.py
------
Disaster Triage — Field Officer Portal
Clean dark theme UI with all three models selectable.
"""

import sys
import os
import random
import string
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.models import build_model


# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent

CHECKPOINTS = {
    "ResNet-50":    str(BASE_DIR / "outputs/checkpoints/resnet50/best.pth"),
    "Improved CNN": str(BASE_DIR / "outputs/checkpoints/improved/best.pth"),
    "Baseline CNN": str(BASE_DIR / "outputs/checkpoints/baseline/best.pth"),
}

MODEL_KEYS = {
    "ResNet-50":    "resnet50",
    "Improved CNN": "improved",
    "Baseline CNN": "baseline",
}

NUM_CLASSES = 13

CLASS_NAMES = [
    'flood_mild', 'flood_moderate', 'flood_severe',
    'fire_mild', 'fire_moderate', 'fire_severe',
    'earthquake_mild', 'earthquake_moderate', 'earthquake_severe',
    'traffic_incident_mild', 'traffic_incident_moderate', 'traffic_incident_severe',
    'non_disaster_mild',
]

ACTIONS = {
    "flood":            "Deploy water rescue units · Alert downstream communities · Coordinate with NDRRMA",
    "fire":             "Alert fire brigade · Evacuate 500m radius · Coordinate aerial firefighting",
    "earthquake":       "Structural safety assessment · Search & rescue priority · Check for aftershocks",
    "traffic_incident": "Dispatch ambulance · Secure perimeter · Check for fuel spill hazard",
    "non_disaster":     "No immediate action required · Log for routine monitoring",
}

SEVERITY_LABEL = {
    "mild":     "MILD",
    "moderate": "MODERATE",
    "severe":   "SEVERE",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Load models ────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
MODEL_CACHE = {}

def load_all_models():
    for name, ckpt in CHECKPOINTS.items():
        key = MODEL_KEYS[name]
        model = build_model(key, num_classes=NUM_CLASSES)
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()
        MODEL_CACHE[name] = model.to(DEVICE)
        print(f"  ✓ {name}")

print("Loading models...")
load_all_models()


# ── Inference ──────────────────────────────────────────────────────────────────

def generate_claim_id():
    return f"DST-{datetime.now().year}-{''.join(random.choices(string.digits, k=5))}"


def predict(image, model_choice):
    if image is None:
        return "", "", "", "", ""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    model  = MODEL_CACHE[model_choice]
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

    top_idx                = int(np.argmax(probs))
    disaster_type, severity = CLASS_NAMES[top_idx].rsplit("_", 1)

    # aggregate by type
    type_probs = {}
    for i, label in enumerate(CLASS_NAMES):
        dtype = label.rsplit("_", 1)[0]
        type_probs[dtype] = type_probs.get(dtype, 0) + float(probs[i])

    top5 = sorted(type_probs.items(), key=lambda x: -x[1])[:5]
    conf_lines = "\n".join(
        f"{k.replace('_',' ').upper():<22} {v*100:5.1f}%"
        for k, v in top5
    )

    disaster_display = disaster_type.replace("_", " ").upper()
    sev_display      = SEVERITY_LABEL.get(severity, severity.upper())
    confidence       = f"{float(probs[top_idx])*100:.1f}%"
    action           = ACTIONS.get(disaster_type, "Assess situation manually")
    claim_id         = generate_claim_id()

    return disaster_display, sev_display, confidence, action, claim_id, conf_lines


# ── Custom CSS ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0a0a0a !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 40px 24px !important;
}

/* Header */
.header-block {
    border-bottom: 1px solid #222;
    padding-bottom: 28px;
    margin-bottom: 36px;
}
.header-block h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    color: #f0ece4 !important;
    margin-bottom: 6px !important;
}
.header-block p {
    font-size: 12px !important;
    color: #555 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Upload area */
.upload-area .wrap {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 4px !important;
    min-height: 320px !important;
    transition: border-color 0.2s;
}
.upload-area .wrap:hover {
    border-color: #444 !important;
}
.upload-area .wrap.dragging {
    border-color: #c8b89a !important;
    background: #141414 !important;
}
.upload-area svg { color: #333 !important; }
.upload-area .upload-text {
    color: #444 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* Model selector */
.model-selector .wrap {
    gap: 8px !important;
}
.model-selector label {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 3px !important;
    color: #666 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    letter-spacing: 0.05em;
}
.model-selector label:hover {
    border-color: #444 !important;
    color: #aaa !important;
}
.model-selector label.selected,
.model-selector input:checked + label {
    background: #1a1a1a !important;
    border-color: #c8b89a !important;
    color: #c8b89a !important;
}

/* Analyse button */
.analyse-btn {
    background: #c8b89a !important;
    border: none !important;
    border-radius: 3px !important;
    color: #0a0a0a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 14px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    margin-top: 12px !important;
}
.analyse-btn:hover {
    background: #d4c4a8 !important;
}

/* Result fields */
.result-field label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #444 !important;
    margin-bottom: 6px !important;
}
.result-field textarea,
.result-field input {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 3px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    padding: 12px 14px !important;
}

/* Disaster type — big display */
.disaster-type-field textarea {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #c8b89a !important;
    letter-spacing: 0.05em !important;
    min-height: 60px !important;
}

/* Severity */
.severity-field textarea {
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
}

/* Confidence breakdown */
.confidence-field textarea {
    font-size: 11px !important;
    color: #666 !important;
    line-height: 1.8 !important;
    min-height: 120px !important;
}

/* Claim ID */
.claim-field textarea {
    color: #555 !important;
    font-size: 11px !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1a1a1a;
    margin: 24px 0;
}

/* Footer */
.footer-block p {
    font-size: 11px !important;
    color: #333 !important;
    letter-spacing: 0.06em;
    line-height: 1.8;
}

/* Hide webcam / paste buttons */
button[aria-label="Paste from clipboard"],
button[aria-label="Use webcam"],
.source-selection { display: none !important; }

/* Remove gradio default styling noise */
.gap, footer { display: none !important; }
.contain { padding: 0 !important; }
"""


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Disaster Triage") as demo:

        gr.HTML("""
        <div class="header-block">
            <h1>Disaster Triage</h1>
            <p>Field Officer Portal &nbsp;·&nbsp; Emergency Image Classification &nbsp;·&nbsp; AIDER Dataset</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left column ───────────────────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="",
                    elem_classes=["upload-area"],
                    sources=["upload"],
                    show_label=False,
                )

                model_choice = gr.Radio(
                    choices=list(CHECKPOINTS.keys()),
                    value="ResNet-50",
                    label="MODEL",
                    elem_classes=["model-selector"],
                )

                submit_btn = gr.Button(
                    "Analyse",
                    elem_classes=["analyse-btn"],
                )

            # ── Right column ──────────────────────────────────────────────────
            with gr.Column(scale=1):

                disaster_out = gr.Textbox(
                    label="Disaster Type",
                    interactive=False,
                    elem_classes=["result-field", "disaster-type-field"],
                )

                with gr.Row():
                    severity_out = gr.Textbox(
                        label="Severity",
                        interactive=False,
                        elem_classes=["result-field", "severity-field"],
                    )
                    confidence_out = gr.Textbox(
                        label="Confidence",
                        interactive=False,
                        elem_classes=["result-field"],
                    )

                action_out = gr.Textbox(
                    label="Recommended Action",
                    interactive=False,
                    elem_classes=["result-field"],
                    lines=2,
                )

                conf_breakdown = gr.Textbox(
                    label="Confidence Breakdown",
                    interactive=False,
                    elem_classes=["result-field", "confidence-field"],
                    lines=5,
                )

                claim_out = gr.Textbox(
                    label="Claim ID",
                    interactive=False,
                    elem_classes=["result-field", "claim-field"],
                )

        gr.HTML('<hr class="divider">')

        gr.HTML("""
        <div class="footer-block">
            <p>
                ResNet-50 · Improved CNN · Baseline CNN &nbsp;|&nbsp;
                5 disaster types × 3 severity levels &nbsp;|&nbsp;
                AIDER — Kyrkou & Theocharides, 2019
            </p>
        </div>
        """)

        submit_btn.click(
            fn=predict,
            inputs=[image_input, model_choice],
            outputs=[disaster_out, severity_out, confidence_out, action_out, claim_out, conf_breakdown],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        share=False,
        css=CSS,
        theme=gr.themes.Base(
            primary_hue="stone",
            neutral_hue="stone",
            font=gr.themes.GoogleFont("DM Mono"),
        ),
    )