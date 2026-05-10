"""
Gradio web app — Medical Imaging AI
Run: python app/main.py
"""
from __future__ import annotations

import os
import sys
import threading
import webbrowser
import warnings
import traceback
from pathlib import Path


def _resource_root() -> Path:
    """Return the directory that holds bundled resources (models/, etc.).

    When the app runs from source: project root (parent of app/).
    When packaged with PyInstaller: sys._MEIPASS (the unpacked bundle dir).
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


# Make sibling packages importable when running from source.
# (PyInstaller handles imports via its own bootloader, so this is a no-op there.)
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import numpy as np
from PIL import Image

from inference.brain_inference import BrainSegmentor
from inference.lung_inference import LungClassifier
from llm.report_generator import OllamaReportGenerator

from app.ui_components import (
    create_segmentation_figure,
    create_lung_results_figure,
    format_brain_metrics_html,
    format_lung_metrics_html,
)

# ---------------------------------------------------------------------------
# Globals — loaded once at startup
# ---------------------------------------------------------------------------

brain_segmentor_2d: BrainSegmentor | None = None
brain_segmentor_3d: BrainSegmentor | None = None
lung_classifier: LungClassifier | None = None
report_generator: OllamaReportGenerator | None = None

MODEL_PATHS = {
    "brain_2d": "models/brain/attnunet2d_best.pth",
    "brain_3d": "models/brain/attnunet3d_best.pth",
    "lung":     "models/lung/chexnet_best.pth",
}

_last_brain_result: dict | None = None
_last_lung_result:  dict | None = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models() -> None:
    global brain_segmentor_2d, brain_segmentor_3d, lung_classifier, report_generator

    project_root = _resource_root()

    brain_2d_path = project_root / MODEL_PATHS["brain_2d"]
    if brain_2d_path.exists():
        try:
            brain_segmentor_2d = BrainSegmentor(str(brain_2d_path), mode="2d")
            print(f"[INFO] Brain 2D U-Net loaded from {brain_2d_path}")
        except Exception as exc:
            warnings.warn(f"[WARN] Could not load Brain 2D model: {exc}")

    brain_3d_path = project_root / MODEL_PATHS["brain_3d"]
    if brain_3d_path.exists():
        try:
            brain_segmentor_3d = BrainSegmentor(str(brain_3d_path), mode="3d")
            print(f"[INFO] Brain 3D U-Net loaded from {brain_3d_path}")
        except Exception as exc:
            warnings.warn(f"[WARN] Could not load Brain 3D model: {exc}")

    lung_path = project_root / MODEL_PATHS["lung"]
    if lung_path.exists():
        try:
            lung_classifier = LungClassifier(str(lung_path), num_classes=4)
            print(f"[INFO] LungClassifier loaded from {lung_path}")
        except Exception as exc:
            warnings.warn(f"[WARN] Could not load Lung model: {exc}")

    report_generator = OllamaReportGenerator(model="llama3")
    if report_generator.is_ollama_available():
        print("[INFO] Ollama is reachable — LLM report generation enabled.")
    else:
        warnings.warn(
            "[WARN] Ollama server is not reachable at http://localhost:11434. "
            "Reports will use the built-in fallback template."
        )


# ---------------------------------------------------------------------------
# Brain MRI tab handler
# ---------------------------------------------------------------------------

def run_brain_segmentation(
    file_obj,
    model_mode: str,
    generate_report: bool,
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    patient_symptoms: str,
) -> tuple:
    global _last_brain_result

    if file_obj is None:
        return (None, "", "", "Hata: Lütfen bir NIfTI dosyası (.nii veya .nii.gz) yükleyin.")

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    mode_key = "2d" if model_mode == "2D U-Net" else "3d"
    segmentor = brain_segmentor_2d if mode_key == "2d" else brain_segmentor_3d

    if segmentor is None:
        missing_path = MODEL_PATHS["brain_2d"] if mode_key == "2d" else MODEL_PATHS["brain_3d"]
        return (
            None,
            _model_not_loaded_html("Beyin Segmentasyon"),
            "",
            f"Uyarı: {model_mode} model ağırlıkları bulunamadı ({missing_path}).",
        )

    try:
        result = segmentor.segment(file_path)
        _last_brain_result = result
    except Exception as exc:
        tb = traceback.format_exc()
        return (None, "", "", f"Segmentasyon hatası: {exc}\n\nDetay:\n{tb}")

    mask = result["mask"]
    probs = result["probabilities"]
    display_vol = probs.mean(axis=0) if probs.ndim == 4 else mask.astype(np.float32)

    try:
        fig_image = create_segmentation_figure(display_vol, mask, num_slices=3)
    except Exception as exc:
        warnings.warn(f"[WARN] Segmentation figure creation failed: {exc}")
        fig_image = Image.new("RGB", (600, 300), color=(240, 240, 240))

    metrics_html = format_brain_metrics_html(result)

    patient_info = {
        "name":     patient_name.strip(),
        "age":      patient_age.strip(),
        "gender":   patient_gender.strip(),
        "symptoms": patient_symptoms.strip(),
    }

    report_text = ""
    if generate_report:
        try:
            report_text = report_generator.generate_brain_report(
                result,
                model_mode=mode_key,
                patient_info=patient_info,
                stream=False,
            )
        except Exception as exc:
            report_text = (
                f"Rapor üretilirken hata oluştu: {exc}\n\n"
                "Ollama sunucusunun çalıştığından emin olun: ollama serve"
            )

    tumor_vol = result.get("tumor_volume_ml", 0.0)
    status = f"Segmentasyon tamamlandı. Tümör hacmi: {tumor_vol:.2f} ml | Model: {model_mode}"
    return (fig_image, metrics_html, report_text, status)


# ---------------------------------------------------------------------------
# Lung X-Ray tab handler
# ---------------------------------------------------------------------------

def run_lung_classification(
    image_input,
    generate_report: bool,
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    patient_symptoms: str,
) -> tuple:
    global _last_lung_result

    if image_input is None:
        return (None, "", "", "Hata: Lütfen bir akciğer X-Ray görüntüsü yükleyin.")

    if lung_classifier is None:
        return (
            None,
            _model_not_loaded_html("Akciğer Sınıflandırma"),
            "",
            f"Uyarı: Akciğer model ağırlıkları bulunamadı ({MODEL_PATHS['lung']}).",
        )

    img_arr = image_input
    if img_arr.dtype != np.uint8:
        img_arr = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    elif img_arr.shape[-1] == 4:
        img_arr = img_arr[..., :3]

    try:
        result = lung_classifier.classify(img_arr)
        _last_lung_result = result
    except Exception as exc:
        tb = traceback.format_exc()
        return (None, "", "", f"Sınıflandırma hatası: {exc}\n\nDetay:\n{tb}")

    gradcam = result.get("gradcam_heatmap", np.zeros((224, 224), dtype=np.float32))
    predictions = result["predictions"]
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]

    try:
        fig_image = create_lung_results_figure(
            img_arr, gradcam, predictions, predicted_class, confidence
        )
    except Exception as exc:
        warnings.warn(f"[WARN] Lung figure creation failed: {exc}")
        fig_image = Image.new("RGB", (600, 300), color=(240, 240, 240))

    metrics_html = format_lung_metrics_html(result)

    patient_info = {
        "name":     patient_name.strip(),
        "age":      patient_age.strip(),
        "gender":   patient_gender.strip(),
        "symptoms": patient_symptoms.strip(),
    }

    report_text = ""
    if generate_report:
        try:
            report_text = report_generator.generate_lung_report(
                result,
                patient_info=patient_info,
                stream=False,
            )
        except Exception as exc:
            report_text = (
                f"Rapor üretilirken hata oluştu: {exc}\n\n"
                "Ollama sunucusunun çalıştığından emin olun: ollama serve"
            )

    status = (
        f"Sınıflandırma tamamlandı. Tahmin: {predicted_class} "
        f"(Güven: %{confidence * 100:.1f})"
    )
    return (fig_image, metrics_html, report_text, status)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_not_loaded_html(model_name: str) -> str:
    return (
        f'<div style="padding:12px;background:#fff3e0;border-left:4px solid #FF9800;'
        f'border-radius:4px;font-family:sans-serif;color:#000000;">'
        f'<strong>Model Yüklenmedi</strong><br>'
        f'{model_name} model ağırlıkları bulunamadı. '
        f'Lütfen eğitilmiş ağırlıkları <code>models/</code> dizinine yerleştirin.'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
/* ══════════════════════════════════════════════════════════════════════
   DARK THEME — Medical Imaging AI
   ══════════════════════════════════════════════════════════════════════ */

/* ── Page & container background ──────────────────────────────────── */
body, html {
    background: #111827 !important;
    min-height: 100vh;
}
.gradio-container {
    background: #111827 !important;
    max-width: 1200px !important;
    margin: auto !important;
}

/* ── Global text: white for all Gradio-rendered elements ──────────── */
.gradio-container * {
    color: #FFFFFF !important;
}

/* ── Block / panel backgrounds: dark ─────────────────────────────── */
.gradio-container .block,
.gradio-container form,
.gradio-container fieldset,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .panel,
.gradio-container section {
    background: #1a2035 !important;
    border-color: #2d3555 !important;
}

/* ── Input & textarea: dark bg, white text ────────────────────────── */
.gradio-container input,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container input[type="search"],
.gradio-container textarea,
.gradio-container select {
    background: #1e2235 !important;
    color: #FFFFFF !important;
    border: 1px solid #3a4060 !important;
    border-radius: 6px !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #9CA3AF !important;
    opacity: 1 !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #6b7ee0 !important;
    box-shadow: 0 0 0 2px rgba(107, 126, 224, 0.25) !important;
    outline: none !important;
}

/* ── Labels ──────────────────────────────────────────────────────── */
.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container legend {
    color: #E0E0E0 !important;
    font-weight: 500;
}

/* ── Markdown headings inside panels ─────────────────────────────── */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container p,
.gradio-container span {
    color: #FFFFFF !important;
}

/* ── Radio / checkbox labels ──────────────────────────────────────── */
.gradio-container [data-testid="radio-group"] span,
.gradio-container [data-testid="checkbox-group"] span,
.gradio-container input[type="radio"] ~ span,
.gradio-container input[type="checkbox"] ~ span {
    color: #E0E0E0 !important;
}

/* ── Tab buttons ─────────────────────────────────────────────────── */
.gradio-container [role="tab"],
.gradio-container .tab-nav button,
.gradio-container .tabs > .tab-nav > button {
    color: #b0b8d8 !important;
    background: transparent !important;
    transition: background 0.3s ease, color 0.3s ease !important;
    border-bottom: 2px solid transparent !important;
}
.gradio-container [role="tab"][aria-selected="true"],
.gradio-container .tab-nav button.selected,
.gradio-container .tabs > .tab-nav > button.selected {
    color: #FFFFFF !important;
    background: #252d4a !important;
    border-bottom-color: #6b7ee0 !important;
}

/* ── File upload area ─────────────────────────────────────────────── */
.gradio-container .upload-container,
.gradio-container [data-testid="file"],
.gradio-container .file-preview-holder {
    background: #1e2235 !important;
    border: 2px dashed #3a4060 !important;
}

/* ── Dropdown / select menus ─────────────────────────────────────── */
.gradio-container .dropdown-menu,
.gradio-container [role="listbox"],
.gradio-container [role="option"] {
    background: #1e2235 !important;
    border-color: #3a4060 !important;
}
.gradio-container [role="option"]:hover {
    background: #252d4a !important;
}

/* ── Primary / stop buttons ──────────────────────────────────────── */
.gradio-container button.primary,
.gradio-container button[variant="primary"] {
    background: #4f6ef7 !important;
    color: #FFFFFF !important;
}
.gradio-container button.stop,
.gradio-container button[variant="stop"] {
    background: #c0392b !important;
    color: #FFFFFF !important;
}

/* ── Scrollbar (dark) ────────────────────────────────────────────── */
.gradio-container ::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
.gradio-container ::-webkit-scrollbar-track {
    background: #1a2035;
}
.gradio-container ::-webkit-scrollbar-thumb {
    background: #3a4060;
    border-radius: 3px;
}

/* ── Header card ─────────────────────────────────────────────────── */
.medical-header {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    margin-bottom: 16px !important;
    backdrop-filter: blur(8px) !important;
}
.medical-header h1 {
    color: #FFFFFF !important;
    margin: 0 0 6px 0 !important;
    font-size: 1.8em !important;
    font-weight: 700 !important;
}
.medical-header p {
    color: #b0b8d8 !important;
    margin: 0 !important;
    font-size: 0.95em !important;
}

/* ── Metrics cards: keep light bg + black text (inline !important) ── */
/* Protects gr.HTML metric boxes from the global white override.      */
/* Black text is enforced via inline style="...!important" in         */
/* ui_components.py — nothing more needed here.                       */

/* ── Radio & Checkbox: belirgin accent rengi ─────────────────────── */
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {
    accent-color: #6b7ee0 !important;
    width: 16px !important;
    height: 16px !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
}

/* Seçili radio/checkbox etiketi vurgusu */
.gradio-container input[type="radio"]:checked + span,
.gradio-container input[type="checkbox"]:checked + span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ── Clear-hint caption ──────────────────────────────────────────── */
.clear-hint {
    font-size: 11px !important;
    color: #9CA3AF !important;
    margin-top: 4px !important;
    margin-bottom: 4px !important;
    display: block !important;
}
"""


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks UI."""

    with gr.Blocks(title="Medical Imaging AI") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="medical-header">
          <h1>🧠 Medical Imaging AI 🫁</h1>
          <p>
            <strong>Beyin MR Tümör Segmentasyonu</strong> ve
            <strong>Akciğer X-Ray Sınıflandırması</strong> —
            Ollama ile Türkçe Radyoloji Raporu
          </p>
          <p style="margin-top:6px;font-size:0.82em;color:#8899bb;">
            ⚠️ Yapay zeka destekli analiz — Sonuçlar uzman radyolog onayı gerektirir.
          </p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Beyin MR ───────────────────────────────────────────
            with gr.Tab("🧠 Beyin MR Segmentasyon"):
                with gr.Row():

                    # Sol kolon — kontroller
                    with gr.Column(scale=1):
                        brain_file = gr.File(
                            label="NIfTI Dosyası (.nii veya .nii.gz)",
                            file_types=[".nii", ".gz"],
                        )
                        with gr.Row():
                            brain_clear_btn = gr.Button(
                                "🗑 Görüntüyü Kaldır",
                                variant="stop",
                                size="sm",
                            )
                        gr.HTML('<span class="clear-hint">Yeni görüntü yüklemek için önce kaldırın</span>')

                        brain_mode = gr.Radio(
                            ["2D U-Net", "3D U-Net"],
                            value="2D U-Net",
                            label="Segmentasyon Modeli",
                        )

                        gr.Markdown("### Hasta Bilgileri")
                        brain_patient_name = gr.Textbox(
                            label="Hasta Adı Soyadı",
                            placeholder="Örn: Ahmet Yılmaz",
                            max_lines=1,
                        )
                        brain_patient_age = gr.Textbox(
                            label="Yaş",
                            placeholder="Örn: 45",
                            max_lines=1,
                        )
                        brain_patient_gender = gr.Radio(
                            ["Erkek", "Kadın", "Belirtilmedi"],
                            value="Belirtilmedi",
                            label="Cinsiyet",
                        )
                        brain_patient_symptoms = gr.Textbox(
                            label="Klinik Semptomlar",
                            placeholder="Örn: Başağrısı, görme bozukluğu, baş dönmesi",
                            lines=3,
                        )

                        brain_report_cb = gr.Checkbox(
                            value=True,
                            label="Türkçe Radyoloji Raporu Üret (Ollama)",
                        )
                        brain_btn = gr.Button(
                            "▶ Segmentasyonu Başlat", variant="primary"
                        )

                    # Sağ kolon — sonuçlar
                    with gr.Column(scale=2):
                        brain_image_out = gr.Image(
                            label="Segmentasyon Sonucu", type="pil"
                        )
                        brain_metrics_out = gr.HTML(label="Metrikler")

                brain_status = gr.Textbox(label="Durum", interactive=False, lines=2)
                brain_report_out = gr.Textbox(
                    label="Türkçe Radyoloji Raporu",
                    lines=15,
                    interactive=False,
                )

                # Bağlantılar
                brain_clear_btn.click(fn=lambda: None, outputs=[brain_file])

                brain_btn.click(
                    fn=run_brain_segmentation,
                    inputs=[
                        brain_file,
                        brain_mode,
                        brain_report_cb,
                        brain_patient_name,
                        brain_patient_age,
                        brain_patient_gender,
                        brain_patient_symptoms,
                    ],
                    outputs=[
                        brain_image_out,
                        brain_metrics_out,
                        brain_report_out,
                        brain_status,
                    ],
                )

            # ── Tab 2: Akciğer X-Ray ─────────────────────────────────────
            with gr.Tab("🫁 Akciğer X-Ray Sınıflandırma"):
                with gr.Row():

                    # Sol kolon — kontroller
                    with gr.Column(scale=1):
                        lung_image_in = gr.Image(
                            label="Akciğer X-Ray Görüntüsü",
                            type="numpy",
                        )
                        with gr.Row():
                            lung_clear_btn = gr.Button(
                                "🗑 Görüntüyü Kaldır",
                                variant="stop",
                                size="sm",
                            )
                        gr.HTML('<span class="clear-hint">Yeni görüntü yüklemek için önce kaldırın</span>')

                        gr.Markdown("### Hasta Bilgileri")
                        lung_patient_name = gr.Textbox(
                            label="Hasta Adı Soyadı",
                            placeholder="Örn: Fatma Kaya",
                            max_lines=1,
                        )
                        lung_patient_age = gr.Textbox(
                            label="Yaş",
                            placeholder="Örn: 62",
                            max_lines=1,
                        )
                        lung_patient_gender = gr.Radio(
                            ["Erkek", "Kadın", "Belirtilmedi"],
                            value="Belirtilmedi",
                            label="Cinsiyet",
                        )
                        lung_patient_symptoms = gr.Textbox(
                            label="Klinik Semptomlar",
                            placeholder="Örn: Öksürük, nefes darlığı, ateş",
                            lines=3,
                        )

                        lung_report_cb = gr.Checkbox(
                            value=True,
                            label="Türkçe Radyoloji Raporu Üret (Ollama)",
                        )
                        lung_btn = gr.Button(
                            "▶ Sınıflandırmayı Başlat", variant="primary"
                        )

                    # Sağ kolon — sonuçlar
                    with gr.Column(scale=2):
                        lung_image_out = gr.Image(
                            label="Grad-CAM Sonucu", type="pil"
                        )
                        lung_metrics_out = gr.HTML(label="Sınıflandırma Sonuçları")

                lung_status = gr.Textbox(label="Durum", interactive=False, lines=2)
                lung_report_out = gr.Textbox(
                    label="Türkçe Radyoloji Raporu",
                    lines=15,
                    interactive=False,
                )

                # Bağlantılar
                lung_clear_btn.click(fn=lambda: None, outputs=[lung_image_in])

                lung_btn.click(
                    fn=run_lung_classification,
                    inputs=[
                        lung_image_in,
                        lung_report_cb,
                        lung_patient_name,
                        lung_patient_age,
                        lung_patient_gender,
                        lung_patient_symptoms,
                    ],
                    outputs=[
                        lung_image_out,
                        lung_metrics_out,
                        lung_report_out,
                        lung_status,
                    ],
                )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _open_browser_delayed(url: str = "http://localhost:7860", delay: float = 2.5) -> None:
    """Open the default browser to the Gradio URL after the server is up."""
    def _open():
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[WARN] Could not open browser automatically: {exc}")
    threading.Timer(delay, _open).start()


if __name__ == "__main__":
    # Frozen (PyInstaller) mode: stdout may be None when --noconsole is used;
    # guard our prints. We keep console enabled for log visibility, but be safe.
    if getattr(sys, "frozen", False) and sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115

    load_models()
    demo = build_ui()
    _open_browser_delayed()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=_CUSTOM_CSS)
