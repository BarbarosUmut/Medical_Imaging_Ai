"""
Gradio UI component builders for each tab.
"""
import gradio as gr  # noqa: F401
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

BRAIN_CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
}

LUNG_CLASS_COLORS = {
    "COVID": "#FF4444",
    "Lung_Opacity": "#FF8800",
    "Normal": "#44AA44",
    "Viral Pneumonia": "#8844FF",
}

BRAIN_CLASS_LABELS = {
    1: "Nekrotik Çekirdek",
    2: "Peritümöral Ödem",
    3: "Kontrast Tutan Tümör",
}

# Koyu tema (#1e2235) için inline stil sabitleri.
# Artık beyaz metin kullanıyoruz; !important yine gerekli çünkü
# bazı Gradio sürümleri inline stilleri override edebilir.
_TXT    = 'style="color:#E0E0E0 !important;"'
_HEAD   = 'style="color:#FFFFFF !important; font-weight:700;"'
_MUTE   = 'style="color:#9CA3AF !important;"'
_CELL   = 'style="padding:5px 8px; color:#E0E0E0 !important;"'
_THDR   = ('style="text-align:left; padding:6px 8px; '
           'color:#FFFFFF !important; background:#252d4a !important;"')
_THDR_R = ('style="text-align:right; padding:6px 8px; '
           'color:#FFFFFF !important; background:#252d4a !important;"')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = arr * 0.0
    return arr.astype(np.uint8)


def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Segmentation visualisation
# ---------------------------------------------------------------------------

def create_brain_segmentation_overlay(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    grey = _normalize_to_uint8(image_slice)
    grey_rgb = np.stack([grey, grey, grey], axis=-1)

    h, w = mask_slice.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in BRAIN_CLASS_COLORS.items():
        color_mask[mask_slice == cls_idx] = color

    fg = mask_slice > 0
    result = grey_rgb.copy()
    if fg.any():
        result[fg] = (
            (1.0 - alpha) * grey_rgb[fg].astype(np.float32)
            + alpha * color_mask[fg].astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    return result


def create_segmentation_figure(
    volume: np.ndarray,
    mask: np.ndarray,
    num_slices: int = 3,
) -> Image.Image:
    if volume.ndim == 4:
        vol3d = volume[0]
    elif volume.ndim == 3:
        vol3d = volume
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "Görüntüleme hatası", ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    depth = vol3d.shape[0]
    margin = max(1, depth // 10)
    indices = np.linspace(margin, depth - 1 - margin, num_slices, dtype=int)
    indices = np.clip(indices, 0, depth - 1)

    fig, axes = plt.subplots(1, num_slices, figsize=(5 * num_slices, 5))
    if num_slices == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img_slice = vol3d[idx]
        msk_slice = (
            mask[idx] if mask.shape[0] > idx
            else np.zeros_like(img_slice, dtype=np.int32)
        )
        overlay = create_brain_segmentation_overlay(img_slice, msk_slice, alpha=0.4)
        ax.imshow(overlay)
        ax.set_title(f"Kesit {idx + 1}", fontsize=11)
        ax.axis('off')

    patches = [
        mpatches.Patch(color=np.array(BRAIN_CLASS_COLORS[k]) / 255.0, label=lbl)
        for k, lbl in BRAIN_CLASS_LABELS.items()
    ]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches),
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Beyin MR Segmentasyon Sonuçları", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Lung visualisation
# ---------------------------------------------------------------------------

def create_lung_results_figure(
    image: np.ndarray,
    gradcam: np.ndarray,
    predictions: dict,
    predicted_class: str,
    confidence: float,
) -> Image.Image:
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] == 1:
        image = np.concatenate([image, image, image], axis=-1)

    h_img, w_img = image.shape[:2]

    gradcam_clipped = np.clip(gradcam, 0.0, 1.0)
    jet_r = np.clip(1.5 - np.abs(gradcam_clipped * 4.0 - 3.0), 0, 1)
    jet_g = np.clip(1.5 - np.abs(gradcam_clipped * 4.0 - 2.0), 0, 1)
    jet_b = np.clip(1.5 - np.abs(gradcam_clipped * 4.0 - 1.0), 0, 1)
    jet_rgb = np.stack([jet_r, jet_g, jet_b], axis=-1)

    if gradcam.shape != (h_img, w_img):
        hm_pil = Image.fromarray((gradcam_clipped * 255).astype(np.uint8))
        hm_pil = hm_pil.resize((w_img, h_img), Image.BILINEAR)
        g2 = np.array(hm_pil, dtype=np.float32) / 255.0
        jet_rgb = np.stack([
            np.clip(1.5 - np.abs(g2 * 4.0 - 3.0), 0, 1),
            np.clip(1.5 - np.abs(g2 * 4.0 - 2.0), 0, 1),
            np.clip(1.5 - np.abs(g2 * 4.0 - 1.0), 0, 1),
        ], axis=-1)

    jet_uint8 = (jet_rgb * 255).astype(np.uint8)
    alpha_overlay = 0.4
    gradcam_overlay = (
        (1.0 - alpha_overlay) * image.astype(np.float32)
        + alpha_overlay * jet_uint8.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Orijinal Görüntü", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gradcam_overlay)
    axes[1].set_title(
        f"Grad-CAM: {predicted_class}\n(Güven: %{confidence * 100:.1f})",
        fontsize=12, fontweight='bold',
    )
    axes[1].axis('off')

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_preds]
    probs   = [item[1] for item in sorted_preds]

    bar_colors_norm = [
        tuple(c / 255 for c in _hex_to_rgb(LUNG_CLASS_COLORS.get(cls, "#5588CC")))
        for cls in classes
    ]
    bars = axes[2].barh(classes, probs, color=bar_colors_norm, edgecolor='white', height=0.6)
    axes[2].set_xlim(0, 1.0)
    axes[2].set_xlabel("Olasılık", fontsize=10)
    axes[2].set_title("Sınıf Olasılıkları", fontsize=12, fontweight='bold')
    axes[2].invert_yaxis()

    for bar, prob in zip(bars, probs):
        axes[2].text(
            min(prob + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"%{prob * 100:.1f}",
            va='center', ha='left', fontsize=9,
        )
    for bar, cls in zip(bars, classes):
        if cls == predicted_class:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Metrics HTML — all black text uses inline !important so it survives
# the global  .gradio-container * { color:#fff !important }  CSS rule.
# ---------------------------------------------------------------------------

def format_brain_metrics_html(segmentation_result: dict) -> str:
    tumor_volume_ml: float = float(segmentation_result.get("tumor_volume_ml", 0.0))
    dice_scores: dict      = segmentation_result.get("dice_scores", {})
    modalities: list       = segmentation_result.get("modalities_found", [])
    mask: np.ndarray       = segmentation_result.get("mask", np.array([]))

    modalities_str = ", ".join(modalities) if modalities else "Bilinmiyor"

    # ── Voxel counts ────────────────────────────────────────────────
    voxel_rows = ""
    if mask.size > 0:
        for cls_idx, label in BRAIN_CLASS_LABELS.items():
            count = int((mask == cls_idx).sum())
            color = "rgb({},{},{})".format(*BRAIN_CLASS_COLORS[cls_idx])
            voxel_rows += (
                f'<tr>'
                f'<td {_CELL}>'
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{color};border-radius:2px;margin-right:6px;"></span>'
                f'{label}</td>'
                f'<td style="text-align:right; padding:5px 8px; color:#000000 !important;">'
                f'{count:,} voksel</td>'
                f'</tr>'
            )

    # ── Dice scores ─────────────────────────────────────────────────
    dice_rows = ""
    if dice_scores:
        for key, val in dice_scores.items():
            label = key.replace("_", " ").title()
            pct = val * 100
            bar_color = "#4CAF50" if pct >= 70 else ("#FF9800" if pct >= 40 else "#F44336")
            dice_rows += (
                f'<tr>'
                f'<td {_CELL}>{label}</td>'
                f'<td style="text-align:right; padding:5px 8px;">'
                f'<div style="background:#374060 !important; border-radius:4px; height:14px;'
                f'width:120px; display:inline-block; vertical-align:middle;">'
                f'<div style="background:{bar_color}; border-radius:4px; height:14px;'
                f'width:{min(pct, 100):.0f}%;"></div>'
                f'</div>'
                f'<span style="color:#E0E0E0 !important; margin-left:6px;">{val:.3f}</span>'
                f'</td>'
                f'</tr>'
            )
    else:
        dice_rows = (
            f'<tr><td colspan="2" {_MUTE}>'
            f'Zemin gerçeği yok; Dice hesaplanamadı.</td></tr>'
        )

    tumor_color = "#C62828" if tumor_volume_ml > 0 else "#2E7D32"
    tumor_label = f"{tumor_volume_ml:.2f} ml" if tumor_volume_ml > 0 else "Tümör Tespit Edilmedi"

    return f"""
<div style="font-family:sans-serif; font-size:14px; padding:14px;
            background:#1e2235 !important; border-radius:8px;
            border:1px solid #2d3555; color:#E0E0E0 !important;">

  <h3 style="margin:0 0 10px 0; color:#FFFFFF !important; font-weight:700;">
    Segmentasyon Metrikleri
  </h3>

  <div style="background:#252d4a !important; border-left:4px solid {tumor_color};
              padding:10px 14px; border-radius:4px; margin-bottom:12px;">
    <strong {_HEAD}>Modalite:</strong>
    <span {_TXT}>{modalities_str}</span><br>
    <strong {_HEAD}>Tümör Hacmi:</strong>
    <span style="color:{tumor_color} !important; font-weight:700;">{tumor_label}</span>
  </div>

  <table style="width:100%; border-collapse:collapse; margin-bottom:12px;">
    <thead>
      <tr>
        <th {_THDR}>Sınıf</th>
        <th {_THDR_R}>Vokseller</th>
      </tr>
    </thead>
    <tbody>
      {voxel_rows if voxel_rows
        else f'<tr><td colspan="2" {_MUTE}>Maske verisi yok.</td></tr>'}
    </tbody>
  </table>

  <table style="width:100%; border-collapse:collapse;">
    <thead>
      <tr>
        <th {_THDR}>Dice Skoru</th>
        <th {_THDR_R}>Değer</th>
      </tr>
    </thead>
    <tbody>
      {dice_rows}
    </tbody>
  </table>

  <p style="margin:10px 0 0 0; font-size:11px; color:#9CA3AF !important;">
    * Sonuçlar uzman radyolog onayı gerektirir.
  </p>
</div>
"""


def format_lung_metrics_html(classification_result: dict) -> str:
    predicted_class: str = classification_result.get("predicted_class", "Bilinmiyor")
    confidence: float    = float(classification_result.get("confidence", 0.0))
    top_findings: list   = classification_result.get("top_findings", [])
    predictions: dict    = classification_result.get("predictions", {})

    badge_color    = LUNG_CLASS_COLORS.get(predicted_class, "#5588CC")
    conf_pct       = confidence * 100
    conf_bar_color = "#4CAF50" if conf_pct >= 70 else ("#FF9800" if conf_pct >= 40 else "#F44336")

    # Badge text is explicitly white so it shows on colored background.
    top_badges = " ".join(
        f'<span style="background:{LUNG_CLASS_COLORS.get(f,"#888")}; '
        f'color:#ffffff !important; padding:2px 8px; border-radius:10px; '
        f'font-size:12px; margin-right:4px;">{f}</span>'
        for f in top_findings
    )

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    prob_rows = ""
    for cls, prob in sorted_preds:
        cls_color  = LUNG_CLASS_COLORS.get(cls, "#5588CC")
        pct        = prob * 100
        is_top     = cls == predicted_class
        row_bg     = "background:#2d3555 !important;" if is_top else "background:transparent;"
        row_bold   = "font-weight:700;" if is_top else ""
        star       = " ★" if is_top else ""
        prob_rows += (
            f'<tr style="{row_bg}">'
            f'<td style="padding:5px 8px; {row_bold} color:#FFFFFF !important;">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{cls_color};border-radius:50%;margin-right:6px;"></span>'
            f'{cls}{star}</td>'
            f'<td style="padding:5px 8px;">'
            f'<div style="background:#374060 !important; border-radius:4px; height:14px;'
            f'width:140px; display:inline-block; vertical-align:middle;">'
            f'<div style="background:{cls_color}; border-radius:4px; height:14px;'
            f'width:{min(pct,100):.1f}%;"></div>'
            f'</div></td>'
            f'<td style="padding:5px 8px; text-align:right; color:#E0E0E0 !important;">'
            f'%{pct:.1f}</td>'
            f'</tr>'
        )

    return f"""
<div style="font-family:sans-serif; font-size:14px; padding:14px;
            background:#1e2235 !important; border-radius:8px;
            border:1px solid #2d3555; color:#E0E0E0 !important;">

  <h3 style="margin:0 0 10px 0; color:#FFFFFF !important; font-weight:700;">
    Akciğer Sınıflandırma Sonuçları
  </h3>

  <div style="background:#252d4a !important; border-left:4px solid {badge_color};
              padding:10px 14px; border-radius:4px; margin-bottom:12px;">
    <strong {_HEAD}>Tahmin:</strong>
    <span style="background:{badge_color}; color:#ffffff !important;
                 padding:2px 10px; border-radius:12px;
                 font-weight:700; margin-left:6px;">{predicted_class}</span>
    <br><br>
    <strong {_HEAD}>Güven Skoru:</strong>
    <span {_TXT}>&nbsp;%{conf_pct:.1f}</span>
    <div style="background:#374060 !important; border-radius:4px; height:16px; margin-top:4px;">
      <div style="background:{conf_bar_color}; border-radius:4px; height:16px;
                  width:{min(conf_pct,100):.1f}%;"></div>
    </div>
  </div>

  <div style="margin-bottom:12px;">
    <strong {_HEAD}>Öne Çıkan Bulgular:</strong><br>
    <div style="margin-top:6px;">
      {top_badges if top_badges
        else f'<span {_TXT}>Yok</span>'}
    </div>
  </div>

  <table style="width:100%; border-collapse:collapse;">
    <thead>
      <tr>
        <th {_THDR}>Sınıf</th>
        <th {_THDR}>Dağılım</th>
        <th {_THDR_R}>Olasılık</th>
      </tr>
    </thead>
    <tbody>
      {prob_rows if prob_rows
        else f'<tr><td colspan="3" {_MUTE}>Olasılık verisi yok.</td></tr>'}
    </tbody>
  </table>

  <p style="margin:10px 0 0 0; font-size:11px; color:#9CA3AF !important;">
    * Sonuçlar uzman radyolog onayı gerektirir.
  </p>
</div>
"""
