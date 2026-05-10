"""
Turkish clinical radiology prompt templates for brain MRI and chest X-ray.
"""

BRAIN_REPORT_TEMPLATE = """
You are an expert Turkish radiologist. Generate a formal Turkish clinical radiology report based on the following brain MRI segmentation results.

ÖNEMLI KURAL: Hasta bilgileri aşağıda verilmişse aynen kullan. Verilmemişse "Hasta bilgisi girilmedi" yaz. Kesinlikle uydurma hasta adı, yaşı veya klinik bilgi ekleme.

HASTA BİLGİLERİ:
{patient_info}

Segmentasyon Sonuçları:
- Tümör Bölgeleri: {tumor_regions}
- Tümör Hacmi: {tumor_volume_ml:.2f} ml
- Dice Skorları: {dice_scores}
- Görüntüleme Modaliteleri: {modalities}
- Segmentasyon Modeli: {model_mode}

Lütfen aşağıdaki format ile Türkçe klinik radyoloji raporu yaz:

## KLİNİK BİLGİ
[Hasta bilgileri ve klinik semptomlar — yalnızca yukarıda verilen bilgileri kullan]

## YÖNTEM
[Görüntüleme tekniği ve model bilgisi]

## BULGULAR
[Detaylı segmentasyon bulguları, lokalizasyon, boyut]

## SONUÇ
[Klinik değerlendirme ve öneri]

## NOT
Yapay zeka destekli analiz — mutlaka uzman radyolog onayı gerektirir.
"""

LUNG_REPORT_TEMPLATE = """
You are an expert Turkish radiologist. Generate a formal Turkish clinical radiology report based on the following chest X-ray classification results.

ÖNEMLI KURAL: Hasta bilgileri aşağıda verilmişse aynen kullan. Verilmemişse "Hasta bilgisi girilmedi" yaz. Kesinlikle uydurma hasta adı, yaşı veya klinik bilgi ekleme.

HASTA BİLGİLERİ:
{patient_info}

Sınıflandırma Sonuçları:
- Tahmin Edilen Tanı: {predicted_class}
- Güven Skoru: {confidence:.1%}
- Tüm Sınıf Olasılıkları: {predictions}
- En Önemli Bulgular: {top_findings}

Lütfen aşağıdaki format ile Türkçe klinik radyoloji raporu yaz:

## KLİNİK BİLGİ
[Hasta bilgileri ve klinik semptomlar — yalnızca yukarıda verilen bilgileri kullan]

## YÖNTEM
[PA akciğer grafisi, yapay zeka destekli analiz]

## BULGULAR
[Detaylı radyolojik bulgular]

## SONUÇ
[Klinik değerlendirme ve olası tanı]

## ÖNERİ
[İleri tetkik veya tedavi önerisi]

## NOT
Yapay zeka destekli analiz — mutlaka uzman radyolog onayı gerektirir.
"""

COMBINED_REPORT_TEMPLATE = """
You are an expert Turkish radiologist. Generate a comprehensive Turkish clinical radiology report combining brain MRI and chest X-ray findings.

Beyin MR Sonuçları:
{brain_summary}

Akciğer Grafisi Sonuçları:
{lung_summary}

Kapsamlı Türkçe klinik radyoloji raporu yaz:

## KLİNİK BİLGİ
[Klinik arka plan]

## YÖNTEM
[Beyin MR ve PA akciğer grafisi, yapay zeka destekli analiz]

## BULGULAR
### Beyin MR Bulguları
[Detaylı segmentasyon bulguları]

### Akciğer Grafisi Bulguları
[Detaylı radyolojik bulgular]

## SONUÇ
[Birleşik klinik değerlendirme ve öneri]

## NOT
Yapay zeka destekli analiz — mutlaka uzman radyolog onayı gerektirir.
"""


# ---------------------------------------------------------------------------
# Helper format functions
# ---------------------------------------------------------------------------

def _build_patient_info_str(patient_info: dict | None) -> str:
    """Build the patient info block for inclusion in prompts.

    Returns 'Hasta bilgisi girilmedi' when patient_info is None or all fields
    are empty/placeholder so the LLM never invents patient details.
    """
    if not patient_info:
        return "Hasta bilgisi girilmedi"

    name = (patient_info.get("name") or "").strip()
    age = (patient_info.get("age") or "").strip()
    gender = (patient_info.get("gender") or "").strip()
    symptoms = (patient_info.get("symptoms") or "").strip()

    if not any([name, age, gender, symptoms]):
        return "Hasta bilgisi girilmedi"

    lines = []
    lines.append(f"Ad Soyad: {name if name else 'Belirtilmedi'}")
    lines.append(f"Yaş: {age if age else 'Belirtilmedi'}")
    lines.append(f"Cinsiyet: {gender if gender else 'Belirtilmedi'}")
    lines.append(f"Klinik Semptomlar: {symptoms if symptoms else 'Belirtilmedi'}")
    return "\n".join(lines)


def format_brain_prompt(
    segmentation_result: dict,
    model_mode: str = "2d",
    patient_info: dict | None = None,
) -> str:
    """Format brain segmentation result into a prompt string.

    Args:
        segmentation_result: Output dict from BrainSegmentor.segment().
        model_mode: ``"2d"`` or ``"3d"``.
        patient_info: Optional dict with keys name/age/gender/symptoms.

    Returns:
        Formatted prompt string ready for the LLM.
    """
    mask = segmentation_result.get("mask")
    tumor_volume_ml: float = float(segmentation_result.get("tumor_volume_ml", 0.0))
    dice_scores: dict = segmentation_result.get("dice_scores", {})
    modalities: list = segmentation_result.get("modalities_found", [])

    region_labels = {
        1: "Nekrotik çekirdek (sınıf 1)",
        2: "Peritumoural ödem (sınıf 2)",
        3: "Kontrast tutan tümör (sınıf 3)",
    }
    if mask is not None:
        import numpy as np
        present_classes = [c for c in [1, 2, 3] if (mask == c).any()]
    else:
        present_classes = []

    tumor_regions = (
        ", ".join(region_labels[c] for c in present_classes)
        if present_classes
        else "Tümör tespit edilmedi"
    )

    dice_str = (
        ", ".join(f"{k}: {v:.3f}" for k, v in dice_scores.items())
        if dice_scores
        else "Zemin gerçeği mevcut değil"
    )

    modalities_str = ", ".join(modalities) if modalities else "Bilinmiyor"
    model_mode_str = "2D U-Net" if model_mode == "2d" else "3D U-Net (kayan pencere)"

    return BRAIN_REPORT_TEMPLATE.format(
        patient_info=_build_patient_info_str(patient_info),
        tumor_regions=tumor_regions,
        tumor_volume_ml=tumor_volume_ml,
        dice_scores=dice_str,
        modalities=modalities_str,
        model_mode=model_mode_str,
    )


def format_lung_prompt(
    classification_result: dict,
    patient_info: dict | None = None,
) -> str:
    """Format lung classification result into a prompt string.

    Args:
        classification_result: Output dict from LungClassifier.classify().
        patient_info: Optional dict with keys name/age/gender/symptoms.

    Returns:
        Formatted prompt string ready for the LLM.
    """
    predicted_class: str = classification_result.get("predicted_class", "Bilinmiyor")
    confidence: float = float(classification_result.get("confidence", 0.0))
    predictions: dict = classification_result.get("predictions", {})
    top_findings: list = classification_result.get("top_findings", [])

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
    predictions_str = ", ".join(f"{cls}: %{prob * 100:.1f}" for cls, prob in sorted_preds)

    top_findings_str = ", ".join(top_findings) if top_findings else "Yok"

    return LUNG_REPORT_TEMPLATE.format(
        patient_info=_build_patient_info_str(patient_info),
        predicted_class=predicted_class,
        confidence=confidence,
        predictions=predictions_str,
        top_findings=top_findings_str,
    )


def format_combined_prompt(brain_result: dict, lung_result: dict) -> str:
    """Format combined brain and lung results into a prompt string."""
    tumor_volume_ml: float = float(brain_result.get("tumor_volume_ml", 0.0))
    dice_scores: dict = brain_result.get("dice_scores", {})
    modalities: list = brain_result.get("modalities_found", [])

    dice_str = (
        ", ".join(f"{k}: {v:.3f}" for k, v in dice_scores.items())
        if dice_scores
        else "Zemin gerçeği mevcut değil"
    )

    brain_summary = (
        f"Tümör Hacmi: {tumor_volume_ml:.2f} ml | "
        f"Dice Skorları: {dice_str} | "
        f"Modaliteler: {', '.join(modalities) if modalities else 'Bilinmiyor'}"
    )

    predicted_class: str = lung_result.get("predicted_class", "Bilinmiyor")
    confidence: float = float(lung_result.get("confidence", 0.0))
    top_findings: list = lung_result.get("top_findings", [])

    lung_summary = (
        f"Tahmin: {predicted_class} (%{confidence * 100:.1f} güven) | "
        f"En Önemli Bulgular: {', '.join(top_findings) if top_findings else 'Yok'}"
    )

    return COMBINED_REPORT_TEMPLATE.format(
        brain_summary=brain_summary,
        lung_summary=lung_summary,
    )
