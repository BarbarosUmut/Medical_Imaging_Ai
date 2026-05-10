"""
Turkish clinical radiology report generator using Ollama LLM.

Combines brain MRI segmentation and chest X-ray classification results
into structured Turkish clinical radiology reports.
"""
from __future__ import annotations

from typing import Iterator

from llm.ollama_client import OllamaClient
from llm.prompt_templates import (
    format_brain_prompt,
    format_combined_prompt,
    format_lung_prompt,
)


class OllamaReportGenerator:
    """Generate structured Turkish clinical radiology reports via Ollama.

    Args:
        model:       Ollama model name (default ``"llama3"``).
        base_url:    Ollama server base URL.
        temperature: Sampling temperature (lower = more deterministic).
                     0.3 is recommended for clinical text generation.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
    ) -> None:
        self.client = OllamaClient(base_url=base_url, model=model)
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_brain_report(
        self,
        segmentation_result: dict,
        model_mode: str = "2d",
        patient_info: dict | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate a Turkish radiology report from brain MRI segmentation results.

        Args:
            segmentation_result: Output dict from ``BrainSegmentor.segment()``.
            model_mode: ``"2d"`` or ``"3d"``.
            patient_info: Optional dict with keys name/age/gender/symptoms.
            stream: If ``True``, return a generator that yields text chunks.
        """
        if not self.is_ollama_available():
            fallback = self._fallback_report("brain", segmentation_result, patient_info)
            if stream:
                return iter([fallback])
            return fallback

        prompt = format_brain_prompt(
            segmentation_result, model_mode=model_mode, patient_info=patient_info
        )
        return self.client.generate(
            prompt,
            stream=stream,
            temperature=self.temperature,
        )

    def generate_lung_report(
        self,
        classification_result: dict,
        patient_info: dict | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate a Turkish radiology report from chest X-ray classification results.

        Args:
            classification_result: Output dict from ``LungClassifier.classify()``.
            patient_info: Optional dict with keys name/age/gender/symptoms.
            stream: If ``True``, return a generator that yields text chunks.
        """
        if not self.is_ollama_available():
            fallback = self._fallback_report("lung", classification_result, patient_info)
            if stream:
                return iter([fallback])
            return fallback

        prompt = format_lung_prompt(classification_result, patient_info=patient_info)
        return self.client.generate(
            prompt,
            stream=stream,
            temperature=self.temperature,
        )

    def generate_combined_report(
        self,
        brain_result: dict,
        lung_result: dict,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate a comprehensive Turkish report combining both modalities.

        Args:
            brain_result: Output dict from ``BrainSegmentor.segment()``.
            lung_result:  Output dict from ``LungClassifier.classify()``.
            stream: If ``True``, return a generator that yields text chunks.

        Returns:
            Complete Turkish report string or ``Iterator[str]`` of chunks.
            Falls back if Ollama is unavailable.
        """
        if not self.is_ollama_available():
            brain_fallback = self._fallback_report("brain", brain_result)
            lung_fallback = self._fallback_report("lung", lung_result)
            combined = (
                "## BİRLEŞİK RADYOLOJİ RAPORU\n\n"
                "### Beyin MR\n"
                f"{brain_fallback}\n\n"
                "### Akciğer Grafisi\n"
                f"{lung_fallback}\n\n"
                "## NOT\n"
                "Bu rapor Ollama LLM sunucusuna erişilemediğinden otomatik olarak "
                "oluşturulmuştur. Mutlaka uzman radyolog onayı gerektirir."
            )
            if stream:
                return iter([combined])
            return combined

        prompt = format_combined_prompt(brain_result, lung_result)
        return self.client.generate(
            prompt,
            stream=stream,
            temperature=self.temperature,
        )

    def is_ollama_available(self) -> bool:
        """Return ``True`` if the Ollama server is reachable."""
        return self.client.is_available()

    # ------------------------------------------------------------------
    # Fallback (no LLM)
    # ------------------------------------------------------------------

    def _fallback_report(
        self, result_type: str, data: dict, patient_info: dict | None = None
    ) -> str:
        """Generate a basic Turkish report without LLM when Ollama is unavailable."""
        if result_type == "brain":
            return self._fallback_brain_report(data, patient_info)
        if result_type == "lung":
            return self._fallback_lung_report(data, patient_info)
        return (
            "## RADYOLOJİ RAPORU (Otomatik)\n\n"
            "LLM sunucusuna erişilemediğinden detaylı rapor üretilemedi.\n\n"
            "## NOT\n"
            "Mutlaka uzman radyolog onayı gerektirir."
        )

    # ------------------------------------------------------------------
    # Private fallback helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_brain_report(data: dict, patient_info: dict | None = None) -> str:
        """Minimal Turkish brain MRI report from segmentation data."""
        from llm.prompt_templates import _build_patient_info_str

        tumor_volume_ml: float = float(data.get("tumor_volume_ml", 0.0))
        dice_scores: dict = data.get("dice_scores", {})
        modalities: list = data.get("modalities_found", [])

        dice_lines = (
            "\n".join(f"  - {k}: {v:.3f}" for k, v in dice_scores.items())
            if dice_scores
            else "  - Zemin gerçeği mevcut değil; Dice hesaplanamadı."
        )

        modalities_str = ", ".join(modalities) if modalities else "Bilinmiyor"
        patient_str = _build_patient_info_str(patient_info)

        tumor_comment = (
            f"Segmentasyon sonucunda toplam tümör hacmi **{tumor_volume_ml:.2f} ml** "
            "olarak hesaplanmıştır."
            if tumor_volume_ml > 0
            else "Segmentasyon sonucunda tümör tespit edilmemiştir."
        )

        return (
            "## BEYİN MR RADYOLOJİ RAPORU (Otomatik - LLM Yok)\n\n"
            "## KLİNİK BİLGİ\n"
            f"{patient_str}\n\n"
            "## YÖNTEM\n"
            f"Yapay zeka destekli beyin MR segmentasyonu. Modaliteler: {modalities_str}.\n\n"
            "## BULGULAR\n"
            f"{tumor_comment}\n\n"
            "### Dice Skorları\n"
            f"{dice_lines}\n\n"
            "## SONUÇ\n"
            "Yapay zeka destekli segmentasyon analizi tamamlanmıştır. "
            "Klinik değerlendirme için uzman radyolog incelemesi zorunludur.\n\n"
            "## NOT\n"
            "Bu rapor Ollama LLM sunucusuna erişilemediğinden otomatik olarak "
            "oluşturulmuştur. Mutlaka uzman radyolog onayı gerektirir."
        )

    @staticmethod
    def _fallback_lung_report(data: dict, patient_info: dict | None = None) -> str:
        """Minimal Turkish chest X-ray report from classification data."""
        from llm.prompt_templates import _build_patient_info_str

        predicted_class: str = data.get("predicted_class", "Bilinmiyor")
        confidence: float = float(data.get("confidence", 0.0))
        top_findings: list = data.get("top_findings", [])
        predictions: dict = data.get("predictions", {})
        patient_str = _build_patient_info_str(patient_info)

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        pred_lines = (
            "\n".join(f"  - {cls}: %{prob * 100:.1f}" for cls, prob in sorted_preds)
            if sorted_preds
            else "  - Olasılık verisi mevcut değil."
        )

        top_str = ", ".join(top_findings) if top_findings else "Yok"

        return (
            "## AKCİĞER GRAFİSİ RADYOLOJİ RAPORU (Otomatik - LLM Yok)\n\n"
            "## KLİNİK BİLGİ\n"
            f"{patient_str}\n\n"
            "## YÖNTEM\n"
            "PA akciğer grafisi, yapay zeka destekli sınıflandırma analizi.\n\n"
            "## BULGULAR\n"
            f"Yapay zeka modeli en olası tanı olarak **{predicted_class}** "
            f"(%{confidence * 100:.1f} güven skoru) tespit etmiştir.\n\n"
            "### Sınıf Olasılıkları (İlk 3)\n"
            f"{pred_lines}\n\n"
            f"**En önemli bulgular:** {top_str}\n\n"
            "## SONUÇ\n"
            "Yapay zeka destekli akciğer grafisi analizi tamamlanmıştır. "
            "Klinik değerlendirme için uzman radyolog incelemesi zorunludur.\n\n"
            "## NOT\n"
            "Bu rapor Ollama LLM sunucusuna erişilemediğinden otomatik olarak "
            "oluşturulmuştur. Mutlaka uzman radyolog onayı gerektirir."
        )
