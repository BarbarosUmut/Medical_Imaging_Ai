"""
Unit tests for LLM report generation.

All Ollama HTTP calls are mocked using unittest.mock — no real network
calls are made during these tests.
"""
from __future__ import annotations

import json
from typing import Iterator
from unittest.mock import MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_BRAIN_RESULT = {
    "mask": None,          # numpy array not required for prompt tests
    "probabilities": None,
    "dice_scores": {"dice_class_1": 0.82, "dice_class_2": 0.75, "dice_mean": 0.785},
    "tumor_volume_ml": 12.34,
    "modalities_found": ["t1", "flair"],
}

SAMPLE_LUNG_RESULT = {
    "predictions": {
        "Normal": 0.05,
        "COVID-19": 0.70,
        "Viral Pneumonia": 0.20,
        "Bacterial Pneumonia": 0.05,
    },
    "top_findings": ["COVID-19", "Viral Pneumonia", "Bacterial Pneumonia"],
    "predicted_class": "COVID-19",
    "confidence": 0.70,
    "gradcam_heatmap": None,
}


def _make_ndjson_response(chunks: list[str], done_at_end: bool = True) -> list[bytes]:
    """Build a list of newline-delimited JSON byte strings mimicking Ollama streaming."""
    lines: list[bytes] = []
    for i, text in enumerate(chunks):
        is_last = done_at_end and i == len(chunks) - 1
        obj = {"response": text, "done": is_last}
        lines.append(json.dumps(obj).encode())
    return lines


# ===========================================================================
# OllamaClient tests
# ===========================================================================

class TestOllamaClient:
    """Tests for llm.ollama_client.OllamaClient."""

    # ------------------------------------------------------------------
    # generate() — non-streaming
    # ------------------------------------------------------------------

    def test_generate_returns_string_on_valid_response(self):
        """generate() should return the 'response' field from Ollama JSON."""
        from llm.ollama_client import OllamaClient

        mock_resp = Mock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Türkçe rapor metni.", "done": True}

        client = OllamaClient()
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = client.generate("Test prompt", stream=False)

        assert isinstance(result, str)
        assert "Türkçe" in result

    def test_generate_raises_on_500_response(self):
        """generate() should raise RuntimeError when the server returns 5xx."""
        from llm.ollama_client import OllamaClient

        mock_resp = Mock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        client = OllamaClient()
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="500"):
                client.generate("Test prompt", stream=False)

    # ------------------------------------------------------------------
    # generate_stream()
    # ------------------------------------------------------------------

    def test_generate_stream_yields_text_chunks(self):
        """generate_stream() should yield incremental text chunks."""
        from llm.ollama_client import OllamaClient

        chunks = ["Beyin", " MR", " raporu"]
        ndjson_lines = _make_ndjson_response(chunks)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(ndjson_lines)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        client = OllamaClient()
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result_chunks = list(client.generate_stream("Test prompt"))

        assert result_chunks == chunks

    def test_generate_stream_stops_on_done_flag(self):
        """generate_stream() must stop yielding when 'done' is True."""
        from llm.ollama_client import OllamaClient

        lines = [
            json.dumps({"response": "chunk1", "done": False}).encode(),
            json.dumps({"response": "chunk2", "done": True}).encode(),
            # This line should NOT be yielded
            json.dumps({"response": "chunk3", "done": False}).encode(),
        ]

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        client = OllamaClient()
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = list(client.generate_stream("prompt"))

        assert result == ["chunk1", "chunk2"]

    def test_generate_returns_generator_when_stream_true(self):
        """generate(stream=True) should return an Iterator, not a str."""
        from llm.ollama_client import OllamaClient

        lines = [json.dumps({"response": "ok", "done": True}).encode()]

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        client = OllamaClient()
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            result = client.generate("prompt", stream=True)

        # Should be a generator / iterator, not a plain string
        assert hasattr(result, "__iter__") and not isinstance(result, str)

    # ------------------------------------------------------------------
    # is_available()
    # ------------------------------------------------------------------

    def test_is_available_returns_true_on_200(self):
        """is_available() should return True when server responds 200."""
        from llm.ollama_client import OllamaClient
        import requests as _requests

        mock_resp = Mock()
        mock_resp.status_code = 200

        client = OllamaClient()
        with patch("llm.ollama_client.requests.get", return_value=mock_resp):
            assert client.is_available() is True

    def test_is_available_returns_false_on_connection_error(self):
        """is_available() should return False when ConnectionError is raised."""
        from llm.ollama_client import OllamaClient
        import requests as _requests

        client = OllamaClient()
        with patch(
            "llm.ollama_client.requests.get",
            side_effect=_requests.exceptions.ConnectionError,
        ):
            assert client.is_available() is False

    def test_is_available_returns_false_on_non_200(self):
        """is_available() should return False on non-200 status codes."""
        from llm.ollama_client import OllamaClient

        mock_resp = Mock()
        mock_resp.status_code = 503

        client = OllamaClient()
        with patch("llm.ollama_client.requests.get", return_value=mock_resp):
            assert client.is_available() is False

    # ------------------------------------------------------------------
    # list_models()
    # ------------------------------------------------------------------

    def test_list_models_parses_response_correctly(self):
        """list_models() should return a flat list of model name strings."""
        from llm.ollama_client import OllamaClient

        mock_resp = Mock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "mistral:latest"},
            ]
        }

        client = OllamaClient()
        with patch("llm.ollama_client.requests.get", return_value=mock_resp):
            models = client.list_models()

        assert models == ["llama3:latest", "mistral:latest"]

    def test_list_models_returns_empty_list_when_no_models(self):
        """list_models() should return [] when 'models' key is empty."""
        from llm.ollama_client import OllamaClient

        mock_resp = Mock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"models": []}

        client = OllamaClient()
        with patch("llm.ollama_client.requests.get", return_value=mock_resp):
            assert client.list_models() == []

    # ------------------------------------------------------------------
    # _post()
    # ------------------------------------------------------------------

    def test_post_raises_runtime_error_on_500(self):
        """_post() should raise RuntimeError when server returns 500."""
        from llm.ollama_client import OllamaClient

        mock_resp = Mock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        client = OllamaClient()
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="500"):
                client._post("/api/generate", {"model": "llama3", "prompt": "hi"})

    def test_post_returns_response_on_success(self):
        """_post() should return the response object on 2xx status."""
        from llm.ollama_client import OllamaClient

        mock_resp = Mock()
        mock_resp.ok = True
        mock_resp.status_code = 200

        client = OllamaClient()
        with patch("llm.ollama_client.requests.post", return_value=mock_resp):
            resp = client._post("/api/generate", {"model": "llama3", "prompt": "hi"})

        assert resp is mock_resp


# ===========================================================================
# OllamaReportGenerator tests
# ===========================================================================

class TestOllamaReportGenerator:
    """Tests for llm.report_generator.OllamaReportGenerator."""

    # ------------------------------------------------------------------
    # Brain report
    # ------------------------------------------------------------------

    def test_generate_brain_report_calls_client_generate(self):
        """generate_brain_report() should delegate to client.generate()."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(return_value="Türkçe beyin raporu.")

        result = generator.generate_brain_report(SAMPLE_BRAIN_RESULT, model_mode="2d")

        generator.client.generate.assert_called_once()
        assert isinstance(result, str)

    def test_generate_brain_report_prompt_contains_turkish_keywords(self):
        """The prompt sent to the LLM must contain Turkish radiology keywords."""
        from llm.report_generator import OllamaReportGenerator

        captured_prompt: list[str] = []

        def capture_and_return(prompt, **kwargs):
            captured_prompt.append(prompt)
            return "rapor"

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(side_effect=capture_and_return)

        generator.generate_brain_report(SAMPLE_BRAIN_RESULT, model_mode="3d")

        assert captured_prompt, "generate() was never called"
        prompt = captured_prompt[0]
        assert "tümör" in prompt.lower() or "Tümör" in prompt
        assert "bulgular" in prompt.lower() or "BULGULAR" in prompt

    def test_generate_brain_report_passes_temperature(self):
        """generate_brain_report() must pass temperature kwarg to the client."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator(temperature=0.1)
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(return_value="rapor")

        generator.generate_brain_report(SAMPLE_BRAIN_RESULT)

        _, kwargs = generator.client.generate.call_args
        assert kwargs.get("temperature") == 0.1

    # ------------------------------------------------------------------
    # Lung report
    # ------------------------------------------------------------------

    def test_generate_lung_report_calls_client_generate(self):
        """generate_lung_report() should delegate to client.generate()."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(return_value="Türkçe akciğer raporu.")

        result = generator.generate_lung_report(SAMPLE_LUNG_RESULT)

        generator.client.generate.assert_called_once()
        assert isinstance(result, str)

    def test_generate_lung_report_prompt_contains_classification_data(self):
        """Lung prompt must contain the predicted class name."""
        from llm.report_generator import OllamaReportGenerator

        captured: list[str] = []

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(side_effect=lambda p, **kw: captured.append(p) or "ok")

        generator.generate_lung_report(SAMPLE_LUNG_RESULT)

        assert captured
        assert "COVID-19" in captured[0]

    # ------------------------------------------------------------------
    # Combined report
    # ------------------------------------------------------------------

    def test_generate_combined_report_calls_client_generate(self):
        """generate_combined_report() should call client.generate() once."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(return_value="Birleşik rapor.")

        result = generator.generate_combined_report(SAMPLE_BRAIN_RESULT, SAMPLE_LUNG_RESULT)

        generator.client.generate.assert_called_once()
        assert isinstance(result, str)

    def test_generate_combined_report_prompt_contains_both_modalities(self):
        """Combined prompt must reference both brain and lung data."""
        from llm.report_generator import OllamaReportGenerator

        captured: list[str] = []

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(side_effect=lambda p, **kw: captured.append(p) or "ok")

        generator.generate_combined_report(SAMPLE_BRAIN_RESULT, SAMPLE_LUNG_RESULT)

        assert captured
        prompt = captured[0]
        # Should reference both beyin (brain) and akciğer (lung)
        assert any(kw in prompt for kw in ["Beyin", "beyin", "MR"])
        assert any(kw in prompt for kw in ["Akciğer", "akciğer", "Lung", "lung", "COVID"])

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def test_streaming_mode_returns_iterator(self):
        """generate_brain_report(stream=True) must return an iterator."""
        from llm.report_generator import OllamaReportGenerator

        def fake_generate(prompt, stream=False, **kwargs):
            if stream:
                return iter(["parça1", " parça2"])
            return "tam metin"

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(side_effect=fake_generate)

        result = generator.generate_brain_report(SAMPLE_BRAIN_RESULT, stream=True)

        assert hasattr(result, "__iter__") and not isinstance(result, str)
        chunks = list(result)
        assert len(chunks) > 0

    def test_lung_streaming_mode_returns_iterator(self):
        """generate_lung_report(stream=True) must return an iterator."""
        from llm.report_generator import OllamaReportGenerator

        def fake_generate(prompt, stream=False, **kwargs):
            if stream:
                return iter(["akciğer", " raporu"])
            return "tam metin"

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=True)
        generator.client.generate = Mock(side_effect=fake_generate)

        result = generator.generate_lung_report(SAMPLE_LUNG_RESULT, stream=True)

        assert hasattr(result, "__iter__") and not isinstance(result, str)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def test_fallback_report_returns_non_empty_turkish_string_for_brain(self):
        """_fallback_report('brain', ...) must return a non-empty Turkish string."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        report = generator._fallback_report("brain", SAMPLE_BRAIN_RESULT)

        assert isinstance(report, str)
        assert len(report) > 0
        # Must contain at least one Turkish keyword
        assert any(kw in report for kw in ["Türkçe", "rapor", "RAPOR", "radyolog", "tümör", "Tümör"])

    def test_fallback_report_returns_non_empty_turkish_string_for_lung(self):
        """_fallback_report('lung', ...) must return a non-empty Turkish string."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        report = generator._fallback_report("lung", SAMPLE_LUNG_RESULT)

        assert isinstance(report, str)
        assert len(report) > 0
        assert any(kw in report for kw in ["akciğer", "Akciğer", "rapor", "RAPOR", "radyolog"])

    def test_fallback_brain_report_includes_tumor_volume(self):
        """Fallback brain report must mention the tumor volume from data."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        report = generator._fallback_report("brain", SAMPLE_BRAIN_RESULT)

        assert "12.34" in report

    def test_fallback_lung_report_includes_predicted_class(self):
        """Fallback lung report must mention the predicted class."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        report = generator._fallback_report("lung", SAMPLE_LUNG_RESULT)

        assert "COVID-19" in report

    def test_fallback_used_when_ollama_unavailable_brain(self):
        """When is_ollama_available() is False, fallback report is returned."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=False)
        generator.client.generate = Mock()  # should NOT be called

        result = generator.generate_brain_report(SAMPLE_BRAIN_RESULT)

        generator.client.generate.assert_not_called()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_used_when_ollama_unavailable_lung(self):
        """When Ollama is unavailable, lung fallback is returned without LLM."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=False)
        generator.client.generate = Mock()

        result = generator.generate_lung_report(SAMPLE_LUNG_RESULT)

        generator.client.generate.assert_not_called()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_stream_returns_iterator_when_unavailable(self):
        """Fallback with stream=True returns an iterator even without Ollama."""
        from llm.report_generator import OllamaReportGenerator

        generator = OllamaReportGenerator()
        generator.client.is_available = Mock(return_value=False)

        result = generator.generate_brain_report(SAMPLE_BRAIN_RESULT, stream=True)

        assert hasattr(result, "__iter__") and not isinstance(result, str)
        chunks = list(result)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)


# ===========================================================================
# Prompt template tests
# ===========================================================================

class TestPromptTemplates:
    """Tests for llm.prompt_templates formatting functions."""

    def test_format_brain_prompt_returns_string(self):
        """format_brain_prompt() must return a non-empty string."""
        from llm.prompt_templates import format_brain_prompt

        result = format_brain_prompt(SAMPLE_BRAIN_RESULT, model_mode="2d")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_brain_prompt_contains_turkish_keywords(self):
        """Brain prompt must contain 'tümör' and 'BULGULAR'."""
        from llm.prompt_templates import format_brain_prompt

        result = format_brain_prompt(SAMPLE_BRAIN_RESULT, model_mode="3d")

        assert "tümör" in result.lower() or "Tümör" in result
        assert "BULGULAR" in result

    def test_format_brain_prompt_contains_tumor_volume(self):
        """Brain prompt must embed the numeric tumor volume."""
        from llm.prompt_templates import format_brain_prompt

        result = format_brain_prompt(SAMPLE_BRAIN_RESULT)

        assert "12.34" in result

    def test_format_brain_prompt_contains_modalities(self):
        """Brain prompt must list the detected MRI modalities."""
        from llm.prompt_templates import format_brain_prompt

        result = format_brain_prompt(SAMPLE_BRAIN_RESULT)

        assert "t1" in result.lower() or "flair" in result.lower()

    def test_format_lung_prompt_returns_string(self):
        """format_lung_prompt() must return a non-empty string."""
        from llm.prompt_templates import format_lung_prompt

        result = format_lung_prompt(SAMPLE_LUNG_RESULT)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_lung_prompt_contains_classification_data(self):
        """Lung prompt must contain the predicted class and confidence."""
        from llm.prompt_templates import format_lung_prompt

        result = format_lung_prompt(SAMPLE_LUNG_RESULT)

        assert "COVID-19" in result
        # Confidence: 0.70 → should appear as percentage somewhere
        assert "70" in result

    def test_format_lung_prompt_contains_required_sections(self):
        """Lung prompt must include BULGULAR and SONUÇ section headers."""
        from llm.prompt_templates import format_lung_prompt

        result = format_lung_prompt(SAMPLE_LUNG_RESULT)

        assert "BULGULAR" in result
        assert "SONUÇ" in result

    def test_format_combined_prompt_returns_string(self):
        """format_combined_prompt() must return a non-empty string."""
        from llm.prompt_templates import format_combined_prompt

        result = format_combined_prompt(SAMPLE_BRAIN_RESULT, SAMPLE_LUNG_RESULT)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_combined_prompt_contains_both_brain_and_lung_data(self):
        """Combined prompt must reference data from both modalities."""
        from llm.prompt_templates import format_combined_prompt

        result = format_combined_prompt(SAMPLE_BRAIN_RESULT, SAMPLE_LUNG_RESULT)

        assert any(kw in result for kw in ["Beyin", "beyin", "MR", "tümör", "Tümör"])
        assert any(kw in result for kw in ["Akciğer", "akciğer", "COVID"])

    def test_brain_template_contains_required_section_headers(self):
        """BRAIN_REPORT_TEMPLATE must define BULGULAR and SONUÇ."""
        from llm.prompt_templates import BRAIN_REPORT_TEMPLATE

        assert "BULGULAR" in BRAIN_REPORT_TEMPLATE
        assert "SONUÇ" in BRAIN_REPORT_TEMPLATE

    def test_lung_template_contains_required_section_headers(self):
        """LUNG_REPORT_TEMPLATE must define BULGULAR and SONUÇ."""
        from llm.prompt_templates import LUNG_REPORT_TEMPLATE

        assert "BULGULAR" in LUNG_REPORT_TEMPLATE
        assert "SONUÇ" in LUNG_REPORT_TEMPLATE

    def test_format_brain_prompt_with_no_mask(self):
        """format_brain_prompt() must not crash when mask is None."""
        from llm.prompt_templates import format_brain_prompt

        data = dict(SAMPLE_BRAIN_RESULT)
        data["mask"] = None

        result = format_brain_prompt(data)
        assert isinstance(result, str)
        # Should say no tumor detected
        assert "Tümör tespit edilmedi" in result

    def test_format_lung_prompt_with_empty_predictions(self):
        """format_lung_prompt() must handle an empty predictions dict."""
        from llm.prompt_templates import format_lung_prompt

        data = {
            "predictions": {},
            "top_findings": [],
            "predicted_class": "Bilinmiyor",
            "confidence": 0.0,
        }

        result = format_lung_prompt(data)
        assert isinstance(result, str)
        assert "Bilinmiyor" in result
