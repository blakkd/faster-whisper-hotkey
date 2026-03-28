"""Tests for models.py (ModelWrapper class)."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class MockSettings:
    """Simple mock settings object for testing."""

    def __init__(
        self, model_type, model_name, device, compute_type=None, language="auto"
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language


class TestModelWrapperInitialization:
    """Test ModelWrapper initialization for different model types."""

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_init_whisper_model(self, mock_whisper):
        """Test loading a whisper model."""
        from faster_whisper_hotkey.models import ModelWrapper

        settings = MockSettings(
            model_type="whisper",
            model_name="small",
            device="cpu",
            compute_type="int8",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "whisper"
        mock_whisper.assert_called_once_with(
            model_size_or_path="small", device="cpu", compute_type="int8"
        )

    @patch("faster_whisper_hotkey.models.ASRModel")
    def test_init_parakeet_model(self, mock_asr):
        """Test loading a parakeet model."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_asr.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="parakeet",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device="cpu",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "parakeet"
        mock_asr.from_pretrained.assert_called_once()

    @patch("faster_whisper_hotkey.models.EncDecMultiTaskModel")
    def test_init_canary_model(self, mock_encdec):
        """Test loading a canary model."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_encdec.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="canary", model_name="nvidia/canary-1b-v2", device="cuda"
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "canary"
        mock_encdec.from_pretrained.assert_called_once()

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_init_voxtral_model_float16(self, mock_voxtral, mock_processor):
        """Test loading a voxtral model with float16."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_voxtral.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "voxtral"
        mock_voxtral.from_pretrained.assert_called_once()

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_init_voxtral_model_int8(self, mock_voxtral, mock_processor):
        """Test loading a voxtral model with int8 quantization."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_voxtral.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="int8",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "voxtral"
        mock_voxtral.from_pretrained.assert_called_once()

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForSpeechSeq2Seq")
    def test_init_cohere_model(self, mock_auto_model, mock_processor):
        """Test loading a cohere model."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "cohere"
        mock_auto_model.from_pretrained.assert_called_once()

    def test_init_unknown_model_type(self):
        """Test that unknown model type raises ValueError."""
        from faster_whisper_hotkey.models import ModelWrapper

        settings = MockSettings(
            model_type="unknown",
            model_name="test",
            device="cpu",
        )

        with pytest.raises(ValueError, match="Unknown model type"):
            ModelWrapper(settings)


class TestModelWrapperTranscribe:
    """Test transcription for different model types."""

    def setup_method(self):
        """Create sample audio data for testing."""
        self.sample_audio = np.random.randn(16000).astype(
            np.float32
        )  # 1 second at 16kHz

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_transcribe_whisper(self, mock_whisper):
        """Test whisper transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_segment = MagicMock(text="hello world")
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper", model_name="tiny", device="cpu", compute_type="int8"
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        assert result == "hello world"
        mock_model.transcribe.assert_called_once()

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_transcribe_whisper_with_language(self, mock_whisper):
        """Test whisper transcription with language parameter."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_segment = MagicMock(text="bonjour le monde")
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper", model_name="tiny", device="cpu", compute_type="int8"
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000, language="fr")

        assert result == "bonjour le monde"
        mock_model.transcribe.assert_called_once()

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_transcribe_whisper_auto_language(self, mock_whisper):
        """Test whisper transcription with auto language (None passed to model)."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_segment = MagicMock(text="hello world")
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper", model_name="tiny", device="cpu", compute_type="int8"
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000, language="auto")

        # language should be None when set to "auto"
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] is None

    @patch("faster_whisper_hotkey.models.ASRModel")
    def test_transcribe_parakeet(self, mock_asr):
        """Test parakeet transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_result = MagicMock(text="parakeet transcription")
        mock_model.transcribe.return_value = [mock_result]
        mock_asr.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="parakeet",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device="cpu",
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        assert result == "parakeet transcription"

    @patch("faster_whisper_hotkey.models.EncDecMultiTaskModel")
    def test_transcribe_canary(self, mock_encdec):
        """Test canary transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_output = MagicMock(text="canary translation")
        mock_model.transcribe.return_value = [mock_output]
        mock_encdec.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="canary", model_name="nvidia/canary-1b-v2", device="cuda"
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000, language="en-de")

        assert result == "canary translation"
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["source_lang"] == "en"
        assert call_kwargs["target_lang"] == "de"

    @patch("faster_whisper_hotkey.models.EncDecMultiTaskModel")
    def test_transcribe_canary_invalid_language(self, mock_encdec):
        """Test canary transcription with invalid language format defaults to en-en."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_output = MagicMock(text="canary translation")
        mock_model.transcribe.return_value = [mock_output]
        mock_encdec.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="canary",
            model_name="nvidia/canary-1b-v2",
            device="cuda",
            language="invalid",  # Invalid: no hyphen
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["source_lang"] == "en"
        assert call_kwargs["target_lang"] == "en"

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_transcribe_voxtral_short_audio(self, mock_voxtral, mock_processor):
        """Test voxtral transcription with short audio (no chunking needed)."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )
        wrapper = ModelWrapper(settings)

        # Mock the internal helper method directly for simplicity
        with patch.object(
            wrapper, "_transcribe_single_chunk_voxtral", return_value="voxtral output"
        ):
            short_audio = np.random.randn(48000).astype(np.float32)  # 3 seconds
            result = wrapper.transcribe(short_audio, 16000)

            assert result == "voxtral output"
            # Verify we called the helper once (no chunking)
            wrapper._transcribe_single_chunk_voxtral.assert_called_once()

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_transcribe_voxtral_chunking(self, mock_voxtral, mock_processor):
        """Test voxtral transcription with long audio requiring chunking."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )
        wrapper = ModelWrapper(settings)

        # Mock the internal helper method with side effects for multiple chunks
        with patch.object(
            wrapper,
            "_transcribe_single_chunk_voxtral",
            side_effect=["first chunk text", "second chunk text"],
        ):
            long_audio = np.random.randn(500000).astype(np.float32)  # ~31 seconds

            result = wrapper.transcribe(long_audio, 16000)

            assert "chunk" in result.lower() or len(result) > 0
            # Verify we called the helper twice (chunking happened)
            assert wrapper._transcribe_single_chunk_voxtral.call_count == 2

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForSpeechSeq2Seq")
    def test_transcribe_cohere(self, mock_auto_model, mock_processor):
        """Test cohere transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["cohere transcription"]
        mock_auto_model.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
            language="en",
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        assert result == "cohere transcription"

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_transcribe_error_handling(self, mock_whisper):
        """Test that transcription errors are handled gracefully."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper", model_name="tiny", device="cpu", compute_type="int8"
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        # Should return empty string on error
        assert result == ""
