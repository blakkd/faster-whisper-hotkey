"""Extended comprehensive tests for models.py (ModelWrapper class)."""

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


class TestGraniteInitEdgeCases:
    """Test Granite model initialization edge cases."""

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForSpeechSeq2Seq")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_uppercase_model_type(
        self, mock_check, mock_auto_model, mock_processor
    ):
        """Test that uppercase model type is normalized to lowercase."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="GRANITE",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)
        assert wrapper.model_type == "granite"

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_trust_remote_code(
        self, mock_check, mock_auto_model, mock_processor
    ):
        """Test that Granite loads with trust_remote_code=True."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="granite-nar",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cuda",
        )

        ModelWrapper(settings)

        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["trust_remote_code"] is True

        proc_call_kwargs = mock_processor.from_pretrained.call_args[1]
        assert proc_call_kwargs["trust_remote_code"] is True

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    def test_init_granite_old_transformers_version(
        self, mock_auto_model, mock_processor
    ):
        """Test that Granite raises ImportError for old transformers version."""
        from faster_whisper_hotkey.models import ModelWrapper

        import transformers as tf_lib

        original_version = tf_lib.__version__
        tf_lib.__version__ = "4.0.0"

        settings = MockSettings(
            model_type="granite-nar",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cuda",
        )

        try:
            with pytest.raises(ImportError, match="requires transformers>=5.5.3"):
                ModelWrapper(settings)
        finally:
            tf_lib.__version__ = original_version


class TestModelWrapperInitEdgeCases:
    """Test ModelWrapper initialization edge cases."""

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_init_whisper_uppercase_model_type(self, mock_whisper):
        """Test that uppercase model type is normalized to lowercase."""
        from faster_whisper_hotkey.models import ModelWrapper

        settings = MockSettings(
            model_type="WHISPER",
            model_name="small",
            device="cpu",
            compute_type="int8",
        )

        wrapper = ModelWrapper(settings)
        assert wrapper.model_type == "whisper"  # Should be normalized

    @patch("faster_whisper_hotkey.models.AutoModelForSpeechSeq2Seq")
    def test_init_cohere_with_cpu_device(self, mock_auto_model):
        """Test Cohere model loading on CPU."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_processor = MagicMock()

        with patch("faster_whisper_hotkey.models.AutoProcessor") as mock_proc:
            mock_proc.from_pretrained.return_value = mock_processor

            settings = MockSettings(
                model_type="cohere",
                model_name="CohereLabs/cohere-transcribe-03-2026",
                device="cpu",
            )

            wrapper = ModelWrapper(settings)
            assert wrapper.model_type == "cohere"

    @patch("faster_whisper_hotkey.models.EncDecMultiTaskModel")
    def test_init_canary_requires_cuda(self, mock_encdec):
        """Test Canary model initialization (CUDA only)."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_encdec.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="canary",
            model_name="nvidia/canary-1b-v2",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)
        assert wrapper.model_type == "canary"
        call_kwargs = mock_encdec.from_pretrained.call_args[1]
        assert call_kwargs["map_location"] == "cuda"


class TestVoxtralChunkingEdgeCases:
    """Test Voxtral chunking behavior with various edge cases."""

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_voxtral_exactly_at_limit(self, mock_voxtral, mock_processor):
        """Test audio exactly at 30-second limit (480,000 samples)."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper, "_transcribe_single_chunk_voxtral", return_value="exact limit"
        ):
            # Exactly 30 seconds at 16kHz
            exact_audio = np.random.randn(480000).astype(np.float32)
            result = wrapper.transcribe(exact_audio, 16000)

            assert result == "exact limit"
            assert wrapper._transcribe_single_chunk_voxtral.call_count == 1

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_voxtral_just_over_limit(self, mock_voxtral, mock_processor):
        """Test audio just over 30-second limit triggers chunking."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_single_chunk_voxtral",
            side_effect=["chunk1", "chunk2"],
        ):
            # Just over 30 seconds but ensuring second chunk >= 1000 samples (not skipped)
            long_audio = np.random.randn(481000).astype(np.float32)
            result = wrapper.transcribe(long_audio, 16000)

            # Should have chunked (2 calls)
            assert wrapper._transcribe_single_chunk_voxtral.call_count == 2

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_voxtral_skips_very_short_chunks(self, mock_voxtral, mock_processor):
        """Test that very short chunks (<1000 samples) are skipped."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )

        wrapper = ModelWrapper(settings)

        # Create audio that will result in a short final chunk
        with patch.object(
            wrapper,
            "_transcribe_single_chunk_voxtral",
            side_effect=["main chunk"],
        ):
            # 30s + very small amount (will be skipped as <1000 samples)
            audio_with_short_tail = np.random.randn(480500).astype(np.float32)
            result = wrapper.transcribe(audio_with_short_tail, 16000)

            # Only one call since the tail is skipped
            assert wrapper._transcribe_single_chunk_voxtral.call_count == 1

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_voxtral_handles_error_in_individual_chunk(
        self, mock_voxtral, mock_processor
    ):
        """Test that errors in individual chunks don't break the whole transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_single_chunk_voxtral",
            side_effect=[Exception("Chunk 1 failed"), "chunk 2 succeeded"],
        ):
            long_audio = np.random.randn(1000000).astype(np.float32)
            result = wrapper.transcribe(long_audio, 16000)

            # Should get the successful chunk despite error
            assert "succeeded" in result.lower() or len(result) > 0

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_voxtral_all_chunks_fail(self, mock_voxtral, mock_processor):
        """Test behavior when all chunks fail."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_single_chunk_voxtral",
            side_effect=Exception("All chunks fail"),
        ):
            long_audio = np.random.randn(1000000).astype(np.float32)
            result = wrapper.transcribe(long_audio, 16000)

            # Should return empty string
            assert result == ""


class TestCohereChunkingEdgeCases:
    """Test Cohere chunking behavior with various edge cases."""

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_cohere_exactly_at_limit(self, mock_cohere, mock_processor):
        """Test audio exactly at 30-second limit (480,000 samples)."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper, "_transcribe_single_chunk_cohere", return_value="exact limit"
        ):
            exact_audio = np.random.randn(480000).astype(np.float32)
            result = wrapper.transcribe(exact_audio, 16000)

            assert result == "exact limit"
            assert wrapper._transcribe_single_chunk_cohere.call_count == 1

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_cohere_just_over_limit(self, mock_cohere, mock_processor):
        """Test audio just over 30-second limit triggers chunking."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_single_chunk_cohere",
            side_effect=["chunk1", "chunk2"],
        ):
            long_audio = np.random.randn(481000).astype(np.float32)
            result = wrapper.transcribe(long_audio, 16000)

            assert wrapper._transcribe_single_chunk_cohere.call_count == 2

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_cohere_skips_very_short_chunks(self, mock_cohere, mock_processor):
        """Test that very short chunks (<1000 samples) are skipped."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_single_chunk_cohere",
            side_effect=["main chunk"],
        ):
            audio_with_short_tail = np.random.randn(480500).astype(np.float32)
            result = wrapper.transcribe(audio_with_short_tail, 16000)

            assert wrapper._transcribe_single_chunk_cohere.call_count == 1

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_cohere_handles_error_in_individual_chunk(
        self, mock_cohere, mock_processor
    ):
        """Test that errors in individual chunks don't break transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_single_chunk_cohere",
            side_effect=[Exception("Chunk 1 failed"), "chunk 2 succeeded"],
        ):
            long_audio = np.random.randn(1000000).astype(np.float32)
            result = wrapper.transcribe(long_audio, 16000)

            assert "succeeded" in result.lower() or len(result) > 0

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_cohere_all_chunks_fail(self, mock_cohere, mock_processor):
        """Test behavior when all chunks fail."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_single_chunk_cohere",
            side_effect=Exception("All chunks fail"),
        ):
            long_audio = np.random.randn(1000000).astype(np.float32)
            result = wrapper.transcribe(long_audio, 16000)

            assert result == ""


class TestCanaryLanguageParsingEdgeCases:
    """Test Canary language parsing with edge cases."""

    @patch("faster_whisper_hotkey.models.EncDecMultiTaskModel")
    def test_canary_language_no_hyphen(self, mock_encdec):
        """Test Canary with invalid language format (no hyphen)."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_output = MagicMock(text="default to en-en")
        mock_model.transcribe.return_value = [mock_output]
        mock_encdec.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="canary",
            model_name="nvidia/canary-1b-v2",
            device="cuda",
            language="invalid",  # No hyphen
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        wrapper.transcribe(sample_audio, 16000)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["source_lang"] == "en"
        assert call_kwargs["target_lang"] == "en"

    @patch("faster_whisper_hotkey.models.EncDecMultiTaskModel")
    def test_canary_empty_language(self, mock_encdec):
        """Test Canary with empty language string."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_output = MagicMock(text="default to en-en")
        mock_model.transcribe.return_value = [mock_output]
        mock_encdec.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="canary",
            model_name="nvidia/canary-1b-v2",
            device="cuda",
            language="",  # Empty
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        wrapper.transcribe(sample_audio, 16000)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["source_lang"] == "en"
        assert call_kwargs["target_lang"] == "en"

    @patch("faster_whisper_hotkey.models.EncDecMultiTaskModel")
    def test_canary_none_language(self, mock_encdec):
        """Test Canary with None language."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_output = MagicMock(text="default to en-en")
        mock_model.transcribe.return_value = [mock_output]
        mock_encdec.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="canary",
            model_name="nvidia/canary-1b-v2",
            device="cuda",
            language=None,
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == "default to en-en"


class TestWhisperLanguageHandling:
    """Test Whisper language parameter handling."""

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_whisper_language_auto_passes_none(self, mock_whisper):
        """Test that 'auto' language passes None to the model."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_segment = MagicMock(text="detected language")
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper",
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="auto",
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        wrapper.transcribe(sample_audio, 16000)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] is None

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_whisper_language_empty_string(self, mock_whisper):
        """Test that empty language string passes None to the model."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_segment = MagicMock(text="no language set")
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper",
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="",
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        wrapper.transcribe(sample_audio, 16000)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] is None


class TestTranscriptionOutputHandling:
    """Test transcription output formatting and edge cases."""

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_whisper_empty_segments(self, mock_whisper):
        """Test Whisper with no transcribed segments."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)  # Empty segments
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper",
            model_name="tiny",
            device="cpu",
            compute_type="int8",
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == ""

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_whisper_multiple_segments(self, mock_whisper):
        """Test Whisper with multiple transcribed segments."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        segment1 = MagicMock(text="first part")
        segment2 = MagicMock(text="second part")
        segment3 = MagicMock(text="third part")
        mock_model.transcribe.return_value = ([segment1, segment2, segment3], None)
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper",
            model_name="tiny",
            device="cpu",
            compute_type="int8",
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(32000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == "first part second part third part"

    @patch("faster_whisper_hotkey.models.ASRModel")
    def test_parakeet_empty_result(self, mock_asr):
        """Test Parakeet with empty transcription result."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.transcribe.return_value = []  # Empty result
        mock_asr.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="parakeet",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device="cpu",
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == ""

    @patch("faster_whisper_hotkey.models.EncDecMultiTaskModel")
    def test_canary_empty_result(self, mock_encdec):
        """Test Canary with empty transcription result."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.transcribe.return_value = []  # Empty result
        mock_encdec.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="canary",
            model_name="nvidia/canary-1b-v2",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == ""


class TestSuppressOutputContextManager:
    """Test the suppress_output utility."""

    @patch("faster_whisper_hotkey.models.os")
    def test_suppress_output_context_manager(self, mock_os):
        """Test that suppress_output properly manages file descriptors."""
        from faster_whisper_hotkey.models import suppress_output

        with suppress_output():
            pass  # Context should work without errors

        # Verify os operations happened
        assert mock_os.open.called
        assert mock_os.dup.called


class TestModelWrapperErrorHandling:
    """Test error handling in ModelWrapper."""

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_transcription_exception_returns_empty_string(self, mock_whisper):
        """Test that transcription exceptions return empty string."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Critical error")
        mock_whisper.return_value = mock_model

        settings = MockSettings(
            model_type="whisper",
            model_name="small",
            device="cpu",
            compute_type="int8",
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == ""


class TestVoxtralInitWithDifferentPrecisions:
    """Test Voxtral initialization with all precision options."""

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_voxtral_int4_quantization(self, mock_voxtral, mock_processor):
        """Test Voxtral model loading with int4 quantization."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="int4",
        )

        wrapper = ModelWrapper(settings)
        assert wrapper.model_type == "voxtral"

        # Check quantization config was used
        call_kwargs = mock_voxtral.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_voxtral_bfloat16_dtype(self, mock_voxtral, mock_processor):
        """Test Voxtral model loading with bfloat16 dtype."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="bfloat16",
        )

        wrapper = ModelWrapper(settings)
        assert wrapper.model_type == "voxtral"


class TestCohereTranscription:
    """Extended Cohere transcription tests."""

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_cohere_empty_result(self, mock_cohere, mock_processor):
        """Test Cohere with empty transcription result."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
            language="en",
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)

        with patch.object(wrapper, "_transcribe_single_chunk_cohere", return_value=""):
            result = wrapper.transcribe(sample_audio, 16000)

        assert result == ""

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_cohere_language_default(self, mock_cohere, mock_processor):
        """Test Cohere with default language parameter."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_feat_extractor = MagicMock()
        mock_feat_extractor.max_duration = 30
        mock_proc_instance = MagicMock()
        mock_proc_instance.feature_extractor = mock_feat_extractor
        mock_proc_instance.decode.return_value = "result"
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_proc_instance.return_value = mock_inputs
        mock_processor.from_pretrained.return_value = mock_proc_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
            language=None,  # Should default to "en"
        )

        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        wrapper.transcribe(sample_audio, 16000)

        # Check that processor was called with language="en" (defaulted from None)
        mock_proc_instance.assert_called_once()
        call_kwargs = mock_proc_instance.call_args[1]
        assert call_kwargs["language"] == "en"


class TestGraniteTranscription:
    """Extended Granite transcription tests."""

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    @patch("faster_whisper_hotkey.models.torch")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_granite_transcription_with_language(
        self, mock_check, mock_torch, mock_auto_model, mock_processor
    ):
        """Test Granite transcription passes audio correctly."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_tensor = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_ctx
        mock_ctx.__exit__.return_value = None
        mock_torch.no_grad.return_value = mock_ctx

        mock_output = MagicMock()
        mock_output.preds = [1, 2, 3]
        mock_model.transcribe.return_value = mock_output
        mock_processor_instance.batch_decode.return_value = ["transcribed text"]

        settings = MockSettings(
            model_type="granite-nar",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cuda",
            language="en",
        )
        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == "transcribed text"

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    @patch("faster_whisper_hotkey.models.torch")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_granite_empty_transcription(
        self, mock_check, mock_torch, mock_auto_model, mock_processor
    ):
        """Test Granite with empty batch_decode result."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )

        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_tensor = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_ctx
        mock_ctx.__exit__.return_value = None
        mock_torch.no_grad.return_value = mock_ctx

        mock_output = MagicMock()
        mock_output.preds = []
        mock_model.transcribe.return_value = mock_output
        mock_processor_instance.batch_decode.return_value = []

        settings = MockSettings(
            model_type="granite-nar",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cpu",
        )
        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == ""

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    @patch("faster_whisper_hotkey.models.torch")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_granite_transcription_error_handling(
        self, mock_check, mock_torch, mock_auto_model, mock_processor
    ):
        """Test Granite handles transcription errors gracefully."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Granite error")
        mock_auto_model.from_pretrained.return_value = mock_model.eval.return_value = (
            mock_model
        )
        mock_processor.from_pretrained.return_value = MagicMock()

        mock_tensor = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_ctx
        mock_ctx.__exit__.return_value = None
        mock_torch.no_grad.return_value = mock_ctx

        settings = MockSettings(
            model_type="granite-nar",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cuda",
        )
        wrapper = ModelWrapper(settings)
        sample_audio = np.random.randn(16000).astype(np.float32)
        result = wrapper.transcribe(sample_audio, 16000)

        assert result == ""
