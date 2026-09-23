"""Tests for models.py (ModelWrapper class)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class MockSettings:
    """Simple mock settings object for testing."""

    def __init__(self, model_type, model_name, device, compute_type=None, language: str | None = "auto"):
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
        mock_whisper.assert_called_once_with(model_size_or_path="small", device="cpu", compute_type="int8")

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

        settings = MockSettings(model_type="canary", model_name="nvidia/canary-1b-v2", device="cuda")

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
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_init_cohere_model(self, mock_cohere, mock_processor):
        """Test loading a cohere model."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_feat_extractor = MagicMock()
        mock_processor_instance = MagicMock()
        mock_processor_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_processor_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "cohere"
        mock_cohere.from_pretrained.assert_called_once()

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_model_cuda(self, mock_check, mock_auto_model, mock_processor):
        """Test loading a granite model on CUDA."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="granite-nar",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "granite-nar"
        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["attn_implementation"] == "sdpa"
        assert call_kwargs["dtype"] == torch.bfloat16

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_model_cpu(self, mock_check, mock_auto_model, mock_processor):
        """Test loading a granite model on CPU."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="granite-nar",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cpu",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "granite-nar"
        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["attn_implementation"] == "sdpa"
        assert call_kwargs["dtype"] == torch.float32

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForMultimodalLM")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_qwen3_asr_model_cuda(self, mock_check, mock_mm_model, mock_processor):
        """Test loading a qwen3-asr model on CUDA."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_mm_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="qwen3-asr",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "qwen3-asr"
        mock_check.assert_called_once_with("5.13.0", "Qwen3-ASR")
        call_kwargs = mock_mm_model.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.bfloat16

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForMultimodalLM")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_qwen3_asr_model_cpu(self, mock_check, mock_mm_model, mock_processor):
        """Test loading a qwen3-asr model on CPU."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_mm_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="qwen3-asr",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            device="cpu",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "qwen3-asr"
        call_kwargs = mock_mm_model.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.bfloat16
        assert call_kwargs["low_cpu_mem_usage"] is False

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForMultimodalLM")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_qwen3_asr_model_int8(self, mock_check, mock_mm_model, mock_processor):
        """Test loading a qwen3-asr model with int8 quantization."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_mm_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="qwen3-asr",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            device="cuda",
            compute_type="int8",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "qwen3-asr"
        call_kwargs = mock_mm_model.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForMultimodalLM")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_qwen3_asr_model_cuda_float32(self, mock_check, mock_mm_model, mock_processor):
        """Test loading a qwen3-asr model on CUDA in float32."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_mm_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="qwen3-asr",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            device="cuda",
            compute_type="float32",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "qwen3-asr"
        call_kwargs = mock_mm_model.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.float32

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForMultimodalLM")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_qwen3_asr_model_cpu_float32(self, mock_check, mock_mm_model, mock_processor):
        """Test loading a qwen3-asr model on CPU in float32."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_mm_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="qwen3-asr",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            device="cpu",
            compute_type="float32",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "qwen3-asr"
        call_kwargs = mock_mm_model.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.float32
        assert call_kwargs["low_cpu_mem_usage"] is False

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_turboctc_model_cuda(self, mock_check, mock_ctc_model, mock_processor):
        """Test loading a granite-turboctc model on CUDA."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cuda",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "granite-turboctc"
        mock_check.assert_called_once_with("5.16.0", "Granite Speech 5.0 TurboCTC")
        call_kwargs = mock_ctc_model.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.bfloat16

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_turboctc_model_cpu(self, mock_check, mock_ctc_model, mock_processor):
        """Test loading a granite-turboctc model on CPU."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cpu",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "granite-turboctc"
        call_kwargs = mock_ctc_model.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.bfloat16
        assert call_kwargs["low_cpu_mem_usage"] is False

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_turboctc_model_int8(self, mock_check, mock_ctc_model, mock_processor):
        """Test loading a granite-turboctc model with int8 quantization."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cuda",
            compute_type="int8",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "granite-turboctc"
        call_kwargs = mock_ctc_model.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_turboctc_model_cuda_float32(self, mock_check, mock_ctc_model, mock_processor):
        """Test loading a granite-turboctc model on CUDA in float32."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cuda",
            compute_type="float32",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "granite-turboctc"
        call_kwargs = mock_ctc_model.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.float32

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_init_granite_turboctc_model_cpu_float32(self, mock_check, mock_ctc_model, mock_processor):
        """Test loading a granite-turboctc model on CPU in float32."""
        import torch

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cpu",
            compute_type="float32",
        )

        wrapper = ModelWrapper(settings)

        assert wrapper.model_type == "granite-turboctc"
        call_kwargs = mock_ctc_model.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.float32
        assert call_kwargs["low_cpu_mem_usage"] is False

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
        self.sample_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_transcribe_whisper(self, mock_whisper):
        """Test whisper transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_segment = MagicMock(text="hello world")
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model

        settings = MockSettings(model_type="whisper", model_name="tiny", device="cpu", compute_type="int8")
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

        settings = MockSettings(model_type="whisper", model_name="tiny", device="cpu", compute_type="int8")
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

        settings = MockSettings(model_type="whisper", model_name="tiny", device="cpu", compute_type="int8")
        wrapper = ModelWrapper(settings)

        wrapper.transcribe(self.sample_audio, 16000, language="auto")

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
        mock_asr.from_pretrained.return_value = mock_model.eval.return_value = mock_model

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
        mock_encdec.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        settings = MockSettings(model_type="canary", model_name="nvidia/canary-1b-v2", device="cuda")
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
        mock_encdec.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        settings = MockSettings(
            model_type="canary",
            model_name="nvidia/canary-1b-v2",
            device="cuda",
            language="invalid",  # Invalid: no hyphen
        )
        wrapper = ModelWrapper(settings)

        wrapper.transcribe(self.sample_audio, 16000)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["source_lang"] == "en"
        assert call_kwargs["target_lang"] == "en"

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_transcribe_voxtral_short_audio(self, mock_voxtral, mock_processor):
        """Test voxtral transcription with short audio."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )
        wrapper = ModelWrapper(settings)

        with patch.object(wrapper, "_transcribe_voxtral", return_value="voxtral output") as mock_fn:
            short_audio = np.random.randn(48000).astype(np.float32)  # 3 seconds
            result = wrapper.transcribe(short_audio, 16000)

            assert result == "voxtral output"
            mock_fn.assert_called_once()

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.VoxtralForConditionalGeneration")
    def test_transcribe_voxtral_long_audio(self, mock_voxtral, mock_processor):
        """Test voxtral transcription with long audio (native chunking)."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_voxtral.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        settings = MockSettings(
            model_type="voxtral",
            model_name="mistralai/Voxtral-Mini-3B-2507",
            device="cuda",
            compute_type="float16",
        )
        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_voxtral",
            return_value="long audio transcription result",
        ) as mock_fn:
            long_audio = np.random.randn(1000000).astype(np.float32)  # ~62 seconds

            result = wrapper.transcribe(long_audio, 16000)

            assert result == "long audio transcription result"
            # Single call - native chunking handles long audio internally
            mock_fn.assert_called_once()

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.CohereAsrForConditionalGeneration")
    def test_transcribe_cohere(self, mock_cohere, mock_processor):
        """Test cohere transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_cohere.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_feat_extractor = MagicMock()
        mock_processor_instance = MagicMock()
        mock_processor_instance.feature_extractor = mock_feat_extractor
        mock_processor.from_pretrained.return_value = mock_processor_instance

        settings = MockSettings(
            model_type="cohere",
            model_name="CohereLabs/cohere-transcribe-03-2026",
            device="cuda",
            language="en",
        )
        wrapper = ModelWrapper(settings)

        with patch.object(
            wrapper,
            "_transcribe_cohere",
            return_value="cohere transcription",
        ):
            result = wrapper.transcribe(self.sample_audio, 16000)

        assert result == "cohere transcription"

    @patch("faster_whisper_hotkey.models.ModelWrapper._warmup_cuda")
    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    @patch("faster_whisper_hotkey.models.torch")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_transcribe_granite(self, mock_check, mock_torch, mock_auto_model, mock_processor, mock_warmup):
        """Test granite transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        # Mock torch operations
        mock_tensor = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_ctx
        mock_ctx.__exit__.return_value = None
        mock_torch.no_grad.return_value = mock_ctx

        # Mock transcribe output
        mock_output = MagicMock()
        mock_output.preds = [0, 1, 2]
        mock_model.transcribe.return_value = mock_output

        mock_processor_instance.batch_decode.return_value = ["granite transcription"]

        settings = MockSettings(
            model_type="granite-nar",
            model_name="ibm-granite/granite-speech-4.1-2b-nar",
            device="cuda",
            language="en",
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        assert result == "granite transcription"
        mock_model.transcribe.assert_called_once()

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModel")
    @patch("faster_whisper_hotkey.models.torch")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_transcribe_granite_empty_result(self, mock_check, mock_torch, mock_auto_model, mock_processor):
        """Test granite with empty transcription result."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model

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

        result = wrapper.transcribe(self.sample_audio, 16000)

        assert result == ""

    @patch("faster_whisper_hotkey.models.ModelWrapper._warmup_cuda")
    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForMultimodalLM")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_transcribe_qwen3_asr(self, mock_check, mock_mm_model, mock_processor, mock_warmup):
        """Test qwen3-asr transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_mm_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_transcription_request.return_value = mock_inputs
        mock_processor_instance.decode.return_value = ["qwen3 transcription"]
        mock_processor.from_pretrained.return_value = mock_processor_instance

        settings = MockSettings(
            model_type="qwen3-asr",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            device="cuda",
            compute_type="bfloat16",
            language="en",
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000, language="en")

        assert result == "qwen3 transcription"
        mock_processor_instance.apply_transcription_request.assert_called_once_with(
            audio=self.sample_audio, language="en"
        )
        generate_kwargs = mock_model.generate.call_args[1]
        assert generate_kwargs["max_new_tokens"] == 512
        assert generate_kwargs["do_sample"] is False
        decode_kwargs = mock_processor_instance.decode.call_args[1]
        assert decode_kwargs["return_format"] == "transcription_only"

    @patch("faster_whisper_hotkey.models.ModelWrapper._warmup_cuda")
    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForMultimodalLM")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_transcribe_qwen3_asr_auto_language(self, mock_check, mock_mm_model, mock_processor, mock_warmup):
        """Test qwen3-asr transcription with auto language (None passed to processor)."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_mm_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_transcription_request.return_value = mock_inputs
        mock_processor_instance.decode.return_value = ["qwen3 transcription"]
        mock_processor.from_pretrained.return_value = mock_processor_instance

        settings = MockSettings(
            model_type="qwen3-asr",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            device="cuda",
            compute_type="bfloat16",
        )
        wrapper = ModelWrapper(settings)

        wrapper.transcribe(self.sample_audio, 16000, language="auto")

        mock_processor_instance.apply_transcription_request.assert_called_once_with(
            audio=self.sample_audio, language=None
        )

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForMultimodalLM")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_transcribe_qwen3_asr_empty_result(self, mock_check, mock_mm_model, mock_processor):
        """Test qwen3-asr with empty transcription result."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.dtype = MagicMock()
        mock_mm_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_transcription_request.return_value = mock_inputs
        mock_processor_instance.decode.return_value = []
        mock_processor.from_pretrained.return_value = mock_processor_instance

        settings = MockSettings(
            model_type="qwen3-asr",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            device="cpu",
            compute_type="bfloat16",
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        assert result == ""

    @patch("faster_whisper_hotkey.models.ModelWrapper._warmup_cuda")
    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_transcribe_granite_turboctc(self, mock_check, mock_ctc_model, mock_processor, mock_warmup):
        """Test granite-turboctc transcription."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_model.dtype = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = mock_inputs
        mock_processor_instance.batch_decode.return_value = ["granite turboctc transcription"]
        mock_processor.from_pretrained.return_value = mock_processor_instance

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cuda",
            compute_type="bfloat16",
            language="en",
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000, language="en")

        assert result == "granite turboctc transcription"
        mock_processor_instance.assert_called_once_with(
            self.sample_audio, sampling_rate=16000, device="cuda", return_tensors="pt"
        )
        mock_processor_instance.batch_decode.assert_called_once()
        decode_kwargs = mock_processor_instance.batch_decode.call_args[1]
        assert decode_kwargs["skip_special_tokens"] is True

    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_transcribe_granite_turboctc_empty_result(self, mock_check, mock_ctc_model, mock_processor):
        """Test granite-turboctc with empty transcription result."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.dtype = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = mock_inputs
        mock_processor_instance.batch_decode.return_value = []
        mock_processor.from_pretrained.return_value = mock_processor_instance

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cpu",
            compute_type="bfloat16",
        )
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        assert result == ""

    @patch("faster_whisper_hotkey.models.WhisperModel")
    def test_transcribe_error_handling(self, mock_whisper):
        """Test that transcription errors are handled gracefully."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_whisper.return_value = mock_model

        settings = MockSettings(model_type="whisper", model_name="tiny", device="cpu", compute_type="int8")
        wrapper = ModelWrapper(settings)

        result = wrapper.transcribe(self.sample_audio, 16000)

        # Should return empty string on error
        assert result == ""


class TestCudaWarmup:
    """Test the one-time CUDA warmup run at model load time."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the load-time warmup")
    @patch("faster_whisper_hotkey.models.ModelWrapper.transcribe")
    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_warmup_runs_on_cuda(self, mock_check, mock_ctc_model, mock_processor, mock_transcribe):
        """A short silent clip is transcribed once after loading on CUDA."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model
        mock_processor.from_pretrained.return_value = MagicMock()

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cuda",
            compute_type="bfloat16",
            language="en",
        )

        ModelWrapper(settings)

        mock_transcribe.assert_called_once()
        dummy = mock_transcribe.call_args[0][0]
        assert dummy.shape == (32000,)
        assert dummy.dtype == np.float32
        assert not dummy.any()
        assert mock_transcribe.call_args[1] == {"sample_rate": 16000, "language": "en"}

    @patch("faster_whisper_hotkey.models.ModelWrapper.transcribe")
    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_warmup_skipped_on_cpu(self, mock_check, mock_ctc_model, mock_processor, mock_transcribe):
        """No warmup for CPU loads."""
        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model
        mock_processor.from_pretrained.return_value = MagicMock()

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cpu",
            compute_type="bfloat16",
        )

        ModelWrapper(settings)

        mock_transcribe.assert_not_called()

    @patch("faster_whisper_hotkey.models.torch.cuda.is_available")
    @patch("faster_whisper_hotkey.models.ModelWrapper.transcribe")
    @patch("faster_whisper_hotkey.models.AutoProcessor")
    @patch("faster_whisper_hotkey.models.AutoModelForCTC")
    @patch("faster_whisper_hotkey.models._check_transformers_version")
    def test_warmup_skipped_when_cuda_unavailable(
        self, mock_check, mock_ctc_model, mock_processor, mock_transcribe, mock_is_available
    ):
        """device=cuda without a CUDA runtime: no warmup."""
        mock_is_available.return_value = False

        from faster_whisper_hotkey.models import ModelWrapper

        mock_model = MagicMock()
        mock_ctc_model.from_pretrained.return_value = mock_model.eval.return_value = mock_model
        mock_processor.from_pretrained.return_value = MagicMock()

        settings = MockSettings(
            model_type="granite-turboctc",
            model_name="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            device="cuda",
            compute_type="bfloat16",
        )

        ModelWrapper(settings)

        mock_transcribe.assert_not_called()
