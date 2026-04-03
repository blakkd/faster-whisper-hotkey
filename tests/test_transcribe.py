"""
Tests for transcribe.py - configuration functions and main flow.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestConfigureLlmCorrection:
    """Test _configure_llm_correction function."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_disabled_returns_tuple(self, mock_wrapper):
        """When user selects 'No', should return (False, '', ''), not None."""
        from faster_whisper_hotkey.transcribe import _configure_llm_correction

        # Mock to return "No" for the Yes/No menu
        mock_wrapper.side_effect = ["No"]

        result = _configure_llm_correction()

        assert result == (False, "", "")

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_enabled_with_custom_values(self, mock_wrapper):
        """When user enables LLM correction, should return configured tuple."""
        from faster_whisper_hotkey.transcribe import _configure_llm_correction

        # Simulate: Yes -> custom endpoint -> custom model name
        mock_wrapper.side_effect = [
            "Yes",
            "http://custom:8080/v1",
            "my-model-v2",
        ]

        result = _configure_llm_correction()

        assert result == (True, "http://custom:8080/v1", "my-model-v2")

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_enabled_with_default_endpoint_and_model(self, mock_wrapper):
        """When enabled but uses defaults, should still return valid tuple."""
        from faster_whisper_hotkey.transcribe import _configure_llm_correction

        # Simulate: Yes -> default endpoint -> empty model (should default to "default")
        mock_wrapper.side_effect = [
            "Yes",
            "http://localhost:8678/v1",
            "",  # Empty model name
        ]

        result = _configure_llm_correction()

        assert result == (True, "http://localhost:8678/v1", "default")

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_escape_aborts_returns_none(self, mock_wrapper):
        """When user presses ESC on first menu, should return None."""
        from faster_whisper_hotkey.transcribe import _configure_llm_correction

        # ESC returns None from curses_menu
        mock_wrapper.side_effect = [None]

        result = _configure_llm_correction()

        assert result is None

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_escape_on_endpoint_input(self, mock_wrapper):
        """When user presses ESC during endpoint input, should return None."""
        from faster_whisper_hotkey.transcribe import _configure_llm_correction

        # Simulate: Yes -> ESC on endpoint input
        mock_wrapper.side_effect = ["Yes", None]

        result = _configure_llm_correction()

        assert result is None


class TestCreateSettingsDict:
    """Test _create_settings_dict helper function."""

    def test_create_settings_dict_basic(self):
        """Test creating settings dict with basic parameters."""
        from faster_whisper_hotkey.transcribe import _create_settings_dict

        llm_result = (True, "http://localhost:8678/v1", "mistral")

        result = _create_settings_dict(
            device_name="alsa_output_test",
            model_type="whisper",
            model_name="small",
            compute_type="float16",
            device="cuda",
            language="en",
            hotkey="pause",
            llm_result=llm_result,
        )

        assert result["device_name"] == "alsa_output_test"
        assert result["model_type"] == "whisper"
        assert result["model_name"] == "small"
        assert result["compute_type"] == "float16"
        assert result["device"] == "cuda"
        assert result["language"] == "en"
        assert result["hotkey"] == "pause"
        assert result["llm_correction_enabled"] is True
        assert result["llm_endpoint"] == "http://localhost:8678/v1"
        assert result["llm_model_name"] == "mistral"

    def test_create_settings_dict_no_llm_correction(self):
        """Test settings dict when LLM correction is disabled."""
        from faster_whisper_hotkey.transcribe import _create_settings_dict

        llm_result = (False, "", "")

        result = _create_settings_dict(
            device_name="test_device",
            model_type="parakeet",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            compute_type="float16",
            device="cpu",
            language="",
            hotkey="f4",
            llm_result=llm_result,
        )

        assert result["llm_correction_enabled"] is False
        assert result["llm_endpoint"] == ""
        assert result["llm_model_name"] == ""


class TestSaveAndCreateSettings:
    """Test _save_and_create_settings helper function."""

    @patch("faster_whisper_hotkey.transcribe.save_settings")
    def test_saves_and_returns_settings(self, mock_save):
        """Should save settings and return Settings object."""
        from faster_whisper_hotkey.transcribe import _save_and_create_settings
        from faster_whisper_hotkey.settings import Settings

        settings_dict = {
            "device_name": "test",
            "model_type": "whisper",
            "model_name": "small",
            "compute_type": "int8",
            "device": "cpu",
            "language": "de",
            "hotkey": "insert",
            "llm_correction_enabled": False,
            "llm_endpoint": "",
            "llm_model_name": "",
        }

        result = _save_and_create_settings(settings_dict)

        mock_save.assert_called_once_with(settings_dict)
        assert isinstance(result, Settings)
        assert result.model_name == "small"


class TestGetHotkey:
    """Test _get_hotkey function."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_returns_selected_hotkey_lowercased(self, mock_wrapper):
        """Should return selected hotkey in lowercase."""
        from faster_whisper_hotkey.transcribe import _get_hotkey

        mock_wrapper.side_effect = ["F4"]

        result = _get_hotkey()

        assert result == "f4"

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_esc_returns_none(self, mock_wrapper):
        """ESC should return None."""
        from faster_whisper_hotkey.transcribe import _get_hotkey

        mock_wrapper.side_effect = [None]

        result = _get_hotkey()

        assert result is None


class TestGetDeviceChoice:
    """Test _get_device_choice function."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_cuda_only_shows_cuda(self, mock_wrapper):
        """When cuda_only=True, should only show CUDA option."""
        from faster_whisper_hotkey.transcribe import _get_device_choice

        mock_wrapper.side_effect = ["cuda"]

        result = _get_device_choice(cuda_only=True)

        assert result == "cuda"
        # Verify wrapper was called - it will internally pass only ["cuda"] to curses_menu
        assert mock_wrapper.called

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_not_cuda_only_shows_both_options(self, mock_wrapper):
        """When cuda_only=False, should show both CUDA and CPU options."""
        from faster_whisper_hotkey.transcribe import _get_device_choice

        mock_wrapper.side_effect = ["cpu"]

        result = _get_device_choice(cuda_only=False)

        assert result == "cpu"


class TestMainFlowNewSettings:
    """Test main() flow for new settings scenario."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe.pulsectl")
    @patch("faster_whisper_hotkey.transcribe.MicrophoneTranscriber")
    def test_new_settings_with_llm_disabled_starts_program(self, mock_transcriber_cls, mock_pulsectl, mock_wrapper):
        """When choosing new settings with LLM disabled, program should start (not loop)."""
        from faster_whisper_hotkey.transcribe import main

        # Setup pulsectl mock
        mock_pulse_instance = MagicMock()
        mock_source = MagicMock()
        mock_source.description = "Test Device"
        mock_source.name = "alsa_test"
        mock_pulse_instance.source_list.return_value = [mock_source]
        mock_pulsectl.Pulse.return_value.__enter__.return_value = mock_pulse_instance

        # Simulate user flow:
        # 1. Choose "Choose New Settings"
        # 2. Select audio device "Test Device"
        # 3. Select model type "parakeet-tdt-0.6b-v3"
        # 4. For parakeet device choice: "cpu"
        # 5. Hotkey: "pause"
        # 6. LLM correction: "No"
        call_sequence = [
            "Choose New Settings",
            "Test Device",
            "parakeet-tdt-0.6b-v3",
            "cpu",
            "pause",
            "No",
        ]
        mock_wrapper.side_effect = call_sequence

        # Setup transcriber mock
        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber
        mock_transcriber.run.return_value = None

        # Run main - it should NOT loop infinitely
        try:
            main()
        except SystemExit:
            pass

        # Verify transcriber was created and run
        mock_transcriber_cls.assert_called_once()
        mock_transcriber.run.assert_called_once()


class TestMainFlowLastSettings:
    """Test main() flow for last settings scenario."""

    @patch("faster_whisper_hotkey.transcribe.load_settings")
    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe.MicrophoneTranscriber")
    def test_last_settings_starts_program(self, mock_transcriber_cls, mock_wrapper, mock_load):
        """Loading last settings should start program without reconfiguration."""
        from faster_whisper_hotkey.transcribe import main
        from faster_whisper_hotkey.settings import Settings

        # Mock loaded settings
        loaded_settings = Settings(
            device_name="test_dev",
            model_type="whisper",
            model_name="small",
            compute_type="int8",
            device="cpu",
            language="en",
            hotkey="pause",
        )
        mock_load.return_value = loaded_settings

        # User chooses "Use Last Settings"
        mock_wrapper.side_effect = ["Use Last Settings"]

        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber

        try:
            main()
        except SystemExit:
            pass

        mock_load.assert_called_once()
        mock_transcriber_cls.assert_called_once()
        mock_transcriber.run.assert_called_once()


class TestConfigureWhisper:
    """Test _configure_whisper function."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_configures_whisper_model(self, mock_device, mock_wrapper):
        """Should configure whisper model correctly."""
        from faster_whisper_hotkey.transcribe import _configure_whisper

        # Setup mocks
        mock_device.return_value = "cuda"
        mock_wrapper.side_effect = ["small", "float16", "en"]

        result = _configure_whisper()

        assert result["model_name"] == "small"
        assert result["compute_type"] == "float16"
        assert result["device"] == "cuda"
        assert result["language"] == "en"

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_esc_aborts_configuration(self, mock_device, mock_wrapper):
        """ESC during config should return None."""
        from faster_whisper_hotkey.transcribe import _configure_whisper

        mock_device.return_value = None
        mock_wrapper.side_effect = [None]

        result = _configure_whisper()

        assert result is None


class TestConfigureCanary:
    """Test _configure_canary function."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_configures_canary_model(self, mock_device, mock_wrapper):
        """Should configure canary model correctly."""
        from faster_whisper_hotkey.transcribe import _configure_canary

        mock_device.return_value = "cuda"
        mock_wrapper.side_effect = ["en", "en"]

        result = _configure_canary()

        assert result["model_name"] == "nvidia/canary-1b-v2"
        assert result["device"] == "cuda"
        assert result["language"] == "en-en"

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_esc_aborts_configuration(self, mock_device, mock_wrapper):
        """ESC during config should return None."""
        from faster_whisper_hotkey.transcribe import _configure_canary

        mock_device.return_value = None
        mock_wrapper.side_effect = [None]

        result = _configure_canary()

        assert result is None


class TestConfigureParakeet:
    """Test _configure_parakeet function."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_configures_parakeet_model(self, mock_device, mock_wrapper):
        """Should configure parakeet model correctly."""
        from faster_whisper_hotkey.transcribe import _configure_parakeet

        mock_device.return_value = "cpu"
        mock_wrapper.side_effect = []

        result = _configure_parakeet()

        assert result["model_name"] == "nvidia/parakeet-tdt-0.6b-v3"
        assert result["device"] == "cpu"
        assert result["language"] == ""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_esc_aborts_configuration(self, mock_device, mock_wrapper):
        """ESC during config should return None."""
        from faster_whisper_hotkey.transcribe import _configure_parakeet

        mock_device.return_value = None
        mock_wrapper.side_effect = [None]

        result = _configure_parakeet()

        assert result is None


class TestConfigureVoxtral:
    """Test _configure_voxtral function."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_configures_voxtral_model(self, mock_device, mock_wrapper):
        """Should configure voxtral model correctly."""
        from faster_whisper_hotkey.transcribe import _configure_voxtral

        mock_device.return_value = "cuda"
        mock_wrapper.side_effect = ["float16", "Continue"]

        result = _configure_voxtral()

        assert result["model_name"] == "mistralai/Voxtral-Mini-3B-2507"
        assert result["device"] == "cuda"
        assert result["language"] == "auto"

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_esc_aborts_configuration(self, mock_device, mock_wrapper):
        """ESC during config should return None."""
        from faster_whisper_hotkey.transcribe import _configure_voxtral

        mock_device.return_value = None
        mock_wrapper.side_effect = [None]

        result = _configure_voxtral()

        assert result is None


class TestConfigureCohere:
    """Test _configure_cohere function."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_configures_cohere_model(self, mock_device, mock_wrapper):
        """Should configure cohere model correctly."""
        from faster_whisper_hotkey.transcribe import _configure_cohere

        mock_device.return_value = "cpu"
        mock_wrapper.side_effect = ["en"]

        result = _configure_cohere()

        assert result["model_name"] == "CohereLabs/cohere-transcribe-03-2026"
        assert result["device"] == "cpu"
        assert result["language"] == "en"

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_esc_aborts_configuration(self, mock_device, mock_wrapper):
        """ESC during config should return None."""
        from faster_whisper_hotkey.transcribe import _configure_cohere

        mock_device.return_value = None
        mock_wrapper.side_effect = [None]

        result = _configure_cohere()

        assert result is None


class TestEnglishOnlyModels:
    """Test handling of English-only models in whisper config."""

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    @patch("faster_whisper_hotkey.transcribe._get_device_choice")
    def test_english_only_model_sets_language_to_en(self, mock_device, mock_wrapper):
        """English-only models should auto-set language to 'en'."""
        from faster_whisper_hotkey.transcribe import _configure_whisper, english_only_models_whisper

        # Get an English-only model
        if english_only_models_whisper:
            en_model = list(english_only_models_whisper)[0]
        else:
            en_model = "large-v3.en"

        mock_device.return_value = "cpu"
        mock_wrapper.side_effect = [en_model, "int8"]  # Language shouldn't be prompted

        result = _configure_whisper()

        assert result["language"] == "en"
        assert result["model_name"] == en_model
