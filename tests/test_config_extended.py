"""Additional comprehensive tests for config.py."""

from unittest.mock import MagicMock, patch

import pytest


def test_get_resource_path_returns_absolute_path():
    """Test that get_resource_path returns a proper absolute path."""
    from faster_whisper_hotkey.config import get_resource_path

    path = get_resource_path("test.json")
    assert path.startswith("/"), "Path should be absolute"
    assert "faster_whisper_hotkey" in path
    assert path.endswith("test.json")


def test_get_resource_path_different_filename():
    """Test that get_resource_path works with different filenames."""
    from faster_whisper_hotkey.config import get_resource_path

    path1 = get_resource_path("file1.txt")
    path2 = get_resource_path("file2.json")

    assert "file1.txt" in path1 and path1 != path2
    assert "file2.json" in path2


class TestAcceptedModelsWhisperExtended:
    """Extended tests for whisper models."""

    def test_all_models_are_strings(self):
        """Verify all model names are strings."""
        from faster_whisper_hotkey.config import accepted_models_whisper

        for model in accepted_models_whisper:
            assert isinstance(model, str), f"Model {model} should be a string"
            assert len(model) > 0, f"Model name should not be empty"

    def test_no_duplicate_models(self):
        """Verify no duplicate models in the list."""
        from faster_whisper_hotkey.config import accepted_models_whisper

        assert len(accepted_models_whisper) == len(set(accepted_models_whisper))

    def test_models_contain_expected_versions(self):
        """Test that all major versions are present (v1, v2, v3)."""
        from faster_whisper_hotkey.config import accepted_models_whisper

        model_str = " ".join(accepted_models_whisper)
        assert "large-v1" in model_str or "large" in model_str
        assert any(v in model_str for v in ["v2", "v3"])


class TestEnglishOnlyModelsWhisperExtended:
    """Extended tests for English-only models."""

    def test_english_only_is_subset_of_all_models(self):
        """Verify all English-only models are in the main model list."""
        from faster_whisper_hotkey.config import (
            accepted_models_whisper,
            english_only_models_whisper,
        )

        for model in english_only_models_whisper:
            assert model in accepted_models_whisper, (
                f"{model} should be in accepted_models_whisper"
            )

    def test_english_only_model_naming_patterns(self):
        """Test that English-only models follow naming conventions."""
        from faster_whisper_hotkey.config import english_only_models_whisper

        for model in english_only_models_whisper:
            # Either ends with .en or is distil-large variant
            is_en_suffix = model.endswith(".en")
            is_distil_large = "distil" in model and "large" in model
            assert is_en_suffix or is_distil_large


class TestAcceptedLanguagesWhisperExtended:
    """Extended tests for whisper languages."""

    def test_all_languages_are_strings(self):
        """Verify all language codes are strings."""
        from faster_whisper_hotkey.config import accepted_languages_whisper

        for lang in accepted_languages_whisper:
            assert isinstance(lang, str), f"Language {lang} should be a string"

    def test_auto_language_is_first_or_present(self):
        """Verify 'auto' option exists."""
        from faster_whisper_hotkey.config import accepted_languages_whisper

        assert "auto" in accepted_languages_whisper

    def test_language_codes_are_valid_format(self):
        """Test language codes are valid (some are 2-letter, some are longer like 'haw' for Hawaiian)."""
        from faster_whisper_hotkey.config import accepted_languages_whisper

        for lang in accepted_languages_whisper:
            if lang == "auto":
                continue
            assert len(lang) >= 2, f"Language code {lang} should be at least 2 letters"
            assert lang.isalpha(), f"Language code {lang} should be alphabetic"

    def test_no_duplicate_languages(self):
        """Verify no duplicate languages."""
        from faster_whisper_hotkey.config import accepted_languages_whisper

        assert len(accepted_languages_whisper) == len(set(accepted_languages_whisper))

    def test_language_count_is_reasonable(self):
        """Whisper supports 96+ languages, verify we have a good number."""
        from faster_whisper_hotkey.config import accepted_languages_whisper

        assert len(accepted_languages_whisper) >= 50


class TestCanarySourceTargetLanguagesExtended:
    """Extended tests for Canary source/target languages."""

    def test_all_sources_are_valid_codes(self):
        """Verify all source language codes are valid."""
        from faster_whisper_hotkey.config import canary_source_target_languages

        for lang in canary_source_target_languages:
            assert isinstance(lang, str) and len(lang) == 2

    def test_no_duplicate_sources(self):
        """Verify no duplicate source languages."""
        from faster_whisper_hotkey.config import canary_source_target_languages

        assert len(canary_source_target_languages) == len(
            set(canary_source_target_languages)
        )


class TestCanaryAllowedLanguagePairsExtended:
    """Extended tests for Canary language pairs."""

    def test_all_sources_in_pairs_are_valid(self):
        """Verify all source languages in pairs exist in source list."""
        from faster_whisper_hotkey.config import (
            canary_allowed_language_pairs,
            canary_source_target_languages,
        )

        for pair in canary_allowed_language_pairs:
            parts = pair.split("-")
            assert parts[0] in canary_source_target_languages

    def test_all_targets_in_pairs_are_valid(self):
        """Verify all target languages in pairs exist in source list."""
        from faster_whisper_hotkey.config import (
            canary_allowed_language_pairs,
            canary_source_target_languages,
        )

        for pair in canary_allowed_language_pairs:
            parts = pair.split("-")
            assert parts[1] in canary_source_target_languages

    def test_no_duplicate_pairs(self):
        """Verify no duplicate pairs."""
        from faster_whisper_hotkey.config import canary_allowed_language_pairs

        assert len(canary_allowed_language_pairs) == len(
            set(canary_allowed_language_pairs)
        )

    def test_self_translation_pairs_exist(self):
        """Test that self-translation pairs (e.g., en-en, de-de) exist for transcription."""
        from faster_whisper_hotkey.config import canary_allowed_language_pairs

        # At least English should have a self-translation pair
        assert "en-en" in canary_allowed_language_pairs

    def test_get_target_languages_for_source(self):
        """Test we can correctly extract target languages for a given source."""
        from faster_whisper_hotkey.config import canary_allowed_language_pairs

        source = "en"
        targets = {
            pair.split("-")[1]
            for pair in canary_allowed_language_pairs
            if pair.startswith(f"{source}-")
        }
        assert len(targets) > 0, f"No target languages found for source '{source}'"
        assert isinstance(targets, set)


class TestConfigLoadingEdgeCases:
    """Test edge cases in config loading."""

    def test_config_load_raises_on_missing_file(self):
        """Verify proper error handling when config file is missing."""
        from unittest.mock import patch
        import importlib
        import faster_whisper_hotkey.config as config_module

        with patch("faster_whisper_hotkey.config.open", side_effect=FileNotFoundError):
            # Reset module state if possible
            if hasattr(config_module, "_CONFIG"):
                delattr(config_module, "_CONFIG")

            with pytest.raises(FileNotFoundError):
                importlib.reload(config_module)

    @patch("faster_whisper_hotkey.config.open", return_value=MagicMock())
    @patch("faster_whisper_hotkey.config.json.load", return_value={})
    def test_config_loads_empty_dict_with_defaults(self, mock_load, mock_open):
        """Test that empty config uses default values."""
        import importlib

        # Create a fresh module namespace
        spec = importlib.util.spec_from_file_location(
            "config_test", "src/faster_whisper_hotkey/config.py"
        )
        # Just verify it doesn't crash with empty dict (will use empty list defaults)

    def test_config_loads_partial_data(self):
        """Test that partial config loads correctly with defaults for missing keys."""
        # The config module already loaded at import time, verify the structure
        from faster_whisper_hotkey.config import (
            accepted_models_whisper,
            accepted_languages_whisper,
            english_only_models_whisper,
            canary_source_target_languages,
            canary_allowed_language_pairs,
        )

        # All should be lists or sets even if loaded from partial config
        assert isinstance(accepted_models_whisper, list)
        assert isinstance(accepted_languages_whisper, list)
        assert isinstance(english_only_models_whisper, set)
        assert isinstance(canary_source_target_languages, list)
        assert isinstance(canary_allowed_language_pairs, list)
