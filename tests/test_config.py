import json
from unittest.mock import MagicMock, patch

import pytest

from faster_whisper_hotkey.config import (
    accepted_languages_whisper,
    accepted_models_whisper,
    canary_allowed_language_pairs,
    canary_source_target_languages,
    english_only_models_whisper,
    get_resource_path,
)


class TestGetResourcePath:
    def test_get_resource_path_returns_string(self):
        path = get_resource_path("test.json")
        assert isinstance(path, str)
        assert "test.json" in path

    def test_get_resource_path_includes_package_name(self):
        path = get_resource_path("test.json")
        assert "faster_whisper_hotkey" in path


class TestAcceptedModelsWhisper:
    def test_accepted_models_is_list(self):
        assert isinstance(accepted_models_whisper, list)

    def test_accepted_models_not_empty(self):
        assert len(accepted_models_whisper) > 0

    def test_accepted_models_contains_expected_models(self):
        expected_models = ["tiny", "base", "small", "medium", "large-v3"]
        for model in expected_models:
            assert model in accepted_models_whisper

    def test_distil_models_present(self):
        distil_models = [m for m in accepted_models_whisper if "distil" in m]
        assert len(distil_models) >= 2


class TestEnglishOnlyModelsWhisper:
    def test_english_only_models_is_set(self):
        assert isinstance(english_only_models_whisper, set)

    def test_english_only_models_contains_en_suffix(self):
        for model in english_only_models_whisper:
            if "distil-large" not in model:
                assert model.endswith(".en"), (
                    f"{model} should end with .en or be a distil-large-vX model"
                )

    def test_english_only_models_not_empty(self):
        assert len(english_only_models_whisper) > 0


class TestAcceptedLanguagesWhisper:
    def test_accepted_languages_is_list(self):
        assert isinstance(accepted_languages_whisper, list)

    def test_accepted_languages_contains_auto(self):
        assert "auto" in accepted_languages_whisper

    def test_accepted_languages_contains_common_languages(self):
        common_langs = ["en", "de", "fr", "es", "it", "pt", "zh", "ja", "ko"]
        for lang in common_langs:
            assert lang in accepted_languages_whisper

    def test_accepted_languages_not_empty(self):
        assert len(accepted_languages_whisper) > 0


class TestCanaryLanguages:
    def test_canary_source_target_languages_is_list(self):
        assert isinstance(canary_source_target_languages, list)

    def test_canary_source_target_languages_not_empty(self):
        assert len(canary_source_target_languages) > 0

    def test_canary_contains_common_languages(self):
        common_langs = ["en", "de", "fr", "es"]
        for lang in common_langs:
            assert lang in canary_source_target_languages


class TestCanaryAllowedLanguagePairs:
    def test_canary_allowed_pairs_is_list(self):
        assert isinstance(canary_allowed_language_pairs, list)

    def test_canary_allowed_pairs_not_empty(self):
        assert len(canary_allowed_language_pairs) > 0

    def test_pair_format(self):
        for pair in canary_allowed_language_pairs:
            parts = pair.split("-")
            assert len(parts) == 2, f"Pair {pair} should be in format 'lang-lang'"
            source, target = parts
            assert isinstance(source, str) and len(source) == 2
            assert isinstance(target, str) and len(target) == 2

    def test_en_en_pair_exists(self):
        assert "en-en" in canary_allowed_language_pairs


class TestConfigErrors:
    @patch("faster_whisper_hotkey.config.open", side_effect=FileNotFoundError)
    def test_missing_config_file_raises_error(self, mock_open):
        import importlib
        import faster_whisper_hotkey.config as config_module

        with pytest.raises(FileNotFoundError):
            setattr(config_module, "_CONFIG", None)
            importlib.reload(config_module)

    @patch(
        "faster_whisper_hotkey.config.json.load",
        side_effect=json.JSONDecodeError("doc", "", 0),
    )
    def test_invalid_json_raises_error(self, mock_load):
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        with patch("faster_whisper_hotkey.config.open", return_value=mock_file):
            import importlib
            import faster_whisper_hotkey.config as config_module

            with pytest.raises(json.JSONDecodeError):
                setattr(config_module, "_CONFIG", None)
                importlib.reload(config_module)
