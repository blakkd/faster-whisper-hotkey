"""Tests for LLMCorrector class in llm_corrector.py."""

import requests
import pytest
from unittest.mock import MagicMock, patch
from faster_whisper_hotkey.llm_corrector import LLMCorrector


class TestLLMCorrectorInit:
    """Tests for LLMCorrector initialization."""

    def test_init_strips_trailing_slash(self):
        """Endpoint trailing slash should be stripped before appending /chat/completions."""
        corrector = LLMCorrector("http://localhost:8080/", "test-model")
        assert corrector.endpoint == "http://localhost:8080/chat/completions"

    def test_init_no_trailing_slash(self):
        """Endpoint without trailing slash should still get /chat/completions appended."""
        corrector = LLMCorrector("http://localhost:8080", "test-model")
        assert corrector.endpoint == "http://localhost:8080/chat/completions"

    def test_init_multiple_trailing_slashes(self):
        """Multiple trailing slashes should all be stripped."""
        corrector = LLMCorrector("http://localhost:8080///", "test-model")
        assert corrector.endpoint == "http://localhost:8080/chat/completions"

    def test_init_sets_model_name(self):
        """Model name should be stored as-is."""
        corrector = LLMCorrector("http://localhost:8080", "my-special-model-v2")
        assert corrector.model_name == "my-special-model-v2"


class TestLLMCorrectorCorrectEmptyInput:
    """Tests for empty/whitespace input handling."""

    def test_correct_empty_string(self):
        """Empty string should return immediately without API call."""
        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("")
        assert result == ""

    def test_correct_whitespace_only(self):
        """Whitespace-only string should return immediately without API call."""
        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("   \n\t  ")
        assert result == "   \n\t  "

    def test_correct_newline_only(self):
        """Newline-only string should return immediately."""
        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("\n\n\n")
        assert result == "\n\n\n"


class TestLLMCorrectorSuccessfulCorrection:
    """Tests for successful LLM correction scenarios."""

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_success_returns_corrected_text(self, mock_post):
        """Successful API response should return corrected text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is corrected text."}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("This is wrong text.")

        assert result == "This is corrected text."

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_unchanged_text(self, mock_post):
        """If LLM returns same text, it should be returned unchanged."""
        original = "This is already perfect."
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": original}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_strips_whitespace_from_result(self, mock_post):
        """LLM response with surrounding whitespace should be stripped."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "  Corrected text with spaces  \n"}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("Original text")

        assert result == "Corrected text with spaces"

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_strips_mismatched_quotes(self, mock_post):
        """Mismatched quote types at start/end should still be stripped."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        content_value = "\"This has mismatched quotes'"
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content_value}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("Original text")

        assert result == "This has mismatched quotes"

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_strips_single_quote_at_start_only(self, mock_post):
        """Quote only at start should be stripped."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        content_val = '"Just a quote at start'
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content_val}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("Original text")

        assert result == "Just a quote at start"

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_strips_single_quote_at_end_only(self, mock_post):
        """Quote only at end should be stripped."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        content_val = 'Just a quote at end"'
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content_val}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("Original text")

        assert result == "Just a quote at end"

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_whitespace_and_quotes_together(self, mock_post):
        """Should strip both whitespace and surrounding quotes."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '  "Corrected text"  '}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct("Original text")

        assert result == "Corrected text"

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_multisentence_input(self, mock_post):
        """Multi-sentence transcription should be handled correctly."""
        original = "Hello world this is a test. Can you hear me clearly? Thank you."
        corrected = "Hello world, this is a test. Can you hear me clearly? Thank you."

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": corrected}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        result = corrector.correct(original)

        assert result == corrected


class TestLLMCorrectorPayloadStructure:
    """Tests to verify the API payload structure."""

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_payload_contains_correct_model(self, mock_post):
        """Payload should contain the configured model name."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "gpt-4-turbo")
        corrector.correct("Test text")

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == "gpt-4-turbo"

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_payload_contains_user_prompt_with_transcription(self, mock_post):
        """User prompt should contain the transcription text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct("My transcribed speech")

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        messages = payload["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "My transcribed speech" in messages[0]["content"]

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_payload_has_temperature_setting(self, mock_post):
        """Payload should have temperature set to 0.6."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct("Test")

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0.6

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_payload_has_max_tokens_setting(self, mock_post):
        """Payload should have max_tokens set to 1024."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct("Test")

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["max_tokens"] == 1024

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_payload_has_chat_template_kwargs(self, mock_post):
        """Payload should disable thinking mode."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct("Test")

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_request_has_json_content_type_header(self, mock_post):
        """Request should include Content-Type header."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct("Test")

        call_args = mock_post.call_args
        headers = call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"


class TestLLMCorrectorErrorHandling:
    """Tests for error handling and fallback behavior."""

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_request_exception(self, mock_post):
        """Request exceptions should fall back to original text."""
        mock_post.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Original text that needs correction"
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_timeout(self, mock_post):
        """Timeout exceptions should fall back to original text."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Text being processed"
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_http_error(self, mock_post):
        """HTTP errors should fall back to original text."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Original text"
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_invalid_json_response(self, mock_post):
        """Invalid JSON responses should fall back to original text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Original text"
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_missing_choices_in_response(self, mock_post):
        """Response missing choices array should fall back to original text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Original text"
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_empty_choices_array(self, mock_post):
        """Empty choices array should fall back to original text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Original text"
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_missing_message_key(self, mock_post):
        """Response missing message key should fall back to original text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Original text"
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_empty_content(self, mock_post):
        """Empty content from LLM should fall back to original text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Original text"
        result = corrector.correct(original)

        assert result == original

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_correct_handles_general_exception(self, mock_post):
        """Any unexpected exception should fall back to original text."""
        mock_post.side_effect = Exception("Unexpected error")

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        original = "Original text"
        result = corrector.correct(original)

        assert result == original


class TestLLMCorrectorTimeoutSetting:
    """Tests for request timeout configuration."""

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    def test_request_uses_120_second_timeout(self, mock_post):
        """Request should use 120 second timeout."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct("Test")

        call_args = mock_post.call_args
        assert call_args.kwargs["timeout"] == 120


class TestLLMCorrectorLogging:
    """Tests for logging behavior."""

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    @patch("faster_whisper_hotkey.llm_corrector.logger")
    def test_logs_correction_when_text_changes(self, mock_logger, mock_post):
        """Should log when text is actually corrected."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Corrected version"}}]
        }
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct("Original version")

        mock_logger.info.assert_called_once_with(
            'LLM corrected text: "Corrected version"'
        )

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    @patch("faster_whisper_hotkey.llm_corrector.logger")
    def test_does_not_log_when_text_unchanged(self, mock_logger, mock_post):
        """Should not log info when text is returned unchanged by LLM."""
        text = "Same text"
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": text}}]}
        mock_post.return_value = mock_response

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct(text)

        mock_logger.debug.assert_called_once_with("LLM correction: no changes needed")
        mock_logger.info.assert_not_called()

    @patch("faster_whisper_hotkey.llm_corrector.requests.post")
    @patch("faster_whisper_hotkey.llm_corrector.logger")
    def test_logs_warning_on_failure(self, mock_logger, mock_post):
        """Should log warning when correction fails."""
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        corrector = LLMCorrector("http://localhost:8080", "test-model")
        corrector.correct("Test text")

        mock_logger.warning.assert_called_once()
        assert "LLM correction failed" in mock_logger.warning.call_args.args[0]
        assert "Network error" in mock_logger.warning.call_args.args[0]
