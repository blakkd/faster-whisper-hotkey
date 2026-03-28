from unittest.mock import MagicMock, patch

import pytest

from faster_whisper_hotkey.clipboard import (
    backup_clipboard,
    restore_clipboard,
    set_clipboard,
)


class TestBackupClipboard:
    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_backup_clipboard_success(self, mock_pyperclip):
        mock_pyperclip.paste.return_value = "original text"

        result = backup_clipboard()

        assert result == "original text"
        mock_pyperclip.paste.assert_called_once()

    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_backup_clipboard_empty(self, mock_pyperclip):
        mock_pyperclip.paste.return_value = ""

        result = backup_clipboard()

        assert result == ""

    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_backup_clipboard_exception(self, mock_pyperclip):
        mock_pyperclip.paste.side_effect = Exception("Clipboard error")

        result = backup_clipboard()

        assert result is None

    @patch("faster_whisper_hotkey.clipboard.pyperclip", None)
    def test_backup_clipboard_no_pyperclip(self):
        result = backup_clipboard()

        assert result is None


class TestSetClipboard:
    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_set_clipboard_success(self, mock_pyperclip):
        result = set_clipboard("test text")

        assert result is True
        mock_pyperclip.copy.assert_called_once_with("test text")

    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_set_clipboard_empty_string(self, mock_pyperclip):
        result = set_clipboard("")

        assert result is True
        mock_pyperclip.copy.assert_called_once_with("")

    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_set_clipboard_exception(self, mock_pyperclip):
        mock_pyperclip.copy.side_effect = Exception("Clipboard error")

        result = set_clipboard("test text")

        assert result is False

    @patch("faster_whisper_hotkey.clipboard.pyperclip", None)
    def test_set_clipboard_no_pyperclip(self):
        result = set_clipboard("test text")

        assert result is False


class TestRestoreClipboard:
    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_restore_clipboard_success(self, mock_pyperclip):
        restore_clipboard("original text")

        mock_pyperclip.copy.assert_called_once_with("original text")

    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_restore_clipboard_none(self, mock_pyperclip):
        restore_clipboard(None)

        mock_pyperclip.copy.assert_not_called()

    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_restore_clipboard_empty_string(self, mock_pyperclip):
        restore_clipboard("")

        mock_pyperclip.copy.assert_called_once_with("")

    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_restore_clipboard_exception(self, mock_pyperclip):
        mock_pyperclip.copy.side_effect = Exception("Clipboard error")

        restore_clipboard("original text")

        mock_pyperclip.copy.assert_called_once()

    @patch("faster_whisper_hotkey.clipboard.pyperclip", None)
    def test_restore_clipboard_no_pyperclip(self):
        restore_clipboard("test text")


class TestClipboardWorkflow:
    @patch("faster_whisper_hotkey.clipboard.pyperclip")
    def test_full_workflow(self, mock_pyperclip):
        original = "original clipboard content"
        new_content = "transcribed text"

        mock_pyperclip.paste.return_value = original

        backed_up = backup_clipboard()
        assert backed_up == original

        set_result = set_clipboard(new_content)
        assert set_result is True

        restore_clipboard(backed_up)

        calls = mock_pyperclip.copy.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == new_content
        assert calls[1][0][0] == original
