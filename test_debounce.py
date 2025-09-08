#!/usr/bin/env python3
"""
Test script to verify the debounce feature in the transcriber.
This script simulates short recordings to ensure they are skipped.
"""

import sys
import os
import time
import numpy as np
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from faster_whisper_hotkey.transcriber import MicrophoneTranscriber
from faster_whisper_hotkey.settings import Settings

def test_debounce_feature():
    """Test that recordings shorter than 1 second are skipped."""

    # Create mock settings
    settings = Settings(
        device_name="test_device",
        model_type="whisper",
        model_name="tiny",
        compute_type="int8",
        device="cpu",
        language="en"
    )

    # Create transcriber instance
    transcriber = MicrophoneTranscriber(settings)

    # Mock the model wrapper's transcribe method to avoid actual transcription
    transcriber.model_wrapper.transcribe = MagicMock(return_value="test transcription")

    # Mock the clipboard functions to avoid actual clipboard operations
    transcriber.set_clipboard = MagicMock(return_value=True)
    transcriber.backup_clipboard = MagicMock(return_value="backup text")
    transcriber.restore_clipboard = MagicMock()
    transcriber.paste_to_active_window = MagicMock()

    # Mock sounddevice InputStream
    with patch('sounddevice.InputStream') as mock_input_stream:
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        # Test 1: Short recording (0.5 seconds) - should be skipped
        print("Test 1: Short recording (0.5s) - should be skipped")
        transcriber.start_recording()
        time.sleep(0.5)  # Simulate 0.5 second recording
        transcriber.stop_recording_and_transcribe()

        # Verify transcription was not processed
        assert not transcriber.model_wrapper.transcribe.called, "Short recording should not trigger transcription"
        print("✓ Short recording correctly skipped")

        # Reset for next test
        transcriber.model_wrapper.transcribe.reset_mock()
        transcriber.transcription_queue.clear()

        # Test 2: Long recording (1.5 seconds) - should be processed
        print("Test 2: Long recording (1.5s) - should be processed")
        transcriber.start_recording()
        time.sleep(1.5)  # Simulate 1.5 second recording
        transcriber.stop_recording_and_transcribe()

        # Verify transcription was processed
        assert transcriber.model_wrapper.transcribe.called, "Long recording should trigger transcription"
        print("✓ Long recording correctly processed")

        print("\nAll tests passed! Debounce feature is working correctly.")

if __name__ == "__main__":
    test_debounce_feature()