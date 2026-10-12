"""
Unit tests for ClearSpeak transcription logic, timestamp formatting,
transcript assembly, temporary file lifecycle, and error handling.
Uses a mock Whisper model to ensure zero external downloads and no audio hardware required.
"""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

from transcriber import (
    MODEL_SPECS,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_LANGUAGES,
    SegmentItem,
    TranscriptionResult,
    build_transcript_text,
    detect_best_device,
    format_timestamp,
    is_supported_audio,
    temp_audio_file,
    transcribe_audio,
)


class MockSegment(NamedTuple):
    """Mock for faster-whisper Segment object."""
    start: float
    end: float
    text: str
    avg_logprob: float = -0.15
    no_speech_prob: float = 0.02


class MockTranscriptionInfo:
    """Mock for faster-whisper TranscriptionInfo object."""
    def __init__(self, language: str = "en", language_probability: float = 0.98, duration: float = 12.5):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


class MockWhisperModel:
    """Mock Whisper model simulating faster_whisper.WhisperModel."""
    def __init__(self, segments: List[MockSegment] = None, info: MockTranscriptionInfo = None, raise_error: Exception = None):
        self.segments = segments if segments is not None else []
        self.info = info if info is not None else MockTranscriptionInfo()
        self.raise_error = raise_error
        self.transcribe_called = False

    def transcribe(self, audio: str, **kwargs) -> Any:
        self.transcribe_called = True
        if self.raise_error:
            raise self.raise_error
        # Return generator of segments and the info object
        return (iter(self.segments), self.info)


class TestTimestampFormatting(unittest.TestCase):
    """Tests for format_timestamp utility."""

    def test_zero_timestamp(self):
        self.assertEqual(format_timestamp(0.0), "00:00:00.000")

    def test_negative_timestamp(self):
        # Negative timestamps should clamp to 0
        self.assertEqual(format_timestamp(-5.2), "00:00:00.000")

    def test_seconds_and_subseconds(self):
        self.assertEqual(format_timestamp(12.345), "00:00:12.345")

    def test_minutes_and_seconds(self):
        self.assertEqual(format_timestamp(83.450), "00:01:23.450")

    def test_hour_boundary(self):
        # 1 hour, 2 minutes, 3 seconds, 450 ms
        total_sec = 3600 + (2 * 60) + 3 + 0.450
        self.assertEqual(format_timestamp(total_sec), "01:02:03.450")

    def test_srt_style_comma_separator(self):
        self.assertEqual(format_timestamp(83.450, format_style="srt"), "00:01:23,450")

    def test_short_style(self):
        # Under 1 hour
        self.assertEqual(format_timestamp(83.450, format_style="short"), "01:23")
        # Over 1 hour
        self.assertEqual(format_timestamp(3665.0, format_style="short"), "01:01:05")

    def test_millisecond_rounding_rollover(self):
        # 0.9999 seconds should roll over cleanly to 00:00:01.000
        result = format_timestamp(0.9999)
        self.assertEqual(result, "00:00:01.000")


class TestTranscriptAssembly(unittest.TestCase):
    """Tests for build_transcript_text utility."""

    def setUp(self):
        self.segments = [
            SegmentItem(start=0.0, end=2.5, text="Welcome to ClearSpeak."),
            SegmentItem(start=2.8, end=5.2, text="Your local speech recognition assistant."),
        ]

    def test_plain_assembly(self):
        text = build_transcript_text(self.segments, include_timestamps=False)
        expected = "Welcome to ClearSpeak. Your local speech recognition assistant."
        self.assertEqual(text, expected)

    def test_timestamped_assembly(self):
        text = build_transcript_text(self.segments, include_timestamps=True, timestamp_style="standard")
        expected_lines = [
            "[00:00:00.000 -> 00:00:02.500] Welcome to ClearSpeak.",
            "[00:00:02.800 -> 00:00:05.200] Your local speech recognition assistant.",
        ]
        self.assertEqual(text, "\n".join(expected_lines))

    def test_empty_segments(self):
        self.assertEqual(build_transcript_text([]), "")

    def test_assembly_from_dicts(self):
        dict_segments = [
            {"start": 1.0, "end": 3.0, "text": "Testing dictionary input."},
            {"start": 3.5, "end": 4.0, "text": ""},  # empty text should be skipped
        ]
        text = build_transcript_text(dict_segments, include_timestamps=False)
        self.assertEqual(text, "Testing dictionary input.")


class TestModelSpecsAndValidation(unittest.TestCase):
    """Tests for model configurations, supported languages, and file formats."""

    def test_all_required_models_present(self):
        required_models = ["tiny", "base", "small", "medium", "large-v3"]
        for model in required_models:
            self.assertIn(model, MODEL_SPECS)
            self.assertIn("vram", MODEL_SPECS[model])
            self.assertIn("relative_speed", MODEL_SPECS[model])
            self.assertIn("disk_size", MODEL_SPECS[model])

    def test_all_required_languages_present(self):
        required_languages = ["en", "hi", "es", "fr", "de", "ja", "pt", "ar", "zh"]
        for lang in required_languages:
            self.assertIn(lang, SUPPORTED_LANGUAGES)

    def test_supported_audio_formats(self):
        formats = [
            "speech.wav", "interview.mp3", "voice.m4a", "clip.mp4",
            "audio.mpeg", "call.webm", "recording.ogg", "music.flac"
        ]
        for name in formats:
            self.assertTrue(is_supported_audio(name), f"Expected {name} to be supported")

        unsupported = ["document.pdf", "image.png", "archive.zip", "video.avi"]
        for name in unsupported:
            self.assertFalse(is_supported_audio(name), f"Expected {name} to be unsupported")

    def test_detect_best_device(self):
        device, compute = detect_best_device()
        self.assertIn(device, ("cpu", "cuda"))
        self.assertIn(compute, ("int8", "float16", "float32"))


class TestTempAudioLifecycle(unittest.TestCase):
    """Tests for safe temporary audio file lifecycle and guaranteed cleanup."""

    def test_cleanup_after_normal_exit(self):
        created_path = None
        sample_content = b"RIFF....WAVEfmt ...."

        with temp_audio_file(sample_content, "test.wav") as temp_p:
            created_path = temp_p
            self.assertTrue(temp_p.exists(), "Temp audio file should exist inside context")
            self.assertEqual(temp_p.read_bytes(), sample_content)

        self.assertFalse(created_path.exists(), "Temp audio file must be deleted upon exit")

    def test_cleanup_after_exception(self):
        created_path = None
        sample_content = b"dummy audio content"

        with self.assertRaises(ValueError):
            with temp_audio_file(sample_content, "error_test.mp3") as temp_p:
                created_path = temp_p
                self.assertTrue(temp_p.exists())
                raise ValueError("Simulated processing crash inside context")

        self.assertIsNotNone(created_path)
        self.assertFalse(created_path.exists(), "Temp file must be removed even when exception occurs")


class TestTranscriptionWithMockModel(unittest.TestCase):
    """Tests for transcribe_audio using a mock model."""

    def test_successful_transcription(self):
        segments = [
            MockSegment(start=0.0, end=3.0, text="Hello world!"),
            MockSegment(start=3.5, end=6.0, text="ClearSpeak transcription test."),
        ]
        info = MockTranscriptionInfo(language="en", language_probability=0.99, duration=6.0)
        mock_model = MockWhisperModel(segments=segments, info=info)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"non-empty fake audio data")
            temp_path = f.name

        try:
            result = transcribe_audio(mock_model, temp_path, language="en", model_name="base")
            self.assertIsInstance(result, TranscriptionResult)
            self.assertEqual(result.detected_language, "en")
            self.assertEqual(result.full_text, "Hello world! ClearSpeak transcription test.")
            self.assertEqual(len(result.segments), 2)
            self.assertFalse(result.is_empty_or_silent)
            self.assertIsNone(result.warning_message)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_silent_or_no_speech_audio(self):
        # Empty segment generator simulating silent audio
        mock_model = MockWhisperModel(segments=[], info=MockTranscriptionInfo(duration=5.0))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"non-empty silent audio data")
            temp_path = f.name

        try:
            result = transcribe_audio(mock_model, temp_path, language="auto", model_name="tiny")
            self.assertTrue(result.is_empty_or_silent)
            self.assertEqual(result.full_text, "")
            self.assertEqual(len(result.segments), 0)
            self.assertIsNotNone(result.warning_message)
            self.assertIn("No speech was detected", result.warning_message)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_zero_byte_empty_file(self):
        mock_model = MockWhisperModel()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name  # 0 bytes

        try:
            result = transcribe_audio(mock_model, temp_path)
            self.assertTrue(result.is_empty_or_silent)
            self.assertEqual(result.warning_message, "The uploaded audio file is empty (0 bytes).")
            # Model should NOT be invoked for a 0-byte file
            self.assertFalse(mock_model.transcribe_called)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_nonexistent_file_raises_error(self):
        mock_model = MockWhisperModel()
        with self.assertRaises(FileNotFoundError):
            transcribe_audio(mock_model, "non_existent_audio_file.wav")

    def test_ffmpeg_missing_error_handling(self):
        mock_model = MockWhisperModel(raise_error=RuntimeError("ffmpeg executable not found in PATH"))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"dummy")
            temp_path = f.name

        try:
            with self.assertRaises(RuntimeError) as cm:
                transcribe_audio(mock_model, temp_path)
            self.assertIn("ffmpeg is required", str(cm.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
