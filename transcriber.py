"""
ClearSpeak - Transcription and Model Helpers
Handles faster-whisper model loading, audio processing, segment extraction,
timestamp formatting, and safe temporary file lifecycle management.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union


# Supported Whisper model identifiers and metadata
MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "tiny": {
        "parameters": "39M",
        "vram": "~1 GB",
        "relative_speed": "~32x",
        "disk_size": "~75 MB",
        "description": "Ultra-fast with minimal memory consumption. Ideal for quick drafts and low-spec systems.",
    },
    "base": {
        "parameters": "74M",
        "vram": "~1 GB",
        "relative_speed": "~16x",
        "disk_size": "~145 MB",
        "description": "Fast and responsive for everyday standard speech, podcasts, and clear audio.",
    },
    "small": {
        "parameters": "244M",
        "vram": "~2 GB",
        "relative_speed": "~6x",
        "disk_size": "~480 MB",
        "description": "Recommended default. Strikes a great balance between accuracy, multi-language support, and speed.",
    },
    "medium": {
        "parameters": "769M",
        "vram": "~5 GB",
        "relative_speed": "~2x",
        "disk_size": "~1.5 GB",
        "description": "High accuracy for accented speakers, multi-speaker meetings, and technical vocabulary.",
    },
    "large-v3": {
        "parameters": "1550M",
        "vram": "~10 GB",
        "relative_speed": "~1x",
        "disk_size": "~3.1 GB",
        "description": "Maximum accuracy and state-of-the-art multilingual recognition. Most computationally intensive.",
    },
}

# Common audio formats supported by faster-whisper (via ffmpeg backend)
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".mpeg",
    ".webm",
    ".ogg",
    ".flac",
}

# Major languages mapping (code -> Display Name)
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "auto": "Automatic Language Detection",
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "ja": "Japanese (日本語)",
    "pt": "Portuguese (Português)",
    "ar": "Arabic (العربية)",
    "zh": "Chinese (中文)",
    "ru": "Russian (Русский)",
    "it": "Italian (Italiano)",
    "ko": "Korean (한국어)",
    "nl": "Dutch (Nederlands)",
    "tr": "Turkish (Türkçe)",
    "pl": "Polish (Polski)",
    "sv": "Swedish (Svenska)",
    "uk": "Ukrainian (Українська)",
}


@dataclass
class SegmentItem:
    """Represents a single transcribed segment."""
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "duration": round(self.duration, 2),
            "avg_logprob": round(self.avg_logprob, 3),
            "no_speech_prob": round(self.no_speech_prob, 3),
        }


@dataclass
class TranscriptionResult:
    """Structured container for full transcription output."""
    full_text: str
    segments: List[SegmentItem] = field(default_factory=list)
    detected_language: str = "en"
    language_probability: float = 1.0
    duration: float = 0.0
    model_name: str = "base"
    is_empty_or_silent: bool = False
    warning_message: Optional[str] = None


def is_supported_audio(filename_or_path: Union[str, Path]) -> bool:
    """Check if the given filename or path has a supported audio extension."""
    suffix = Path(filename_or_path).suffix.lower()
    return suffix in SUPPORTED_AUDIO_EXTENSIONS


def detect_best_device() -> Tuple[str, str]:
    """
    Detect the recommended device and compute type.
    Defaults to CUDA if torch.cuda.is_available(), otherwise CPU with int8 quantization.
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass

    # CPU default: int8 quantization provides optimal balance of speed and low memory
    return "cpu", "int8"


def load_whisper_model(
    model_size: str = "small",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    cpu_threads: int = 4,
    download_root: Optional[str] = None,
) -> Any:
    """
    Load a faster-whisper WhisperModel instance.
    Raises RuntimeError if faster_whisper is not installed.
    """
    if model_size not in MODEL_SPECS:
        valid_models = ", ".join(MODEL_SPECS.keys())
        raise ValueError(f"Invalid model size '{model_size}'. Choose from: {valid_models}")

    if device is None or compute_type is None:
        auto_device, auto_compute = detect_best_device()
        device = device or auto_device
        compute_type = compute_type or auto_compute

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Please install it using: "
            "`pip install faster-whisper`"
        ) from exc

    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        download_root=download_root,
    )


def format_timestamp(seconds: float, format_style: str = "standard") -> str:
    """
    Format a floating point duration in seconds into a human-readable timestamp.

    Styles:
    - 'standard': HH:MM:SS.mmm (e.g. 00:01:23.450)
    - 'short': MM:SS or HH:MM:SS if >= 1 hour (e.g. 01:23)
    - 'srt': HH:MM:SS,mmm (comma separator per SubRip standard)
    - 'vtt': HH:MM:SS.mmm (standard WebVTT)
    """
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    # Catch rollover in milliseconds rounding
    if millis >= 1000:
        millis -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1

    if format_style == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    elif format_style == "short":
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    else:  # 'standard' and 'vtt'
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def build_transcript_text(
    segments: List[Union[SegmentItem, Dict[str, Any]]],
    include_timestamps: bool = False,
    timestamp_style: str = "standard",
) -> str:
    """
    Assemble a list of segment objects or dictionaries into a formatted transcript string.
    """
    if not segments:
        return ""

    lines: List[str] = []
    for item in segments:
        if isinstance(item, SegmentItem):
            start = item.start
            end = item.end
            text = item.text.strip()
        elif isinstance(item, dict):
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
            text = str(item.get("text", "")).strip()
        else:
            continue

        if not text:
            continue

        if include_timestamps:
            t_start = format_timestamp(start, format_style=timestamp_style)
            t_end = format_timestamp(end, format_style=timestamp_style)
            lines.append(f"[{t_start} -> {t_end}] {text}")
        else:
            lines.append(text)

    if include_timestamps:
        return "\n".join(lines)
    return " ".join(lines)


@contextmanager
def temp_audio_file(
    file_bytes_or_buffer: Union[bytes, Any],
    file_name: Optional[str] = None,
) -> Generator[Path, None, None]:
    """
    Context manager that safely writes an uploaded file to a temporary file on disk,
    yields the Path, and guarantees deletion when the block completes or an error occurs.
    """
    suffix = Path(file_name).suffix.lower() if file_name else ".wav"
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        suffix = ".wav"

    temp_fd, temp_path_str = tempfile.mkstemp(prefix="clearspeak_", suffix=suffix)
    temp_path = Path(temp_path_str)

    try:
        with os.fdopen(temp_fd, "wb") as temp_file:
            if isinstance(file_bytes_or_buffer, bytes):
                temp_file.write(file_bytes_or_buffer)
            elif hasattr(file_bytes_or_buffer, "read"):
                # Handle Streamlit UploadedFile or file-like buffer
                file_bytes_or_buffer.seek(0)
                shutil.copyfileobj(file_bytes_or_buffer, temp_file)
            else:
                raise TypeError("Input must be bytes or a file-like object with a .read() method")

        yield temp_path
    finally:
        # Guaranteed cleanup of temporary audio file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def transcribe_audio(
    model: Any,
    audio_path: Union[str, Path],
    language: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    model_name: str = "custom",
) -> TranscriptionResult:
    """
    Transcribe the specified audio file using a loaded faster-whisper model.
    Handles silent audio, empty outputs, and exceptions gracefully.
    """
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    if audio_path_obj.stat().st_size == 0:
        return TranscriptionResult(
            full_text="",
            segments=[],
            detected_language="unknown",
            language_probability=0.0,
            duration=0.0,
            model_name=model_name,
            is_empty_or_silent=True,
            warning_message="The uploaded audio file is empty (0 bytes).",
        )

    # Normalize language: "auto" or None indicates automatic detection
    lang_param: Optional[str] = None
    if language and language.lower() not in ("auto", "none", ""):
        lang_param = language.lower()

    try:
        raw_segments_generator, info = model.transcribe(
            str(audio_path_obj),
            language=lang_param,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500) if vad_filter else None,
        )

        segments: List[SegmentItem] = []
        for raw in raw_segments_generator:
            clean_text = raw.text.strip()
            if clean_text:
                segments.append(
                    SegmentItem(
                        start=raw.start,
                        end=raw.end,
                        text=clean_text,
                        avg_logprob=getattr(raw, "avg_logprob", 0.0),
                        no_speech_prob=getattr(raw, "no_speech_prob", 0.0),
                    )
                )

        det_lang = getattr(info, "language", lang_param or "en")
        det_lang_prob = getattr(info, "language_probability", 1.0)
        total_duration = getattr(info, "duration", 0.0)
        if total_duration == 0.0 and segments:
            total_duration = segments[-1].end

        full_text = build_transcript_text(segments, include_timestamps=False)
        is_silent = len(segments) == 0 or len(full_text.strip()) == 0

        warning: Optional[str] = None
        if is_silent:
            warning = "No speech was detected in the audio file. It may be silent or only contain background noise."

        return TranscriptionResult(
            full_text=full_text,
            segments=segments,
            detected_language=det_lang,
            language_probability=det_lang_prob,
            duration=total_duration,
            model_name=model_name,
            is_empty_or_silent=is_silent,
            warning_message=warning,
        )

    except Exception as exc:
        err_msg = str(exc)
        # Check for common ffmpeg installation issue
        if "ffmpeg" in err_msg.lower() or "ffprobe" in err_msg.lower():
            raise RuntimeError(
                "ffmpeg is required for audio decoding but was not found. "
                "Please install ffmpeg on your system and ensure it is on your PATH."
            ) from exc
        raise
