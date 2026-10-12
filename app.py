"""
ClearSpeak - Automatic Speech Recognition (ASR) Web Application
A polished Streamlit interface for private, local audio transcription using faster-whisper.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from transcriber import (
    MODEL_SPECS,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_LANGUAGES,
    TranscriptionResult,
    build_transcript_text,
    detect_best_device,
    format_timestamp,
    is_supported_audio,
    load_whisper_model,
    temp_audio_file,
    transcribe_audio,
)

# ---------------------------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClearSpeak | Local Speech-to-Text ASR",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Modern, Premium Aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography & Layout */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0px;
    }

    .badge-privacy {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 12px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }

    /* Segment Card */
    .segment-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .segment-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .timestamp-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #818cf8;
        background: rgba(99, 102, 241, 0.15);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .segment-text {
        font-size: 0.96rem;
        color: #e2e8f0;
        line-height: 1.5;
    }

    /* File Info Box */
    .file-pill {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 10px 16px;
        border-radius: 10px;
        margin-top: 8px;
        margin-bottom: 14px;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_whisper_model(
    model_size: str,
    device: str,
    compute_type: str,
) -> Any:
    """
    Load and cache a WhisperModel instance for the entire session.
    Avoids reloading weights across multiple file transcriptions.
    """
    return load_whisper_model(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )


# ---------------------------------------------------------------------------
# Sidebar: Settings & Controls
# ---------------------------------------------------------------------------
def render_sidebar() -> dict[str, Any]:
    """Render the sidebar with model options, language selections, and hardware configs."""
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")

        # 1. Whisper Model Selector
        model_options = list(MODEL_SPECS.keys())
        default_index = model_options.index("small") if "small" in model_options else 0

        selected_model = st.selectbox(
            "Whisper Model Tier",
            options=model_options,
            index=default_index,
            help="Choose the model tier. Larger models offer higher transcription accuracy at the cost of speed and memory.",
        )

        current_spec = MODEL_SPECS[selected_model]
        st.caption(
            f"⚡ **Speed**: {current_spec['relative_speed']} | "
            f"💾 **Disk**: {current_spec['disk_size']} | "
            f"🧠 **RAM**: {current_spec['vram']}"
        )

        # Model Trade-off Guide Expander
        with st.expander("📊 Model Comparison Guide", expanded=False):
            st.markdown(
                """
                | Model | Speed | Memory | Accuracy | Best For |
                | :--- | :--- | :--- | :--- | :--- |
                | **tiny** | ~32x | ~1 GB | Basic | Rapid drafts |
                | **base** | ~16x | ~1 GB | Good | Everyday clear speech |
                | **small** | ~6x | ~2 GB | Very High | **Recommended default** |
                | **medium** | ~2x | ~5 GB | Excellent | Accents & meetings |
                | **large-v3** | ~1x | ~10 GB | State-of-the-art | Complex & multi-lingual |
                """
            )
            st.info(
                "💡 **First Run Note**: Model weights are downloaded automatically "
                "from Hugging Face the first time you select a tier, then saved locally.",
                icon="ℹ️",
            )

        st.divider()

        # 2. Language Selection
        st.markdown("### 🌐 Language")
        language_keys = list(SUPPORTED_LANGUAGES.keys())
        language_labels = [SUPPORTED_LANGUAGES[k] for k in language_keys]

        selected_language_idx = st.selectbox(
            "Spoken Language",
            options=range(len(language_keys)),
            format_func=lambda i: language_labels[i],
            index=0,
            help="Select 'Automatic Language Detection' to let Whisper identify the audio language automatically.",
        )
        chosen_lang_code = language_keys[selected_language_idx]

        st.divider()

        # 3. Hardware & Advanced Configuration
        with st.expander("🛠️ Advanced Options", expanded=False):
            auto_device, auto_compute = detect_best_device()

            device_override = st.selectbox(
                "Inference Device",
                options=["auto", "cpu", "cuda"],
                index=0,
                help=f"Detected default: {auto_device.upper()}",
            )
            actual_device = auto_device if device_override == "auto" else device_override

            compute_options = ["int8", "float16", "float32"] if actual_device == "cuda" else ["int8", "float32"]
            chosen_compute = st.selectbox(
                "Compute Type",
                options=compute_options,
                index=0,
                help="int8 uses less RAM/VRAM with virtually identical accuracy.",
            )

            vad_filter = st.checkbox(
                "Enable Voice Activity Detection (VAD)",
                value=True,
                help="Filters out silent pauses to improve speed and prevent hallucinations.",
            )

            beam_size = st.slider(
                "Beam Size",
                min_value=1,
                max_value=10,
                value=5,
                help="Higher beam size gives slightly better transcripts but takes slightly longer.",
            )

            timestamp_format = st.selectbox(
                "Timestamp Style",
                options=["standard", "short", "srt"],
                index=0,
                help="Formatting style for timestamps in the export and segment view.",
            )

        st.markdown("---")
        st.caption("ClearSpeak v1.0 • Powered by faster-whisper & CTranslate2")

    return {
        "model_size": selected_model,
        "language_code": chosen_lang_code,
        "device": actual_device,
        "compute_type": chosen_compute,
        "vad_filter": vad_filter,
        "beam_size": beam_size,
        "timestamp_format": timestamp_format,
    }


# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Hero Header
    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">🎙️ ClearSpeak</h1>
            <p class="hero-subtitle">High-precision, local Automatic Speech Recognition powered by faster-whisper.</p>
            <div class="badge-privacy">
                🔒 100% Local & Private — Audio never leaves your computer
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Get User Configurations from Sidebar
    config = render_sidebar()

    # 3. Audio File Uploader
    st.markdown("### 1. Upload Audio Recording")
    supported_extensions_list = [ext.lstrip(".") for ext in sorted(SUPPORTED_AUDIO_EXTENSIONS)]

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=supported_extensions_list,
        help=f"Supported audio formats: {', '.join(supported_extensions_list).upper()}",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        # Guidance state when no audio file is uploaded
        st.info(
            "👋 **Welcome!** Upload an audio recording above to begin.\n\n"
            "- **Supported Formats**: WAV, MP3, M4A, MP4, MPEG, WebM, OGG, FLAC.\n"
            "- **Privacy**: Audio processing runs entirely on your local CPU or GPU.\n"
            "- **Recommended**: Use **small** model for the best balance of speed and clarity.",
            icon="✨",
        )
        return

    # File information pill
    file_size_mb = uploaded_file.size / (1024 * 1024)
    file_suffix = Path(uploaded_file.name).suffix.upper()
    st.markdown(
        f"""
        <div class="file-pill">
            <span>🎵 <strong>{uploaded_file.name}</strong></span>
            <span>•</span>
            <span>Format: <code>{file_suffix}</code></span>
            <span>•</span>
            <span>Size: <code>{file_size_mb:.2f} MB</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Audio Player Preview
    with st.expander("🎧 Audio Player Preview", expanded=True):
        st.audio(uploaded_file)

    st.markdown("### 2. Run Transcription")

    # Action Button
    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        transcribe_button = st.button("🚀 Transcribe Audio", type="primary", use_container_width=True)
    with col_hint:
        lang_display = SUPPORTED_LANGUAGES.get(config["language_code"], "Auto-Detect")
        st.markdown(
            f"<div style='padding-top: 8px; color: #94a3b8; font-size: 0.9rem;'>"
            f"Configured: <strong>{config['model_size']}</strong> model | Language: <strong>{lang_display}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Session State Storage for Results
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "result_filename" not in st.session_state:
        st.session_state.result_filename = ""

    # Execute Transcription when clicked
    if transcribe_button:
        if uploaded_file.size == 0:
            st.error("❌ The uploaded file is empty (0 bytes). Please upload a valid recording.")
            return

        progress_placeholder = st.empty()
        start_time = time.time()

        try:
            # Step A: Load model (cached)
            with progress_placeholder.container():
                st.status("⏳ Loading Whisper model into memory...", expanded=True)

            model = get_cached_whisper_model(
                model_size=config["model_size"],
                device=config["device"],
                compute_type=config["compute_type"],
            )

            # Step B: Write temporarily and transcribe
            with progress_placeholder.container():
                with st.spinner("🎙️ Transcribing audio locally... This may take a few moments depending on audio length."):
                    with temp_audio_file(uploaded_file, file_name=uploaded_file.name) as temp_path:
                        result = transcribe_audio(
                            model=model,
                            audio_path=temp_path,
                            language=config["language_code"],
                            beam_size=config["beam_size"],
                            vad_filter=config["vad_filter"],
                            model_name=config["model_size"],
                        )

            elapsed_time = time.time() - start_time
            progress_placeholder.empty()

            # Store result in session state
            st.session_state.latest_result = result
            st.session_state.result_filename = uploaded_file.name
            st.session_state.elapsed_time = elapsed_time

            st.success(f"✅ Transcription complete in {elapsed_time:.2f}s!")

        except RuntimeError as r_err:
            progress_placeholder.empty()
            st.error(f"❌ **System Error**: {r_err}")
            st.info(
                "💡 **Troubleshooting ffmpeg**: ClearSpeak uses ffmpeg for decoding audio. "
                "Ensure ffmpeg is installed and available in your system PATH.",
                icon="🔧",
            )
            return
        except Exception as exc:
            progress_placeholder.empty()
            st.error(f"❌ **An unexpected error occurred during transcription**: {exc}")
            return

    # 5. Render Transcription Results
    result: Optional[TranscriptionResult] = st.session_state.latest_result
    if result is not None:
        st.markdown("### 3. Transcription Results")

        if result.warning_message:
            st.warning(f"⚠️ {result.warning_message}")

        # Metrics Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        lang_label = SUPPORTED_LANGUAGES.get(result.detected_language, result.detected_language.upper())
        prob_pct = f"{int(result.language_probability * 100)}%" if result.language_probability else "N/A"
        duration_formatted = format_timestamp(result.duration, format_style="short")
        total_words = len(result.full_text.split()) if result.full_text else 0

        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Detected Language</div>
                    <div class="metric-value">{lang_label}</div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">Confidence: {prob_pct}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Audio Duration</div>
                    <div class="metric-value">{duration_formatted}</div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">{result.duration:.1f} seconds</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Words</div>
                    <div class="metric-value">{total_words}</div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">{len(result.segments)} segments</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Model Used</div>
                    <div class="metric-value">{result.model_name.capitalize()}</div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">Device: {config['device'].upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Tabs for result presentation
        tab_plain, tab_timed, tab_table = st.tabs([
            "📄 Clean Transcript",
            "⏱️ Timestamped Segments",
            "🔍 Segment Quality Inspector",
        ])

        # TAB 1: Clean Transcript
        with tab_plain:
            if not result.full_text:
                st.info("No speech text available for this recording.")
            else:
                st.text_area(
                    "Transcript Text",
                    value=result.full_text,
                    height=240,
                    label_visibility="collapsed",
                )

                # Export / Download Buttons
                base_name = Path(st.session_state.result_filename).stem or "transcript"
                plain_filename = f"{base_name}_transcript.txt"

                col_dl1, col_dl2 = st.columns([1, 3])
                with col_dl1:
                    st.download_button(
                        label="📥 Download Transcript (.txt)",
                        data=result.full_text.encode("utf-8"),
                        file_name=plain_filename,
                        mime="text/plain; charset=utf-8",
                        use_container_width=True,
                    )
                with col_dl2:
                    st.caption("UTF-8 encoded standard text file compatible with any editor.")

        # TAB 2: Timestamped Segments
        with tab_timed:
            if not result.segments:
                st.info("No segments detected.")
            else:
                timestamped_text = build_transcript_text(
                    result.segments,
                    include_timestamps=True,
                    timestamp_style=config["timestamp_format"],
                )

                # Segment cards
                for idx, seg in enumerate(result.segments, 1):
                    t_start = format_timestamp(seg.start, format_style=config["timestamp_format"])
                    t_end = format_timestamp(seg.end, format_style=config["timestamp_format"])
                    st.markdown(
                        f"""
                        <div class="segment-card">
                            <div class="segment-header">
                                <span class="timestamp-badge">[{t_start} → {t_end}]</span>
                                <span style="font-size: 0.78rem; color: #64748b;">Segment #{idx} • {seg.duration:.1f}s</span>
                            </div>
                            <div class="segment-text">{seg.text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Download timestamped transcript
                timed_filename = f"{base_name}_with_timestamps.txt"
                st.download_button(
                    label="📥 Download Timestamped Text (.txt)",
                    data=timestamped_text.encode("utf-8"),
                    file_name=timed_filename,
                    mime="text/plain; charset=utf-8",
                )

        # TAB 3: Quality Inspector Table
        with tab_table:
            if not result.segments:
                st.info("No segment data to display.")
            else:
                table_rows = [seg.to_dict() for seg in result.segments]
                st.dataframe(
                    table_rows,
                    use_container_width=True,
                    column_config={
                        "start": st.column_config.NumberColumn("Start (s)", format="%.2f"),
                        "end": st.column_config.NumberColumn("End (s)", format="%.2f"),
                        "duration": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
                        "text": st.column_config.TextColumn("Transcribed Text", width="large"),
                        "avg_logprob": st.column_config.NumberColumn("Avg Log Prob", format="%.3f"),
                        "no_speech_prob": st.column_config.NumberColumn("No-Speech Prob", format="%.3f"),
                    },
                )
                st.caption(
                    "📌 **Avg Log Prob**: Closer to 0 indicates higher model confidence. "
                    "**No-Speech Prob**: Probability that the segment contains no speech."
                )


if __name__ == "__main__":
    main()
