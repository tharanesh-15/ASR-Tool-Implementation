# ClearSpeak 🎙️

**ClearSpeak** is a local, privacy-first Automatic Speech Recognition (ASR) web application built with **Streamlit** and **faster-whisper**. It provides a modern, beginner-friendly interface to upload audio recordings, transcribe spoken dialogue, inspect time-synchronized segments, and export clean text transcripts directly on your machine.

---

## 🌟 Key Features

- **100% Local & Private Processing**: Audio recordings are processed on your local machine using CTranslate2. No data is ever transmitted to external transcription APIs or remote servers.
- **🎙️ Live Microphone Recorder**: Record your speech directly inside the web browser using the built-in live audio recorder and convert it into text on the fly.
- **📁 Multi-Format Audio File Upload**: Supports `.wav`, `.mp3`, `.m4a`, `.mp4`, `.mpeg`, `.webm`, `.ogg`, and `.flac`.
- **Multiple Whisper Model Tiers**: Seamlessly switch between `tiny`, `base`, `small`, `medium`, and `large-v3` models based on your hardware capabilities and accuracy needs.
- **Intelligent Language Support**: Choose automatic language detection or specify one of 15+ major languages (including English, Hindi, Spanish, French, German, Japanese, Portuguese, Arabic, and Chinese).
- **Time-Synchronized Segments**: Review transcripts segmented chronologically with start and end timestamps.
- **Quality Inspector**: View segment-level confidence scores (average log-probability) and no-speech probabilities.
- **Single-Click UTF-8 Export**: Download your clean transcript or timestamped subtitles as a standard `.txt` file.
- **Session-Level Model Caching**: Whisper models are cached in memory via Streamlit's `@st.cache_resource`, avoiding repetitive disk reads during an active session.
- **Automated Audio Cleanup**: Uploaded and recorded audio buffers are securely removed from disk immediately following processing.

---

## 🧰 Tools and Technologies Used

| Technology | Purpose | Description |
| :--- | :--- | :--- |
| **Python (>= 3.10)** | Runtime Environment | High-performance core programming language. |
| **Streamlit** | Web Interface | Lightweight, reactive Python framework for interactive web applications. |
| **faster-whisper** | ASR Inference Engine | Re-implementation of OpenAI's Whisper model utilizing **CTranslate2**, delivering up to 4x faster execution and lower memory usage than standard Whisper. |
| **CTranslate2** | Model Quantization | Fast inference engine supporting INT8, FP16, and FP32 quantization on CPU and GPU. |
| **ffmpeg** | Audio Demuxing & Decoding | Universal media decoder used under the hood to ingest diverse audio codecs. |
| **pytest / unittest** | Automated Testing | Comprehensive unit testing suite using mock models to ensure continuous reliability without hardware or download dependencies. |

---

## 💻 Hardware and Resource Requirements

| Whisper Model | Disk Size | RAM / VRAM | Relative Speed | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **tiny** | ~75 MB | ~1 GB | ~32x | Rapid drafts, very low-spec hardware |
| **base** | ~145 MB | ~1 GB | ~16x | Podcasts, clear English speech |
| **small** *(Default)* | ~480 MB | ~2 GB | ~6x | **Recommended**: Great balance of speed & accuracy |
| **medium** | ~1.5 GB | ~5 GB | ~2x | Accented speakers, technical jargon |
| **large-v3** | ~3.1 GB | ~10 GB | ~1x | Maximum precision, complex multilingual audio |

> [!NOTE]
> ClearSpeak automatically detects whether an NVIDIA GPU with CUDA is available. If an NVIDIA GPU is detected, ClearSpeak uses `float16` compute. On CPU systems, ClearSpeak defaults to `int8` quantization to ensure fast transcription and low memory consumption.

---

## 📋 Prerequisites

1. **Python 3.10 or newer** installed.
2. **ffmpeg** installed and added to your system `PATH`:
   - **Windows**: Install using winget (`winget install Gyan.FFmpeg`) or Chocolatey (`choco install ffmpeg`).
   - **macOS**: Install using Homebrew (`brew install ffmpeg`).
   - **Linux (Ubuntu/Debian)**: Install using apt (`sudo apt update && sudo apt install -y ffmpeg`).

---

## 🚀 Setup and Installation

### Windows PowerShell

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd C:\Users\thara\.gemini\antigravity\scratch\clearspeak
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

4. Upgrade pip and install the dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### macOS and Linux

1. Open your terminal and navigate to the project directory:
   ```bash
   cd clearspeak
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Upgrade pip and install the dependencies:
   ```bash
   python3 -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

With your virtual environment activated:

```bash
streamlit run app.py
```

The Streamlit server will start and open in your default browser at `http://localhost:8501`.

### Step-by-Step Usage:
1. **Choose Model & Language**: From the left sidebar, select a Whisper model tier (default: `small`) and desired language (default: `Auto-Detect`).
2. **Provide Audio**:
   - **Option A - Upload Audio**: Drag and drop your audio recording (`.wav`, `.mp3`, `.m4a`, etc.).
   - **Option B - Live Microphone**: Select **🔴 Live Microphone Recorder**, click the record button, speak into your mic, and click stop.
3. **Preview**: Verify your audio using the built-in player.
4. **Transcribe**: Click **🚀 Transcribe Audio**.
5. **Review & Export**:
   - Inspect detected language and metrics.
   - Switch between **Clean Transcript**, **Timestamped Segments**, and the **Quality Inspector**.
   - Download the transcript as a UTF-8 text file.

---

## 🧪 Running the Tests

ClearSpeak includes a comprehensive suite of unit tests in `test_transcriber.py`. The tests use synthetic mock models, meaning **no model downloads, network access, or audio hardware are required**.

Run tests using `pytest`:
```bash
pytest test_transcriber.py -v
```

Or run tests using the built-in Python `unittest` runner:
```bash
python -m unittest test_transcriber.py -v
```

### What the Tests Verify:
- Exact timestamp formatting across seconds, minutes, hours, and sub-second boundaries.
- Clean text and timestamped segment assembly from segment objects and raw dictionaries.
- Handling of silent and 0-byte audio without unhandled crashes.
- Guaranteed deletion of temporary audio files on both success and exceptions.
- Model metadata integrity and audio file extension checks.

---

## 📦 Setting Up a GitHub Repository

To push this project to a new GitHub repository:

1. Initialize git in the project root:
   ```bash
   git init
   ```

2. Add the project files and create your initial commit:
   ```bash
   git add .
   git commit -m "Initial commit: ClearSpeak ASR application"
   ```

3. Set your main branch:
   ```bash
   git branch -M main
   ```

4. Create a new repository on GitHub (e.g., named `clearspeak`).

5. Link your local repository to GitHub and push:
   ```bash
   git remote add origin https://github.com/<your-username>/clearspeak.git
   git push -u origin main
   ```

*(Note: The included `.gitignore` automatically prevents model caches, virtual environments, audio samples, and temporary files from being committed.)*

---

## ⚠️ Important Notices: Model Downloads & Data Privacy

- **First-Time Model Download**: The first time you select a model tier (e.g., `small`), `faster-whisper` downloads the model weights from Hugging Face to your local cache (`~/.cache/huggingface/hub`). Subsequent runs load the model instantly from your local disk without redownloading.
- **Privacy & Hosting Note**: When running locally, all transcription takes place strictly on your device. However, **if you deploy this Streamlit app to a remote server or public cloud** (e.g. Streamlit Community Cloud, AWS, GCP), audio uploaded by users will be transmitted to and processed on that remote host.

---

## 📄 License

MIT License. Free to use, modify, and distribute for personal and commercial projects.
