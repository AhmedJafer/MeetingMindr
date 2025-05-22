# MeetingMindr

> AI-powered meeting transcription with speaker diarization, intelligent summarization, and automated quality evaluation.

[![Python Version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/)

## 🎯 Overview

MeetingMindr automatically processes audio and video meeting recordings to generate high-quality transcripts with speaker identification and AI-powered summaries. The system includes built-in quality evaluation to ensure summary accuracy and completeness.

## ✨ Key Features

- **Multi-format Support**: Process audio files (WAV, MP3) and video files (MP4, MOV, AVI, MKV, WEBM)
- **Advanced Speaker Diarization**: Powered by PyAnnote for accurate speaker identification
- **High-Quality Transcription**: Uses OpenAI Whisper for precise speech-to-text conversion
- **Intelligent Summarization**: AI-generated structured summaries with key decisions and action items
- **Quality Assurance**: Built-in summary evaluation and automatic improvement system
- **Flexible Output**: Generates both detailed transcripts and concise summaries

## 🏗️ Processing Pipeline

MeetingMindr follows a systematic processing pipeline:

1. **Input Processing**: Automatic video-to-audio conversion if needed
2. **Speaker Diarization**: Identify when each speaker talks using PyAnnote
3. **Speech Transcription**: Convert speech to text using Whisper
4. **Speaker Mapping**: Align transcribed text with identified speakers
5. **Transcript Formatting**: Structure output with timestamps and speaker labels
6. **AI Summarization**: Generate structured summary using LLM
7. **Quality Evaluation**: Assess and improve summary quality automatically
8. **Output Generation**: Save formatted transcript and summary to files

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- Hugging Face account and token
- OpenAI API access or compatible LLM API

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/meetingmindr.git
cd meetingmindr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

1. **Get Hugging Face Token**: Visit [Hugging Face](https://huggingface.co/settings/tokens) and create an access token
2. **Accept PyAnnote License**: Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the license
3. **Create Environment File**: Create a `.env` file in the project root:

```env
# Required
HUGGING_FACE_TOKEN=your_hugging_face_token
LLM_API_KEY=your_llm_api_key

# Optional - for separate evaluation model
EVALUATION_API_KEY=your_evaluation_api_key
EVALUATION_BASE_URL=https://api.openai.com/v1
EVALUATION_MODEL_NAME=gpt-4o-mini
```

## 📋 Usage

### Command Line Interface (Recommended)

```bash
# Basic usage with environment variables
python main.py --audio_or_video_file "meeting.mp4" --model_name "gemini-1.5-flash"

# With explicit parameters
python main.py \
  --audio_or_video_file "team_meeting.mp4" \
  --device "cuda" \
  --hugging_face_token "your_hf_token" \
  --api_key "your_api_key" \
  --base_url "https://api.openai.com/v1" \
  --model_name "gpt-4o-mini" \
  --output_dir "Project_Meetings" \
  --audio_output_dir "Audio_Files"

# With separate evaluation model
python main.py \
  --audio_or_video_file "interview.mp3" \
  --model_name "gpt-4o" \
  --evaluation_model_name "gpt-4o-mini" \
  --evaluation_base_url "https://api.openai.com/v1"
```

### CLI Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--audio_or_video_file` | ✅ | - | Path to input audio/video file |
| `--model_name` | ✅ | - | LLM model name (e.g., gpt-4o, gemini-1.5-flash) |
| `--device` | ❌ | Auto-detect | Processing device (cuda/cpu) |
| `--hugging_face_token` | ❌ | From .env | Your Hugging Face token |
| `--api_key` | ❌ | From .env | LLM API key |
| `--base_url` | ❌ | Google AI | LLM API base URL |
| `--output_dir` | ❌ | Meeting_Summaries | Output directory for results |
| `--audio_output_dir` | ❌ | Extracted_Audio | Directory for extracted audio |
| `--evaluation_api_key` | ❌ | From .env | Optional separate evaluation API key |
| `--evaluation_base_url` | ❌ | From .env | Optional evaluation API base URL |
| `--evaluation_model_name` | ❌ | From .env | Optional evaluation model name |

### Programmatic Usage

```python
from summarization.interaction import CallInteraction

# Initialize MeetingMindr
processor = CallInteraction(
    audio_or_video_file="meeting.mp4",
    device="cuda",  # or "cpu"
    hugging_face_token="your_hf_token",
    api_key="your_openai_api_key",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4o-mini"
)

# Process the meeting
transcript, summary, feedback = processor.summarize_call()

print("Summary:", summary)
print("Quality Feedback:", feedback)
```

### Model Support

MeetingMindr supports various LLM providers:

**Google AI (Default)**
```bash
python main.py --audio_or_video_file "meeting.mp4" --model_name "gemini-1.5-flash"
```

**OpenAI**
```bash
python main.py \
  --audio_or_video_file "meeting.mp4" \
  --base_url "https://api.openai.com/v1" \
  --model_name "gpt-4o-mini"
```

**Custom/Local LLMs**
```bash
python main.py \
  --audio_or_video_file "meeting.mp4" \
  --base_url "http://localhost:1234/v1" \
  --model_name "llama-3.1-8b"
```

## 🔧 Core Components

### 1. Speaker Diarization (`processing/diarization.py`)
- Uses PyAnnote Audio 3.1 for state-of-the-art speaker separation
- Handles multiple speakers automatically
- GPU acceleration support

### 2. Speech Transcription (`processing/transcription.py`)
- Powered by OpenAI Whisper (base model)
- English language optimization
- Timestamp-accurate transcription

### 3. Speaker-Text Mapping (`processing/mapping.py`)
- Aligns transcribed text with identified speakers
- Handles overlapping speech segments
- Timestamp synchronization

### 4. Transcript Formatting (`processing/formatting.py`)
- Structures output for readability
- Time-stamped speaker attribution
- Clean text formatting

### 5. AI Summarization (`summarization/interaction.py`)
- Generates structured meeting summaries
- Includes key decisions, action items, and next steps
- Customizable prompt templates

### 6. Quality Evaluation (`summarization/evaluator.py`)
- Automated summary quality assessment
- Coverage, faithfulness, and clarity evaluation 
- Iterative improvement system (up to 5 retries)

## 📊 Output Format

MeetingMindr generates comprehensive output files containing:

### Summary Structure
- **Meeting Overview**: Title, attendees, duration
- **Executive Summary**: Key takeaways in 2-3 sentences
- **Main Topics**: Prioritized discussion points
- **Key Decisions**: Clearly marked decision points
- **Action Items**: With owners and deadlines
- **Next Steps**: Follow-up actions and meetings

### Full Transcript Format
```
[Speaker_00] : Welcome everyone to today's quarterly review meeting.
[Speaker_01] : Thanks for organizing this. I have the revenue numbers ready.
[Speaker_00] : Perfect. Let's start with the Q3 performance metrics.
```

## 🛠️ Configuration Options

### Environment Variables

Create a `.env` file for easier configuration:

```env
# Required
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_API_KEY=your_api_key_here

# Optional - for different evaluation model
EVALUATION_API_KEY=your_eval_api_key
EVALUATION_BASE_URL=https://api.openai.com/v1
EVALUATION_MODEL_NAME=gpt-4o-mini
```

### Device Configuration
- **Auto-detection**: System automatically selects best available device
- **GPU**: Use `--device cuda` for NVIDIA GPU acceleration
- **CPU**: Use `--device cpu` for CPU-only processing

### Model Options
- **Whisper**: Uses "base" model (good balance of speed/accuracy)
- **LLM Models**: 
  - `gemini-1.5-flash` (fast, cost-effective)
  - `gpt-4o-mini` (balanced performance)
  - `gpt-4o` (highest quality)
  - Custom/local models supported

### Output Structure
```
Meeting_Summaries/           # Main output directory
├── meeting_name.txt         # Combined summary + transcript
Extracted_Audio/             # Temporary audio files
├── meeting_name.wav         # Converted audio (if from video)
```

## 📈 Performance Characteristics

- **Processing Speed**: ~2-3x real-time for audio files
- **GPU Memory**: ~4-8GB VRAM recommended for optimal performance

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Use CPU instead
device = "cpu"
```

**Hugging Face Authentication Error**
```bash
# Ensure you've accepted the PyAnnote license
# Visit:
# - https://huggingface.co/pyannote/speaker-diarization-3.1
# - https://huggingface.co/pyannote/segmentation
# - https://huggingface.co/pyannote/speaker-diarization
# - https://huggingface.co/pyannote/segmentation-3.0
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Dependencies & Credits

- **[PyAnnote Audio](https://github.com/pyannote/pyannote-audio)**: Speaker diarization
- **[OpenAI Whisper](https://github.com/openai/whisper)**: Speech recognition
- **[MoviePy](https://github.com/Zulko/moviepy)**: Video processing
- **[OpenAI API](https://platform.openai.com/)**: AI summarization
- **[Pydantic](https://pydantic.dev/)**: Data validation

## 📞 Support

For issues, feature requests, or questions:
- Open an issue on [GitHub Issues](https://github.com/yourusername/meetingmindr/issues)
---

*Transform your meetings into actionable insights with MeetingMindr* 🎯