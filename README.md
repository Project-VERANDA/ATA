# Audio-Transcript-Anonymizer (ATA)

A comprehensive pipeline for privacy-preserving audio processing. This tool accepts audio or video files, transcribes the content using WhisperX, applies speaker diarization via Pyannote, and anonymizes Personally Identifiable Information (PII) using BERT and LLMs. It uniquely supports audio-level anonymization, replacing sensitive spoken segments with beeps to create fully anonymized audio files.

Ideal for: Interviews, therapy sessions, legal consultations, and any conversation involving multiple speakers where privacy is paramount. 

# 🚀 Features
    
  Video to Audio Extraction: Automatically isolates audio from video files (MP4, MKV, etc.).
  
  * High-Accuracy Transcription: Powered by WhisperX (Faster-Whisper) with support for 12+ languages.
  
  * Speaker Diarization: Identifies and separates speakers using Pyannote.audio.
  
  * Text Anonymization (BERT): Detects and replaces PII (names, emails, locations, phone numbers, etc.) using a multilingual BERT model.
  
   * 🆕 Audio Anonymization (Beep Replacement): Maps identified PII to audio timestamps and replaces sensitive segments with beeps, preserving the rest of the conversation.
  
  * LLM Rewriting: Uses Large Language Models to generalize indirect identifiers (e.g., specific job titles, rare locations) that BERT might miss.
  
  * Adversarial Mode: Optional Red Team vs. Blue Team loop to iteratively improve anonymization quality.
  
  * Web Interface: User-friendly GUI for uploading, recording, editing, and downloading anonymized results.
  
  * Bulk Processing: Supports batch processing via command line or ZIP file uploads.

  * TTS Backends:
    
  * Piper TTS (Recommended)
- Fast offline synthesis
- 30+ languages
- GPL v3 license
- Voice: en_US-lessac-medium

  * Coqui XTTS v2
- Voice cloning support
- 17 languages
- CPML license (non-commercial)
- Model: ~2GB download

# 📦 Installation Prerequisites

    Python 3.8+
    FFmpeg: Required for audio/video conversion.
    Ubuntu/Debian: sudo apt-get install ffmpeg
    macOS: brew install ffmpeg
    Windows: Download from ffmpeg.org and add to **PATH**.
    **NVIDIA** **GPU** (Optional but Recommended): For 10x faster processing.

**Step-by-Step Setup**

  Clone the repository:
```
    git clone https://github.com/Project-VERANDA/ATA/
    cd ATA
``` 
   Run the setup script:
```
    ./ATA_Setup.sh
```
   Follow the prompts.

   * Choose y to install Flask (Web Interface) if you plan to use the **GUI**.
   * Input n if you only need the command-line interface.

   * Configure Hugging Face Access (One-time):
   
   * Create an account at Hugging Face.
   
   * Request access to the diarization model: Visit pyannote/speaker-diarization-community-1 and fill out the form.
   
   * Generate a Read-Only Token: Go to Settings/Tokens.
   
   * When prompted during setup, paste your token.
   
   **(Optional) Configure LLM API:**
   
   * Create a .env file in the ATA root directory (see Configuration below).
   
   * This is required for the LLM rewriting and adversarial features.



# 🛠️ Usage Option A: Command Line Interface (**CLI**)

Run the full pipeline with granular control:

## Run full pipeline (Default)

`python pipeline/process.py`

## Run with specific language (e.g., German)

`python pipeline/process.py --lang DE`

## Skip diarization for faster processing

`python pipeline/process.py --disable-diarization`

## Run only LLM rewrite on existing anonymized files

`python pipeline/process.py --llm-only --llm-model medgemma27b`

## Enable adversarial anonymization (3 iterations)

`python pipeline/process.py --adversarial`

## Exclude specific tags from anonymization

`python pipeline/process.py --exclude-tags PROFESSION QUANTITY`

# CLI Arguments Reference
Argument	Description
```--disable-transcription	Skip audio extraction from video files.

--disable-diarization	Skip speaker identification (uses generic labels).

--disable-anonymization	Skip **BERT**-based **PII** removal.

--disable-llm	Disable **LLM**-based indirect identifier removal.

--llm-only	Skip **BERT**; process existing annonym folder files with **LLM** only.

--lang <**CODE**>	Force language (AR, DE, EN, FI, FR, HI, IT, PL, PT, SP, ES, TR).

--llm-model <**MODEL**>	Select specific **LLM** model (e.g., medgemma, gpt-oss-120b).

--verbose	Enable debug-level logging.

--adversarial	Enable dual-agent adversarial anonymization loop.

--include-tags <**TAGS**>	Restrict anonymization to specific tags (e.g., **PERSON** **ORG**).

--exclude-tags <**TAGS**>	Ignore specific tags (e.g., **PROFESSION**).`
```

## Option B: Web Interface

  Start the server:

    python interactive_app/app.py

  Note: The server generates a self-signed **SSL** certificate automatically. You may need to accept the security warning in your browser.

  Access the interface:
    Open [https://**127**.0.0.1:**5001**](https://**127**.0.0.1:**5001**) in your browser.
  HTTPS is required for the microphone recording feature.

  Workflow:
   * Upload: Drag & drop audio/video files or click to browse.
   
   * Record: Use the built-in microphone recorder for live transcription.
   
   * Transcribe: Select language and click *Transcribe*.
   
   *  Edit: Correct any transcription errors manually.
   
   * Anonymize: Click Anonymize Text to apply BERT and/or LLM.
   
   * 🆕 Generate Anonymized Audio: Click this button to replace PII segments with beeps.
   
   * Download: Download the text transcript and the _beeped.wav audio file.

🔧 Configuration Environment Variables (.env)

Create a .env file in the ATA root directory to configure optional features:

## LLM API Configuration (Required for LLM Rewrite & Adversarial Mode)

```
CHAT_AI_API_KEY=your_api_key_here 
CHAT_AI_ENDPOINT=https://your-api-endpoint.com/v1
CHAT_AI_MODEL=gpt-oss-120b
```

## Web Interface Settings
```
USE_HTTPS=true
```
  ⚠️ Note: You must update this to your own API provider (e.g., OpenAI, local LLM server) for external use.

# Hardware Requirements

|Component|CPU Only|With NVIDIA GPU|
----|----|----
| WhisperX Transcription|~13x slower|Optimal|
----|----|----|
|Speaker DiarizatioN|~13x slower|Optimal|
----|----|----
|Memory (**RAM**)|**8GB** minimum|**16GB** recommended|
----|----|----
|Storage (Models)|~**30GB**|	~**30GB**|


### Verify **GPU** Availability:

```
python -c "import torch; print('**CUDA** Available:', torch.cuda.is_available())"
```

# 📂 Output Directory Structure

**ATA**/ 
├── pipeline/ │   
├── videos/          # Input video files │   
├── audios/          # Extracted WAV files │   
├── transcripts/     # Raw transcription files (.txt) │   
├── anonym/          # BERT-anonymized transcripts (_anon.txt) │   ├── LLM-Anon/    # LLM-rewritten transcripts (_llm_.txt, _adversarial_.txt) 
│   
├── uploads/         # Temporary storage for web uploads (includes _beeped.wav files) 
│   ├── model/           # Downloaded AI models 
│    └── logs/            # Session logs (session_YYYY-MM-DD_HH-MM-SS.txt) 
└── interactive_app/     # Web interface files

Note on Audio Outputs:

    Anonymized audio files generated via the web interface are saved in pipeline/uploads/ with the suffix _beeped.wav (e.g., interview_beeped.wav).
    These files are temporary; download them via the web interface or move them manually.

🎙️ New Feature: Audio Beep Replacement

The pipeline now supports audio-level anonymization. ### How It Works

    Transcription & Identification: The system transcribes audio and identifies **PII** entities using the **BERT** model.
    Offset Mapping: The system maps the text-based **PII** tags back to their precise time offsets in the original audio file.
    Beep Replacement: Using AudioBeepReplacer, the identified segments are replaced with a 1000Hz sine wave beep (adjustable).
    Output: Generates a new audio file where sensitive information is audibly masked while preserving the rest of the conversation.

### Programmatic Usage

If you are integrating this into your own scripts:

from audio_utils import AudioBeepReplacer

# 'offsets' must be obtained from the transcription step (returned by transcribe_audio_locally)

replacer = AudioBeepReplacer(beep_freq=1000, beep_gain_db=-6) output_file = replacer.replace_offsets_with_beeps(input.wav, offsets, output_beeped.wav)

🛡️ Supported **PII** Tags

The BERT anonymization model detects and replaces the following entity types:

|Tag |	Description	| Replacement|

PERSON|Names|[**PERSON**]
----|----|----
PERSON_EMAIL |	Email addresses |	[**EMAIL**]
----|----|----
PERSON_SOCIAL_RELATION |	Family/Social relations	| [NAME_RELATIVE]
----|----|----
ORG |	Organizations |	[**ORGANISATION**]
----|----|----
LOC_CITY | Cities |	[**CITY**]
----|----|----
LOC_COUNTRY	| Countries |	[**COUNTRY**]
----|----|----
LOC_STREET	| Street addresses	| [**STREET**]
----|----|----
DATETIME	| Dates/Times |	[**DATETIME**]
----|----|----
DATETIME_AGE |	Age references |	[**AGE**]
----|----|----
CODE_PHONE	| Phone numbers	| [**PHONE**]
----|----|----
CODE_URL |	URLs |	[**URL**]
----|----|----
**PROFESSION** |	Job titles |	[**PROFESSION**]

Use --include-tags or --exclude-tags to customize which tags are anonymized. 

# 🐞 Troubleshooting Issue: 

## Issue: *Offset alignment error* in logs

Cause: The text transcription does not perfectly match the audio timing. Solution:

    This is a known limitation of **ASR**. The system attempts auto-correction.
    Manually editing the transcript in the web interface before generating audio improves alignment.

## Issue: *Diarization model access denied*

Solution:

    Ensure you have requested access at Hugging Face.
    Run huggingface-cli login and enter your token.

## Issue: Web interface not accessible

Solution:

    Check if port **5001** is blocked by a firewall.
    Verify USE_HTTPS=true in .env if using **HTTPS**.
    Review logs in pipeline/logs/.

# 🔄 Pending Updates & Roadmap

* [ ] Batch Processing: Add folder-based batch processing (currently ZIP only).
* [ ] Local LLM Support: Switch from API calls to local LLM models (e.g., Ollama, LM Studio).
* [ ] Live Streaming: Implement chunk-processing for live transcription and anonymization.
* [ ] Non-NVIDIA GPU: Add support for AMD/Intel GPUs.
* [x] Re-adding the anonymization process.
* [x] Add a recording button to the web interface for demos.
* [ ] Add batch transcripting via selected folder instead of only with selected files.
* [ ] Add automatic cleanup of the input video/audio folders. 
* [ ] Add arguments for a debug running of the pipeline.
* [x] Switch the diarization to speaker-diarization-community-1 from the older diarization model.
* [x] Update the Running the Script section of this ReadMe.
* [ ] Add code enabling other non-nvidia GPUs.

* [ ] Setup & Installation
  *  [x] Automate the downloading of WhisperX models.
* [ ] Web interface
  *   [ ] Update the existing web interface to be a "demo" interface.
  *   [ ] Create a new production interface.
      * [ ] Add buttons for the interface to enable/disable each feature in the pipeline.
  *   [ ] Move from Flask to a production service.
  *   [x] Integrate Docker
  *   [ ] Integrate Keycloak Auth
    *   [ ] Connect to Charité Keycloak instance.
    *   [ ] Spin-up local Keycloak server if no existing service detected.
  *   [ ] Update with correct and full citation information & contact information.
* [ ] Transcription & Audio Processing
  *  [x] Disable the translation feature of WhisperX.
  *  [ ] Confidence scoring	❌ Missing	Can't filter low-quality segments
  *  [ ] Audio normalization	❌ Missing	Inconsistent volume levels affect ASR accuracy
  *  [ ] Noise reduction preprocessing	❌ Missing	Background noise reduces transcription quality
  *  [ ] Long-file segmentation	❌ Missing	Files >30min may timeout or degrade
  *  [ ] Language detection confidence	⚠️ Limited	Auto-detect has no confidence metric exposed
  *  [ ] Speaker count verification	❌ Missing	No validation of MIN/MAX speaker settings
* [ ] Output & Formatting
  *  [ ] SRT/VTT subtitle export	❌ Missing	No video subtitle generation
  *  [ ] Timestamp preservation	❌ Missing	Final transcripts lose timing data
  *  [ ] JSON structured export	❌ Missing	Only plaintext output
  *  [ ] Speaker metadata retention	⚠️ Partial	Diarization info discarded after transcript generation
  *  [ ] Word-level alignment export	❌ Missing	Character/timestamp data not saved
* [ ] Batch Processing & Performance
  *  [ ] Progress bars	❌ Missing	No real-time status during long runs
  *  [ ] Parallel file processing	❌ Missing	Sequential only — slow for large batches
  *  [ ] Retry/resume on failure	❌ Missing	Failed files abort entire batch
  *  [ ] Checkpointing	❌ Missing	Can't restart mid-workflow
  *  [ ] CPU/GPU utilization monitoring	❌ Missing	No resource usage visibility
* [ ] Configuration & Usability
  *  [ ] Config file support	❌ Missing	Everything via CLI args only
  *  [ ] Named presets	❌ Missing	Can't save/load workflow configurations
  *  [ ] Interactive prompts	❌ Missing	No guided setup for new users
  *  [ ] Default profile switching	❌ Missing	All options reset per run
  *  [ ] Verbose vs quiet modes	⚠️ Partial	Only one verbose flag
  *  [x] Add arguments for process.py to enable/disable each feature in the pipeline.  
* [ ] Validation & Quality Assurance
  *  [ ] Post-transcription quality metrics	❌ Missing	No WER/CER scoring
  *  [ ] PII leakage testing	❌ Missing	Can't verify anonymization completeness
  *  [ ] Duplicate file detection	❌ Missing	Processes same file multiple times
  *  [ ] File integrity checks	❌ Missing	No hash verification
  *  [ ] Speaker overlap detection	❌ Missing	Can't identify overlapping speech
* [ ] Data Management
  *  [ ] Automatic cleanup	❌ Missing	Intermediate WAV files accumulate
  *  [ ] Archive/backup	❌ Missing	No version control of outputs
  *  [ ] Cross-run deduplication	❌ Missing	Same transcript regenerated repeatedly
  *  [ ] Organized output structure	⚠️ Basic	Flat folders, no categorization
  *  [ ] Retention policies	❌ Missing	Manual file deletion required
* [ ] Integration & API
  *  [ ] REST API endpoint	❌ Missing	No programmatic access
  *  [ ] Webhook notifications	❌ Missing	Can't trigger downstream workflows
  *  [ ] Cloud storage hooks	❌ Missing	S3/GCS/Azure integration missing
  *  [ ] Database connectivity	❌ Missing	No persistent metadata store
  *  [ ] Event logging system	⚠️ Basic	Only session logs, no event stream
* [ ] Security & Compliance
  *  [ ] Secure file deletion	❌ Missing	rm leaves recoverable data
  *  [ ] Output encryption	❌ Missing	Plain-text files only
  *  [ ] Access control	❌ Missing	No user authentication
  *  [ ] Audit trail	⚠️ Basic	Session logs lack detailed actions
  *  [ ] Compliance reporting	❌ Missing	No GDPR/HIPAA documentation
* [ ] User Feedback & Monitoring
  *  [ ] Real-time processing preview	❌ Missing	Wait blindly until completion
  *  [ ] Cost estimation	❌ Missing	LLM token costs unknown upfront
  *  [ ] Error aggregation/analytics	❌ Missing	Hard to spot systematic failures
  *  [ ] Performance benchmarking	❌ Missing	No baseline for optimization
  *  [ ] Health check endpoints	❌ Missing	Can't verify system readiness
* [ ] Large language models
  * [ ] Switch from API calls for the AI models to local model processing.
    *  [ ] Add a check/installation for local models, and if not, fall back to API calls.
    *  [ ] Disable the LLM rewrite function when no LLM model is available.
    *  [x] Switch the .env file to also host the LLM API endpoint variable, instead of hardcoding it into the code.
  * [ ] Do some optimizations for the LLM system prompt.
  * [ ] Try getting all the local LLM models working.
* [ ] Text-to-speech
  *   [x] Add text to speech for the output transcripts.
  *   [x] Improved TTS: Switch from Google TTS to a local text-to-speech engine.
  *   [ ] Get Coquit xtts working.

* [ ] 

# 📄 License & Credits

This project utilizes:

    WhisperX (Faster-Whisper)
    Pyannote.audio
    Hugging Face Transformers (mmbert_multilingual_pii_ner)
    Flask

Developed for privacy-preserving audio analysis.

# Long-term improvements

* Improved dialogue and accent handling.
* K-Anonymity for databases or folders.
