import os
import sys
import tempfile
import time
import re
import subprocess
import logging
import ipaddress
import zipfile
import shutil
import difflib
import random
import asyncio
from openai import OpenAI
from pathlib import Path
from datetime import datetime, timezone, timedelta
from werkzeug.utils import secure_filename
from collections import OrderedDict
from audio_utils import (AudioBeepReplacer, AudioError, FFMPEG, SUPPORTED_OUTPUT_FORMATS,
                         compute_waveform, render_redacted_audio)
from io import BytesIO
from dotenv import load_dotenv
import json
import requests
import hmac
import uuid
import threading
from werkzeug.exceptions import HTTPException


import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ============================================================================

CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://llm.cloud.cci.charite.de/v1')

# --- Path Configuration ---s

load_dotenv()


# Load config.sh

CONFIG_PATH = Path(__file__).parent / "config.sh"
if CONFIG_PATH.exists():
    # Source config.sh and load into environment
    import subprocess
    result = subprocess.run(
        ["bash", "-c", f"source {CONFIG_PATH} && env"],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value.strip('"\''))

# Certificate paths (use your config.sh variables)
FLASK_SSL_CERT = os.getenv("LETSENCRYPT_FULLCHAIN_PATH") or \
                 "/etc/letsencrypt/live/transcriber.cloud.cci.charite.de/fullchain.pem"
FLASK_SSL_KEY = os.getenv("LETSENCRYPT_PRIVKEY_PATH") or \
                "/etc/letsencrypt/live/transcriber.cloud.cci.charite.de/privkey.pem"

# Bind settings
FLASK_BIND_HOST = os.getenv("FLASK_BIND_HOST") or "0.0.0.0"
FLASK_BIND_PORT = int(os.getenv("FLASK_BIND_PORT") or "5001")

# Domain name (for logging)
DOMAIN_NAME = os.getenv("DOMAIN_NAME") or "transcriber.cloud.cci.charite.de"

# Determine the directory containing this script (interactive_app/)
current_script_dir = Path(__file__).resolve().parent

# Determine the project root (ATA/)
project_root = current_script_dir.parent

# Define Pipeline and Model paths EXACTLY as process.py does
pipeline_path = project_root / "pipeline"
MODEL_FOLDER = pipeline_path / "model"

# Add paths to sys.path
if str(pipeline_path) not in sys.path:
    sys.path.insert(0, str(pipeline_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- TTS Imports (moved from process.py to tts_engine) ---
try:
    from process import (
        transcribe_audio_locally, 
        load_models, 
        call_llm_rewriter, 
        generate_paraphrase,
        AVAILABLE_LLM_MODELS, 
        CHAT_AI_API_KEY, 
        CHAT_AI_ENDPOINT,
        MODEL_FOLDER as PROCESS_MODEL_FOLDER,
        LLM_ANONYM_FOLDER,
        ANONYM_FOLDER,
        BASE_PATH,
        AnonymizationEngine,
        transcribe_audio_result
    )
except ImportError as e:
    logging.critical(f"Failed to import from process.py: {e}")
    logging.critical(f"Looking in pipeline: {pipeline_path}")
    sys.exit(1)

# Import TTS directly from tts_engine (NOT through process.py)
try:
    from pipeline.tts.tts_engine import (
        generate_speech,
        synthesize_segment,
        get_available_tts_voices,
        get_tts_status,
        TTS_BACKEND,
        TTS_ENABLED,
    )
except ImportError as e:
    logging.critical(f"Failed to import tts_engine: {e}")
    sys.exit(1)

import wave

MIN_TTS_WAV_SECONDS = 1.0   # a multi-minute transcript must yield far more than this


def _wav_content_seconds(path):
    """Audio duration measured from the WAV data chunk, independent of file size."""
    try:
        with wave.open(path, 'rb') as wav:
            return wav.getnframes() / float(wav.getframerate() or 1)
    except (wave.Error, EOFError, OSError):
        return 0.0

# Verify model path consistency
if MODEL_FOLDER != PROCESS_MODEL_FOLDER:
    logging.warning(f"⚠️  Path mismatch detected! app.py: {MODEL_FOLDER}, process.py: {PROCESS_MODEL_FOLDER}")
    # Force alignment (process.py usually wins, but we align)
    MODEL_FOLDER = PROCESS_MODEL_FOLDER

# --- Dependency Checks ---
try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("PyDub not available. Audio processing features may be limited.")
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logger.warning("WhisperX not available. Transcription features may be limited.")

# --- Application Configuration ---
BERT_ANONYMIZER_AVAILABLE = True
DEFAULT_MODEL = 'bert-base-ner'
TTS_AVAILABLE = TTS_ENABLED

SSL_CERT_DIR = project_root / "ssl_certs"
SSL_CERT_FILE = SSL_CERT_DIR / "server.crt"
SSL_KEY_FILE = SSL_CERT_DIR / "server.key"
SSL_MARKER_FILE = SSL_CERT_DIR / ".generated_by_ata"

# Ensure SSL directory exists
SSL_CERT_DIR.mkdir(parents=True, exist_ok=True)

import torch
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('ATA_MAX_UPLOAD_BYTES', 5 * 1024 * 1024 * 1024))

# ============================================================================
# Security, storage and API infrastructure
# ============================================================================
from pipeline import redaction as rd  # noqa: E402
from pipeline.jobs import ID_RE, JobError, JobStore  # noqa: E402
from pipeline.secure_store import (  # noqa: E402
    MasterKeyError, StorageError, ensure_free_space, ensure_private_dir, get_work_dir,
    load_master_key, work_path)
from pipeline.surrogates import apply_surrogates  # noqa: E402


def _env_flag(name, default=False):
    """Reads a boolean environment variable (1/true/yes/on)."""
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


MAX_JSON_BYTES = int(os.getenv('ATA_MAX_JSON_BYTES', 5 * 1024 * 1024))   # JSON request bodies
SIGNED_URL_TTL_S = int(os.getenv('ATA_SIGNED_URL_TTL', 300))
CORS_ORIGINS = {o.strip() for o in os.getenv('ATA_CORS_ORIGINS', '').split(',') if o.strip()}
BULK_ENABLED = _env_flag('ATA_ENABLE_BULK')
FILE_DOWNLOADS_ENABLED = _env_flag('ATA_ENABLE_FILE_DOWNLOADS')

UPLOAD_ROOT = ensure_private_dir(Path(os.getenv('ATA_UPLOAD_DIR') or project_root / 'uploads'))
app.config['UPLOAD_FOLDER'] = str(UPLOAD_ROOT)
app.bulk_batches = {}                      # batch_id -> metadata (bulk endpoints are opt-in)

try:
    MASTER_KEY = load_master_key()         # refuses to start if ATA_REQUIRE_MASTER_KEY=1 and no key
except MasterKeyError as exc:
    logger.critical("Cannot start: %s", exc)
    sys.exit(1)
JOB_STORE = JobStore.from_env(Path(os.getenv('ATA_JOBS_DIR') or project_root / 'jobs'), MASTER_KEY)
JOB_STORE.startup_sweep()
JOB_STORE.start_purge_thread()
logger.info("Job store ready (idle TTL %ss, max age %ss, key: %s)", int(JOB_STORE.idle_ttl),
            int(JOB_STORE.max_age), MASTER_KEY.source)


class ApiError(Exception):
    """Client-facing error rendered as ``{"success": false, "error", "code"}``."""

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def api_error(code, message, status=400, **extra):
    """Builds the standard JSON error response."""
    body = {'success': False, 'error': message, 'code': code}
    body.update(extra)
    return jsonify(body), status


def request_owner():
    """Identity hook for per-user job ownership (return the Keycloak ``sub`` once auth exists)."""
    return None


def _json_body():
    """Returns the JSON object body or raises ``ApiError``."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError('invalid_json', 'The request body must be a JSON object.')
    return data


def require_job(data=None, job_id=None):
    """Authenticates the job named in the request (id + token).

    The token comes from the ``X-Job-Token`` header (preferred; never put it in a URL).
    """
    data = data or {}
    job_id = job_id or data.get('job_id') or request.args.get('job_id') or request.form.get('job_id')
    token = request.headers.get('X-Job-Token') or data.get('job_token') or request.form.get('job_token')
    if not job_id:
        raise ApiError('job_required', 'A job_id is required. Transcribe the audio first.')
    try:
        return JOB_STORE.authenticate(job_id, token, request_owner())
    except JobError as exc:
        raise ApiError(exc.code, exc.message, exc.status)


def _origin_allowed(origin):
    return bool(origin) and ('*' in CORS_ORIGINS or origin in CORS_ORIGINS)


@app.before_request
def guard_request():
    """CORS preflight and a size cap for JSON bodies (uploads keep the larger limit)."""
    if request.method == 'OPTIONS' and _origin_allowed(request.headers.get('Origin')):
        return '', 204
    if request.mimetype == 'application/json' and (request.content_length or 0) > MAX_JSON_BYTES:
        return api_error('payload_too_large', 'The request body is too large.', 413)
    return None


@app.after_request
def add_security_headers(response):
    """Security headers, optional CORS, and no caching of API responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "media-src 'self' blob: data:; "
        "img-src 'self' data: https:; "
        "frame-ancestors 'none'; "
        "base-uri 'self';"
    )
    if not request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'no-store'      # transcripts/audio must not be cached
    origin = request.headers.get('Origin')
    if _origin_allowed(origin):
        response.headers['Access-Control-Allow-Origin'] = '*' if '*' in CORS_ORIGINS else origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Job-Token, X-Session-ID'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        response.headers['Access-Control-Max-Age'] = '600'
        response.headers.add('Vary', 'Origin')
    return response


# --- Model Pre-loading ---
try:
    if load_models()[0] is None:
        logger.error("Interactive App: Whisper model could not be pre-loaded; retrying on first use.")
    else:
        logger.info("Interactive App: Models pre-loaded successfully.")
except Exception as e:
    logger.error(f"Interactive App: Failed to pre-load models: {e}")

def transcribe_audio(audio_path, language='auto'):
    """
    Calls the shared engine from process.py with language support.
    Args:
        audio_path: Path to audio file
        language: 'auto', 'SP', 'ES', 'DE', etc.
    """
    lang_code = None
    if language and language != 'auto' and language in WHISPER_LANG_MAP:
        lang_code = WHISPER_LANG_MAP[language]
        logger.info(f"Transcribing with forced language: {lang_code}")
    else:
        logger.info("Transcribing with auto-detect")
    
    return transcribe_audio_locally(audio_path, language=lang_code)

# --- CONFIGURATION ---
CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://llm.cloud.cci.charite.de/v1')

# Faker Surrogate processing
try:
    from process import (
        apply_surrogate_substitution, 
        SURROGATES, 
        TAG_TO_SURROGATE_CATEGORY,
        ANONYM_FOLDER,
        LLM_ANONYM_FOLDER
    )
    HAS_FAKER = True
    logger.info("✅ Successfully imported surrogate functions from process.py")
except ImportError as e:
    HAS_FAKER = False
    SURROGATES = {}
    TAG_TO_SURROGATE_CATEGORY = {}
    apply_surrogate_substitution = None
    ANONYM_FOLDER = Path('anonym')
    LLM_ANONYM_FOLDER = Path('LLM-Anon')
    logger.error(f"❌ Failed to import from process.py: {e}")
    logger.warning("Some features may be disabled.")

# Model mappings for the UI

AVAILABLE_MODELS = {
    'medgemma': 'medgemma',
    'medgemma27b': 'medgemma27b',
    'gpt-oss-120b': 'gpt-oss-120b',
    'Qwen3.6-27B': 'Qwen3.6-27B',
    'Qwen3.5-27B': 'Qwen3.5-27B',      
    'qwen3-asr-1.7b': 'qwen3-asr-1.7b',
    'cle-Kimi-K2.6': 'cle-Kimi-K2.6',  
    'cle-Qwen3-Coder-Next-FP8': 'cle-Qwen3-Coder-Next-FP8',
    'cle-Qwen3.5-397B-A17B-FP8': 'cle-Qwen3.5-397B-A17B-FP8'
}

WHISPER_MODELS = OrderedDict([
    ('tiny', '1: Tiny (Fastest)'),
    ('base', '2: Base (Default)'),
    ('small', '3: Small'),
    ('medium', '4: Medium'),
    ('large', '5: Large (Best)')
])

# --- Language Configuration ---
# Supported Languages for UI
SUPPORTED_LANGUAGES = {
    'AR': 'Arabic',
    'DE': 'German',
    'EN': 'English',
    'FI': 'Finnish',
    'FR': 'French',
    'HI': 'Hindi',
    'IT': 'Italian',
    'PL': 'Polish',
    'PT': 'Portuguese',
    'SP': 'Spanish',
    'ES': 'Spanish',
    'TR': 'Turkish'
}

# Mapping for WhisperX
WHISPER_LANG_MAP = {
    'AR': 'ar', 'DE': 'de', 'EN': 'en', 'FI': 'fi', 'FR': 'fr',
    'HI': 'hi', 'IT': 'it', 'PL': 'pl', 'PT': 'pt', 
    'SP': 'es', 'ES': 'es', # Both map to 'es'
    'TR': 'tr'
}

# The static surrogate pools now live in pipeline/surrogates.py (single source of truth).

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg', 'webm'}

# REPLACE the simple pydub version with:
def generate_beep(duration_ms=400, freq=1000):
    """Generate a beep sound for PII tag placeholders (uses process.py implementation)."""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
        return Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-12)
    except ImportError:
        logger.warning("pydub not available. Beep generation disabled.")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def diff_relevant_offsets(offs, s2):
    """Legacy helper: spans to mask for word-level segments ``offs`` vs. redacted text ``s2``.

    Kept for backwards compatibility; new code uses :mod:`pipeline.redaction` directly.
    Returns ``[{"start", "end", "word": [..]}]`` like the old implementation, but with
    correct boundaries (first/last word are covered) and without masking punctuation,
    capitalisation or manual-edit differences.
    """
    words, _ = rd.words_from_segments(offs)
    return rd.legacy_offsets(rd.derive_spans(words, s2).spans)


def get_relevant_offsets(conversation, offsets):
    conversation=re.sub(r"SPEAKER_[0-9]*:", "", conversation)
    conversation=re.sub(r"\n", " ", conversation)
    conversation=re.sub(r"  *", " ", conversation)
    conversation=re.sub(r"^ *", "", conversation)

    tokens=conversation.split(" ")

    peep_array=[]
    cnt=0
    for i in range(len(offsets)):
        for j in range(len(offsets[i]['words'])):
            x_word=offsets[i]['words'][j]['word']
	
            if cnt<len(tokens):
                if re.match(r"\[.*\]", tokens[cnt]):
                    print ("peep!")
                    peep_array.append({'start': offsets[i]['words'][j]['start'], 'end':offsets[i]['words'][j]['end'], 'word':offsets[i]['words'][j]['word']})
                elif x_word!=tokens[cnt]:
                    logger.error(f"offset alginment error: {x_word} | {tokens[cnt]}")
	           
            cnt+=1
    
    return peep_array

_ANONYMIZER = None
_ANONYMIZER_LOCK = threading.Lock()
_TAG_NAME_RE = re.compile(r'^[A-Z][A-Z0-9_]{0,39}$')


def _get_anonymizer():
    """Lazily creates the shared anonymization engine (model load happens once)."""
    global _ANONYMIZER
    with _ANONYMIZER_LOCK:
        if _ANONYMIZER is None:
            engine = AnonymizationEngine(method="local_mmbert", level="standard",
                                         model_path=MODEL_FOLDER / "multilingual_DialogPII_NER")
            if not engine.method:
                raise ApiError('anonymizer_unavailable', 'The anonymization model is not available.', 503)
            _ANONYMIZER = engine
        return _ANONYMIZER


def _tag_list(value):
    """Validates an optional list of model tag names (``None`` stays ``None``)."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 64:
        raise ApiError('invalid_tags', 'include_tags/exclude_tags must be a list of tag names.')
    tags = [str(t).strip().upper() for t in value]
    if any(not _TAG_NAME_RE.match(t) for t in tags):
        raise ApiError('invalid_tags', 'Invalid tag name.')
    return tags


@app.route('/anonymize', methods=['POST'])
def anonymize():
    """Anonymizes a transcript.

    JSON body: ``transcript`` (or ``text``), optional ``include_tags`` / ``exclude_tags``
    (model tag names; an empty ``include_tags`` list anonymizes nothing) and optional
    ``job_id`` + token to link the result to uploaded audio for redaction.

    The anonymized text and the entity map (original PII) are kept in server *memory* only,
    with an idle timeout; they are never written to disk.
    """
    data = _json_body()
    text = data.get('transcript') or data.get('text') or ''
    if not isinstance(text, str) or not text.strip():
        raise ApiError('no_transcript', 'No transcript provided.')
    include_tags = _tag_list(data.get('include_tags'))
    exclude_tags = _tag_list(data.get('exclude_tags'))
    job = require_job(data) if data.get('job_id') else None

    engine = _get_anonymizer()
    result_text, success, msg, entity_map, _ = engine.anonymize(
        text, use_surrogates=False, include_tags=include_tags, exclude_tags=exclude_tags)
    if not success or result_text is None:
        logger.error("Anonymization failed: %s", msg)
        return api_error('anonymization_failed', 'Anonymization failed.', 500)

    session_id = job.id if job else uuid.uuid4().hex
    JOB_STORE.remember(session_id, {'entity_map': entity_map, 'placeholder_text': result_text})
    entity_count = sum(len(v) for v in entity_map.values())
    logger.info("Anonymization done: %d -> %d chars, %d entities", len(text), len(result_text), entity_count)
    return jsonify({
        'status': 'success', 'success': True,
        'original': text, 'anonymized': result_text,
        'chars_original': len(text), 'chars_anonymized': len(result_text),
        'session_id': session_id, 'job_id': job.id if job else None,
        'entity_count': entity_count,
    })


def replace_surrogates(text, lang="EN"):
    """Legacy helper: fill ``[TAG]`` placeholders with fakes (static pools, no consistency)."""
    return apply_surrogates(text, None, lang=lang, backend='pool').text


# ============================================================================
# ERROR HANDLERS (always JSON: {"success": false, "error", "code"})
# ============================================================================

@app.errorhandler(ApiError)
def handle_api_error(error):
    return api_error(error.code, error.message, error.status)


@app.errorhandler(JobError)
def handle_job_error(error):
    return api_error(error.code, error.message, error.status)


@app.errorhandler(rd.RedactionError)
def handle_redaction_error(error):
    return api_error(error.code, error.message, 400)


@app.errorhandler(AudioError)
def handle_audio_error(error):
    status = {'media_too_long': 413, 'unsupported_format': 415, 'ffmpeg_missing': 503,
              'render_failed': 500, 'render_timeout': 504, 'waveform_failed': 500}.get(error.code, 400)
    return api_error(error.code, error.message, status)


@app.errorhandler(StorageError)
def handle_storage_error(error):
    return api_error(error.code, error.message, 507)


@app.errorhandler(HTTPException)
def handle_http_exception(error):
    if error.code == 413:
        return api_error('payload_too_large', 'The upload exceeds the size limit.', 413)
    return api_error('http_%d' % error.code, error.description or error.name, error.code)


@app.errorhandler(Exception)
def handle_exception(error):
    """Catch-all: log the details, return a generic message (no internals to the client)."""
    logger.error("Unhandled exception: %s", error, exc_info=True)
    return api_error('internal_error', 'An unexpected server error occurred.', 500)


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def get_models():
    models_to_send = {}
    if BERT_ANONYMIZER_AVAILABLE:
        models_to_send['bert-base-ner'] = 'Local BERT Model (PII Removal)'
    
    return jsonify({
        'available_models': models_to_send,
        'default_model': DEFAULT_MODEL,
        'llm_models': AVAILABLE_MODELS
    })

@app.route('/transcription_models')
def get_transcription_models():
    # Scan MODEL_FOLDER for actual available Whisper models
    model_map = {
        'large': 'faster-whisper-large-v3',
        'large-turbo': 'faster-whisper-large-v3-turbo',
        'medium': 'faster-whisper-medium',
        'small': 'faster-whisper-small',
        'base': 'faster-whisper-base',
        'tiny': 'faster-whisper-tiny',
    }
    
    available_models = {}
    for ui_key, folder_name in model_map.items():
        model_path = MODEL_FOLDER / f"Systran--{folder_name}"
        if model_path.exists():
            available_models[ui_key] = model_map[ui_key]  # Store UI key + label
    
    if not available_models:
        logger.warning("No Whisper models found in MODEL_FOLDER!")
        available_models = {'base': 'Base (Fallback)'}
    
    return jsonify({
        'whisper_models': [
            {'key': k, 'label': f'{k.title()}' if k != 'large-turbo' else 'Large Turbo'}
            for k, v in available_models.items()
        ],
        'default_whisper_model': 'large' if 'large' in available_models else 'base'
    })

@app.route('/available_tags')
def get_available_pii_tags():
    """Return all available PII tag categories."""
    try:
        from process import AVAILABLE_TAGS
        
        # Human-readable label mapping
        pii_tag_mapping = {
            'PERSON': { 'label': 'Names', 'desc': 'Person names and identities' },
            'PERSON_EMAIL': { 'label': 'Email Addresses', 'desc': 'Email addresses' },
            'PERSON_SOCIAL_RELATION': { 'label': 'Social Relations', 'desc': 'Family members, relatives' },
            'ORG': { 'label': 'Organizations', 'desc': 'Companies, institutions, organizations' },
            'LOC_CITY': { 'label': 'Cities', 'desc': 'City names' },
            'LOC_COUNTRY': { 'label': 'Countries', 'desc': 'Country names' },
            'LOC_STREET': { 'label': 'Street Addresses', 'desc': 'Street names and numbers' },
            'LOC_ZIP': { 'label': 'Postal Codes', 'desc': 'ZIP/postal codes' },
            'LOC_HOUSENUMBER': { 'label': 'House Numbers', 'desc': 'Building/house numbers' },
            'LOC_OTHER': { 'label': 'Other Locations', 'desc': 'Geographic locations' },
            'DATETIME': { 'label': 'Dates & Times', 'desc': 'Dates, times, timestamps' },
            'DATETIME_AGE': { 'label': 'Ages', 'desc': 'Age information' },
            'CODE': { 'label': 'Codes', 'desc': 'Identifiers and codes' },
            'CODE_PHONE': { 'label': 'Phone Numbers', 'desc': 'Telephone numbers' },
            'CODE_URL': { 'label': 'URLs & Links', 'desc': 'Web addresses and URLs' },
            'PROFESSION': { 'label': 'Professions', 'desc': 'Occupations and job titles' },
            'PRODUCT': { 'label': 'Products', 'desc': 'Commercial products' },
            'QUANTITY': { 'label': 'Quantities', 'desc': 'Measurements and amounts' },
            'MISC': { 'label': 'Miscellaneous', 'desc': 'Other identifiable information' }
        }
        
        return jsonify({
            'tags': AVAILABLE_TAGS,
            'tag_labels': pii_tag_mapping
        })
        
    except Exception as e:
        logger.error(f"Error getting PII tags: {e}")
        return jsonify({'error': f'Error loading tags: {str(e)}'}), 500

_TRANSCRIPTION_STATUS = {
    'invalid_language': 400, 'invalid_model': 400, 'unreadable_media': 400, 'no_audio_stream': 400,
    'unsupported_audio': 400, 'unsupported_format': 415, 'media_too_long': 413,
    'model_unavailable': 503, 'model_load_failed': 503, 'busy': 503,
}


def _transcribe_and_store(path, language, model, display_name):
    """Transcribes ``path`` and stores audio + word timeline as an encrypted job.

    Failure is decided by ``TranscriptionResult.ok`` only - never by looking at the
    transcript text, so a conversation containing "failed" or "Error:" is fine.
    """
    result = transcribe_audio_result(path, language=language, whisper_model_name=model)
    if not result.ok:
        return api_error(result.error_code, result.error,
                         _TRANSCRIPTION_STATUS.get(result.error_code, 500))
    warnings = list(result.warnings)
    job_id = token = None
    try:
        job, token = JOB_STORE.create(path, result.segments, result.duration or 0.0, request_owner())
        job_id = job.id
    except (OSError, StorageError):
        logger.exception("Could not store the job; redaction will be unavailable for this upload")
        warnings.append('job_storage_failed')
    return jsonify({
        'success': True,
        'transcription': result.text,
        'offsets': result.segments,                 # word-level timeline (legacy key; prefer job_id)
        'audio_name': display_name,                 # client-supplied name only, never a server path
        'job_id': job_id, 'job_token': token,
        'job_expires_in': int(JOB_STORE.idle_ttl),
        'duration': result.duration, 'language': result.language, 'warnings': warnings,
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Transcribes an uploaded audio/video file.

    Form fields: ``audio`` (file), ``language`` (``auto`` or e.g. ``DE``/``de``), and
    ``model_size`` (UI) or ``whisper_model`` (legacy) - ``tiny`` ... ``large``.
    """
    file = request.files.get('audio')
    if file is None or not file.filename:
        raise ApiError('no_file', 'No audio file provided.')
    if not allowed_file(file.filename):
        raise ApiError('invalid_format', 'Invalid file format.', 415)
    ensure_free_space(get_work_dir(), request.content_length or 0)
    ext = file.filename.rsplit('.', 1)[1].lower()
    with work_path('.' + ext) as tmp:
        file.save(tmp)
        return _transcribe_and_store(tmp, request.form.get('language'),
                                     request.form.get('model_size') or request.form.get('whisper_model'),
                                     secure_filename(file.filename))


@app.route('/transcribe_recording', methods=['POST'])
def transcribe_recording():
    """Transcribes a browser recording (WebM). Same response as ``/upload``."""
    blob = request.files.get('audio')
    if blob is None:
        raise ApiError('no_file', 'No audio recording provided.')
    ensure_free_space(get_work_dir(), request.content_length or 0)
    with work_path('.webm') as raw_path, work_path('.wav') as wav_path:
        blob.save(raw_path)
        if os.path.getsize(raw_path) == 0:
            raise ApiError('empty_recording', 'Empty audio file saved.')
        # MediaRecorder WebM has no duration header; a 16 kHz mono WAV is exact and cheap.
        try:
            subprocess.run([FFMPEG, '-nostdin', '-hide_banner', '-loglevel', 'error',
                            '-protocol_whitelist', 'file,pipe', '-i', 'file:' + raw_path,
                            '-vn', '-map_metadata', '-1', '-ar', '16000', '-ac', '1', '-y', wav_path],
                           check=True, capture_output=True, timeout=600)
        except (subprocess.SubprocessError, OSError):
            raise ApiError('unreadable_media', 'The recording could not be decoded.')
        return _transcribe_and_store(wav_path, request.form.get('language'),
                                     request.form.get('model_size') or request.form.get('whisper_model'),
                                     'recording.wav')


@app.route('/surrogate_text', methods=['POST'])
def surrogate_text():
    """Replaces ``[TAG]`` placeholders with realistic fake values.

    JSON body: ``text``, optional ``lang`` (``DE``/``de``/``auto``...), ``session_id`` or
    ``job_id`` (enables the same fake for the same original), optional ``seed``.
    Without a (still valid) session the text is still processed, just without
    cross-mention consistency (``warnings`` contains ``no_entity_map``).
    """
    data = _json_body()
    text = data.get('text')
    if not isinstance(text, str) or not text.strip():
        raise ApiError('no_text', 'No text provided.')
    seed = data.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ApiError('invalid_seed', 'seed must be an integer.')
    sid = data.get('session_id') or data.get('job_id')
    entry = JOB_STORE.recall(sid) if isinstance(sid, str) and ID_RE.fullmatch(sid) else None
    result = apply_surrogates(text, (entry or {}).get('entity_map'), lang=data.get('lang'), seed=seed)
    warnings = list(result.warnings) + ([] if entry else ['no_entity_map'])
    return jsonify({'success': True, 'surrogated_text': result.text,
                    'entities_replaced': result.replaced, 'surrogate_registry': result.registry,
                    'warnings': warnings, 'backend': result.backend})


@app.route('/rephrase_text', methods=['POST'])
def rephrase_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        la = data.get('lang') or 'EN'

        p_text = generate_paraphrase(text, la)

        return jsonify({
            'success': True, 
            'rephrased_text': p_text,
            'tts_available': TTS_AVAILABLE,
            'tts_backend': TTS_BACKEND,
        })
    
    except Exception as e:
        logger.error(f"Error in anonymize route: {str(e)}")
        return jsonify({'error': f'Error anonymizing text: {str(e)}'}), 500
        
    
@app.route('/get_redaction_segments', methods=['POST'])
def get_redaction_segments():
    """Computes the audio spans that correspond to anonymized words.

    JSON body (job mode): ``job_id`` (+ ``X-Job-Token``), optional ``text`` and ``options``
    (``pad_s``, ``merge_gap_s``, ``min_span_s``, ``mode``, ``include_deletions``).
    The spans are derived from the *placeholder* text - ``text`` if it contains ``[TAG]``
    placeholders, otherwise the text stored by ``/anonymize`` - so later Surrogate/Rephrase
    edits of the visible text never change what gets masked.

    Legacy stateless mode: ``{text, offSet}`` with word-level segments in ``offSet``.

    Returns ``{success, segments:[{id,start,end,word,tag,source}], duration, warnings,
    used_text, schema_version, options}``.
    """
    data = _json_body()
    options = rd.RedactionOptions.from_mapping(data.get('options'))
    text = data.get('text') if isinstance(data.get('text'), str) else ''
    if data.get('job_id'):
        job = require_job(data)
        segments, duration, used = JOB_STORE.load_segments(job), job.duration, 'request'
        if not rd.has_placeholders(text):
            text = (JOB_STORE.recall(job.id) or {}).get('placeholder_text') or ''
            used = 'session'
        if not text:
            raise ApiError('no_placeholder_text',
                           'Run the anonymization first (the session may have expired).', 409)
    else:
        segments, duration, used = data.get('offSet'), None, 'request'
        if not isinstance(segments, list) or len(segments) > 200000:
            raise ApiError('invalid_offsets', 'offSet must be a list of word-level segments.')
    words, word_warnings = rd.words_from_segments(segments)
    derived = rd.derive_spans(words, text, options, duration)
    return jsonify({'success': True, 'segments': [s.to_dict() for s in derived.spans],
                    'duration': duration, 'warnings': word_warnings + derived.warnings,
                    'used_text': used, 'schema_version': rd.SCHEMA_VERSION,
                    'options': options.to_dict()})


def _float_arg(name):
    """Optional float query parameter (``None`` if absent)."""
    value = request.args.get(name)
    if value in (None, ''):
        return None
    try:
        number = float(value)
    except ValueError:
        raise ApiError('invalid_parameter', '%s must be a number.' % name)
    if number != number or number in (float('inf'), float('-inf')):
        raise ApiError('invalid_parameter', '%s must be finite.' % name)
    return number


@app.route('/waveform', methods=['GET'])
def waveform():
    """RMS levels for the editor. Query: ``job_id``, ``bars`` (50-4000), ``start``, ``end``;
    the token travels in the ``X-Job-Token`` header."""
    job = require_job()
    try:
        bars = int(request.args.get('bars', 800))
    except ValueError:
        raise ApiError('invalid_parameter', 'bars must be an integer.')
    with JOB_STORE.open_audio(job) as path:
        data = compute_waveform(path, bars=bars, start=_float_arg('start'), end=_float_arg('end'))
    return jsonify(dict(data, success=True))


@app.route('/generate_org_audio', methods=['POST'])
def generate_org_audio_route():
    """Renders the redacted audio.

    JSON body: ``job_id`` (+ ``X-Job-Token``), ``segments`` (``[{start,end}]`` in seconds,
    as edited by the user) and optional ``format`` (``wav``/``mp3``/``flac``). The client
    never sends file paths; the audio is read from the encrypted job store.

    Returns ``{success, audio_file, audio_url, expires_in, message, segments_applied,
    duration, format}``. ``audio_url`` is a short-lived signed link.
    """
    data = _json_body()
    job = require_job(data)
    fmt = data.get('format', 'wav')
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ApiError('invalid_format', 'format must be one of %s.' % (SUPPORTED_OUTPUT_FORMATS,))
    spans = rd.sanitize_spans(data.get('segments'), job.duration)
    if not spans:
        raise ApiError('no_segments', 'There is nothing to redact.')
    ensure_free_space(get_work_dir(), int(job.duration * 192000) + (50 << 20))
    with JOB_STORE.open_audio(job) as source, work_path('.' + fmt) as rendered:
        info = render_redacted_audio(source, rendered, spans, fmt=fmt)
        JOB_STORE.store_redacted(job, rendered, fmt)
    exp, sig = JOB_STORE.sign(job.id, 'redacted', SIGNED_URL_TTL_S)
    return jsonify({
        'success': True, 'audio_file': 'redacted.' + fmt,
        'audio_url': '/download_redacted/%s?exp=%d&sig=%s' % (job.id, exp, sig),
        'expires_in': SIGNED_URL_TTL_S, 'message': 'Redacted audio generated successfully!',
        'segments_applied': info['spans_applied'], 'duration': info['duration'], 'format': fmt,
    })


@app.route('/download_redacted/<job_id>', methods=['GET'])
def download_redacted(job_id):
    """Streams the redacted audio. Authorised by a short-lived signed link (``exp``, ``sig``)
    so that ``<audio src>`` and plain links work without custom headers."""
    if not JOB_STORE.verify(job_id, 'redacted', request.args.get('exp'), request.args.get('sig')):
        return api_error('invalid_link', 'This download link is invalid or has expired.', 403)
    job = JOB_STORE.get_by_id(job_id)
    scope = JOB_STORE.open_redacted(job)
    path, mimetype = scope.__enter__()
    try:
        response = send_file(path, mimetype=mimetype, conditional=True,
                             as_attachment=request.args.get('download') == '1',
                             download_name='redacted.' + (job.meta.get('redacted_format') or 'wav'))
    except BaseException:
        scope.__exit__(None, None, None)
        raise
    response.call_on_close(lambda: scope.__exit__(None, None, None))   # delete the plaintext copy
    return response


@app.route('/job/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Deletes a job immediately (its key is destroyed, so the data is unrecoverable)."""
    job = require_job(job_id=job_id)
    JOB_STORE.delete(job.id)
    return jsonify({'success': True})


@app.route('/generate_speech', methods=['POST'])
def generate_speech_api():
    """Generate speech from text with optional multi-speaker randomization."""
    try:
        from pipeline.tts.tts_engine import generate_multi_speaker_tts
        
        data = request.get_json()
        text = data.get('text', '').strip()
        lang = data.get('lang', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"🎵 Speech request: {len(text)} chars")
        
        output_path, voice_mapping = generate_multi_speaker_tts(
            transcript_text=text,
            word_offsets=[],
            audio_source_path=None
        )
        
        # ✅ CRITICAL VERIFICATION
        if output_path is None:
            logger.error("TTS returned None output path")
            return jsonify({'error': 'TTS returned no output path'}), 500
        
        if not os.path.exists(output_path):
            logger.error(f"TTS file not found: {output_path}")
            return jsonify({'error': 'Generated file does not exist'}), 500
        
        file_size = os.path.getsize(output_path)
        seconds = _wav_content_seconds(output_path)
        if seconds < MIN_TTS_WAV_SECONDS:
            from pipeline.tts.tts_engine import get_tts_status
            tts_status = get_tts_status()
            logger.error("TTS produced %.2f s of audio for %d chars (file %d bytes). "
                         "Backend: %s, status: %s", seconds, len(text),
                         os.path.getsize(output_path), TTS_BACKEND, tts_status)
            return jsonify({
                'error': 'TTS produced an audio file with no usable content (%.2f s). '
                         'Check /tts/status and the server log.' % seconds,
                'tts_status': tts_status
            }), 500
        
        logger.info(f"✅ Generated speech: {file_size} bytes")
        
        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'tts_{int(time.time())}.wav'
        )
        
    except Exception as e:
        logger.error(f"Speech generation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/available_voices')
def get_available_voices():
    try:
        language = request.args.get('language', 'en').lower()[:2]
        voices = get_available_tts_voices(language)
        
        # Enhance with additional metadata
        for voice in voices:
            if TTS_BACKEND == 'piper':
                voice['quality'] = voice.get('id', '').split('-')[-1] if '-' in voice.get('id', '') else 'medium'
                voice['offline'] = True
            elif TTS_BACKEND == 'coqui_xtts':
                voice['cloning_supported'] = True
                voice['offline'] = True
                
        return jsonify({
            'voices': voices, 
            'tts_available': TTS_AVAILABLE,
            'tts_backend': TTS_BACKEND,
            'backend_version': 'v2.0' if TTS_BACKEND == 'coqui_xtts' else 'v1.4.2'
        })
    except Exception as e:
        return jsonify({'error': f'Error getting voices: {str(e)}'}), 500

@app.route('/health')
def health_check():
    try:
        tts_status = get_tts_status()
        return jsonify({
            'status': 'healthy',
            'transcription_available': True,
            'transcription_type': 'shared-engine',
            'speaker_diarization': True,
            'whisperx_available': WHISPERX_AVAILABLE,
            'chat_ai_configured': CHAT_AI_API_KEY is not None,
            'default_model': DEFAULT_MODEL,
            'available_models': list(AVAILABLE_MODELS.keys()),
            'tts_available': TTS_AVAILABLE,
            'tts_backend': tts_status.get('backend', TTS_BACKEND),
            'tts_enabled': tts_status.get('enabled', TTS_ENABLED),
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

def create_self_signed_cert():
    """
    Creates self-signed certificates in unified location.
    Respects existing user-provided certificates.
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from datetime import datetime, timezone, timedelta
        import os
        
        force = "--force-regen" in sys.argv
        if SSL_CERT_FILE.exists() and SSL_KEY_FILE.exists():
            if not SSL_MARKER_FILE.exists():
                print(f"INFO: User-provided certs found at {SSL_CERT_DIR}")
                print("      Respecting user certificates (not overwriting)")
                return str(SSL_CERT_FILE), str(SSL_KEY_FILE)
            if not force:
                print(f"INFO: Self-generated certs found at {SSL_CERT_DIR}")
                print("      Will not overwrite (use --force-regen to replace)")
                return str(SSL_CERT_FILE), str(SSL_KEY_FILE)
            print("WARNING: --force-regen flag detected, backing up and regenerating...")
            backup_dir = SSL_CERT_DIR / ".backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(SSL_CERT_FILE, backup_dir / f"server_{timestamp}.crt")
            shutil.copy2(SSL_KEY_FILE, backup_dir / f"server_{timestamp}.key")

        # Generate new certificate
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "DE"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Berlin"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Berlin"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VERANDA Project"),
            x509.NameAttribute(NameOID.COMMON_NAME, "transcriber.cloud.cci.charite.de"),
        ])
        
        now = datetime.now(timezone.utc)
        cert = x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(private_key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now).not_valid_after(now + timedelta(days=365)).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),   # IPs need an IPAddress SAN
                x509.IPAddress(ipaddress.ip_address("::1")),
                x509.DNSName("transcriber.cloud.cci.charite.de"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Save certificates (the private key is created with mode 0600 from the start)
        with open(SSL_CERT_FILE, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        key_fd = os.open(SSL_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(key_fd, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Mark as self-generated
        with open(SSL_MARKER_FILE, "w") as f:
            f.write(f"Generated by ATA on {datetime.now().isoformat()}\n")
        
        # Set permissions
        os.chmod(SSL_KEY_FILE, 0o600)
        os.chmod(SSL_CERT_FILE, 0o644)
        
        logger.info(f"Created self-signed certificate at {SSL_CERT_FILE}")
        logger.info(f"Private key at {SSL_KEY_FILE}")
        
        return str(SSL_CERT_FILE), str(SSL_KEY_FILE)
    
    except ImportError:
        logger.warning("cryptography package not available, using ad-hoc SSL context")
        return None, None

@app.route('/llm_rewrite', methods=['POST'])
def llm_rewrite_route():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        model_key = data.get('model', 'medgemma')  # ← Use 'model' instead of 'lang'
        enabled = data.get('enabled', True)

        if not enabled:
            return jsonify({'error': 'LLM rewriting is disabled'}), 400

        if not CHAT_AI_API_KEY:
            return jsonify({'error': 'LLM API Key not configured on server'}), 500

        rewritten_text, status = call_llm_rewriter(text, model_key)  # Pass model_key directly
        
        if not rewritten_text:
            return jsonify({'error': f'LLM rewrite failed: {status}'}), 500
        
        return jsonify({
            'success': True,
            'rewritten_text': rewritten_text,
            'model_used': model_key
        })
    
    except Exception as e:
        logger.error(f"Error in LLM rewrite route: {str(e)}")
        return jsonify({'error': f'Error rewriting text: {str(e)}'}), 500

# --- NEW ROUTES FOR EDITING & BULK UPLOAD ---

@app.route('/save_transcription', methods=['POST'])
def save_transcription():
    """Keeps an edited transcript in server memory (idle timeout; never on disk).

    Requires ``text`` and optionally ``job_id``/``session_id``. Without a session nothing is
    stored (``persisted: false``) - raw transcripts are never retained server-side without need.
    """
    data = _json_body()
    text = data.get('text')
    if not isinstance(text, str):
        raise ApiError('no_text', 'No text provided.')
    sid = data.get('job_id') or data.get('session_id')
    persisted = False
    if isinstance(sid, str) and ID_RE.fullmatch(sid):
        entry = JOB_STORE.recall(sid) or {}
        entry['edited_text'] = text
        JOB_STORE.remember(sid, entry)
        persisted = True
    return jsonify({'success': True, 'persisted': persisted, 'message': 'Transcription saved'})


# --- Bulk processing (opt-in: ATA_ENABLE_BULK=1) --------------------------------
ZIP_MAX_FILES = int(os.getenv('ATA_ZIP_MAX_FILES', 500))
ZIP_MAX_BYTES = int(os.getenv('ATA_ZIP_MAX_BYTES', 10 * 1024 ** 3))
ZIP_MAX_RATIO = float(os.getenv('ATA_ZIP_MAX_RATIO', 200))
_BULK_EXT = tuple('.' + e for e in ALLOWED_EXTENSIONS)


def _bulk_guard():
    if not BULK_ENABLED:
        raise ApiError('feature_disabled', 'Bulk processing is disabled on this server.', 404)


def _safe_extract_zip(zip_path, dest):
    """Extracts a zip with hard limits (entry count, real uncompressed bytes, ratio, no nesting).

    Members are written to generated names, so archive paths, symlinks and duplicate names
    cannot escape or collide. Sizes are counted while streaming, not taken from the header.
    Returns a list of ``{filename, filepath, size}``.
    """
    out, total = [], 0
    archive_size = max(os.path.getsize(zip_path), 1)
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > ZIP_MAX_FILES:
            raise ApiError('zip_rejected', 'The archive contains too many files.', 413)
        for info in infos:
            name = os.path.basename(info.filename)
            if name.lower().endswith(('.zip', '.gz', '.tgz', '.tar', '.7z', '.rar')):
                raise ApiError('zip_rejected', 'Nested archives are not allowed.', 400)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:        # symlink entry
                continue
            if not name.lower().endswith(_BULK_EXT):
                continue
            target = os.path.join(dest, uuid.uuid4().hex + os.path.splitext(name)[1].lower())
            with zf.open(info) as src, open(target, 'wb') as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > ZIP_MAX_BYTES or total / archive_size > ZIP_MAX_RATIO:
                        raise ApiError('zip_rejected', 'The archive expands beyond the allowed size.', 413)
                    dst.write(chunk)
            out.append({'filename': secure_filename(name), 'filepath': target,
                        'size': os.path.getsize(target)})
    return out


@app.route('/upload_bulk', methods=['POST'])
def upload_bulk():
    """Registers a batch of audio files or one zip for later transcription (opt-in feature)."""
    _bulk_guard()
    files = request.files.getlist('files')
    if not files or not files[0].filename:
        raise ApiError('no_file', 'No files selected.')
    ensure_free_space(UPLOAD_ROOT, request.content_length or 0)
    batch_id = uuid.uuid4().hex
    batch_dir = ensure_private_dir(UPLOAD_ROOT / 'batches' / batch_id)
    audio_files, text_files = [], []
    try:
        for file in files:
            name = secure_filename(file.filename or '')
            lower = name.lower()
            if lower.endswith('.zip'):
                zip_path = os.path.join(batch_dir, uuid.uuid4().hex + '.zip')
                file.save(zip_path)
                try:
                    audio_files.extend(_safe_extract_zip(zip_path, batch_dir))
                except zipfile.BadZipFile:
                    raise ApiError('zip_rejected', 'Not a valid zip file.')
                finally:
                    os.unlink(zip_path)
            elif lower.endswith(_BULK_EXT):
                target = os.path.join(batch_dir, uuid.uuid4().hex + os.path.splitext(lower)[1])
                file.save(target)
                audio_files.append({'filename': name, 'filepath': target, 'size': os.path.getsize(target)})
            elif lower.endswith('.txt'):
                text_files.append(name)
    except BaseException:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise
    app.bulk_batches[batch_id] = {'created': time.time(), 'audio_files': audio_files,
                                  'text_files': text_files, 'processed': False, 'results': [],
                                  'dir': str(batch_dir)}
    return jsonify({'success': True, 'processed_count': len(audio_files) + len(text_files),
                    'audio_count': len(audio_files), 'text_count': len(text_files),
                    'batch_id': batch_id,
                    'message': '%d file(s) uploaded. Audio files registered for transcription.'
                               % (len(audio_files) + len(text_files))})


@app.route('/list_batches', methods=['GET'])
def list_batches():
    """Lists uploaded batches (opt-in feature)."""
    _bulk_guard()
    return jsonify({'batches': [
        {'batch_id': bid, 'created': b['created'], 'audio_count': len(b['audio_files']),
         'text_count': len(b['text_files']), 'processed': b.get('processed', False),
         'result_count': len(b.get('results', []))} for bid, b in app.bulk_batches.items()]})


def _transcribe_single_file(audio_file, language=None):
    """Transcribes one batch member; failure is read from the structured result only."""
    result = transcribe_audio_result(audio_file['filepath'], language=language)
    if not result.ok:
        return {'filename': audio_file['filename'], 'status': 'failed',
                'error': result.error, 'code': result.error_code}
    return {'filename': audio_file['filename'], 'status': 'success',
            'transcription': result.text, 'offsets': result.segments,
            'warnings': result.warnings}


def _transcribe_batch(batch_id, language=None):
    """Transcribes all files of a batch sequentially."""
    batch = app.bulk_batches.get(batch_id)
    if batch is None:
        raise ApiError('not_found', 'Batch not found.', 404)
    results = [_transcribe_single_file(f, language) for f in batch['audio_files']]
    batch['processed'], batch['results'] = True, results
    ok = sum(1 for r in results if r['status'] == 'success')
    return {'success': True, 'batch_id': batch_id, 'success_count': ok,
            'failed_count': len(results) - ok, 'results': results}


@app.route('/process_batch/<batch_id>', methods=['POST'])
def process_batch(batch_id):
    """Transcribes all audio files in a batch (opt-in feature)."""
    _bulk_guard()
    if not ID_RE.fullmatch(batch_id):
        raise ApiError('not_found', 'Batch not found.', 404)
    return jsonify(_transcribe_batch(batch_id, request.form.get('language')))


@app.route('/process_all_batches', methods=['POST'])
def process_all_batches():
    """Transcribes every unprocessed batch (opt-in feature)."""
    _bulk_guard()
    language = request.form.get('language')
    pending = [b for b, d in app.bulk_batches.items() if not d.get('processed')]
    results = {b: _transcribe_batch(b, language) for b in pending}
    return jsonify({'success': True, 'batches_processed': len(pending),
                    'total_success': sum(r['success_count'] for r in results.values()),
                    'total_failed': sum(r['failed_count'] for r in results.values()),
                    'results': results})


def _purge_batches(now=None):
    """Deletes bulk batches older than the job lifetime. Returns the number removed."""
    now = time.time() if now is None else now
    stale = [b for b, d in app.bulk_batches.items() if now - d['created'] > JOB_STORE.max_age]
    for batch_id in stale:
        shutil.rmtree(app.bulk_batches.pop(batch_id).get('dir', ''), ignore_errors=True)
    return len(stale)


if BULK_ENABLED:
    def _batch_sweeper():
        while True:
            time.sleep(300)
            _purge_batches()
    threading.Thread(target=_batch_sweeper, name='ata-batch-purge', daemon=True).start()


@app.route('/download_text/<file_type>/<filename>')
def download_text(file_type, filename):
    """Downloads text files (original, anonymized, LLM). Disabled unless ATA_ENABLE_FILE_DOWNLOADS=1."""
    if not FILE_DOWNLOADS_ENABLED:
        return api_error('feature_disabled', 'File downloads are disabled on this server.', 404)
    try:
        from process import ANONYM_FOLDER, LLM_ANONYM_FOLDER
        
        # ✅ FIX: Use absolute paths from process.py
        folder_map = {
            'original': app.config['UPLOAD_FOLDER'],
            'bert': str(ANONYM_FOLDER.absolute()),
            'llm': str(LLM_ANONYM_FOLDER.absolute())
        }
        
        if file_type not in folder_map:
            return jsonify({'error': 'Invalid file type'}), 400
        
        base_path = folder_map[file_type]
        safe_filename = secure_filename(filename)
        file_path = os.path.join(base_path, safe_filename)
        
        # Verify file exists within allowed directory
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            # Debug: Return what folders actually exist
            return jsonify({
                'error': 'File not found',
                'file_path': file_path,
                'exists': os.path.exists(file_path),
                'base_path': base_path,
                'available_folders': {
                    'uploads': list(os.listdir(app.config['UPLOAD_FOLDER'])),
                    'anonym': list(os.listdir(ANONYM_FOLDER)) if ANONYM_FOLDER.exists() else []
                }
            }), 404
        
        return send_file(file_path, as_attachment=True, download_name=safe_filename)
    except Exception as e:
        logger.error(f"Error downloading text: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_speech/<filename>')
def download_speech_route(filename):
    """Wrapper for existing speech download with security check"""
    try:
        if not filename.startswith('speech_output_') or '..' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        logger.error(f"Error downloading speech: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tts/status')
def get_tts_status_route():
    """Return current TTS configuration."""
    status = get_tts_status()
    # Add backend-specific info
    if TTS_BACKEND == 'piper':
        status['backend_details'] = 'Piper TTS (offline neural TTS)'
        status['supported_languages'] = ['en', 'de', 'fr', 'es', 'it', 'pl', 'pt', 'fi', 'ar', 'hi', 'tr']
    elif TTS_BACKEND == 'coqui_xtts':
        status['backend_details'] = 'Coqui XTTS v2 (voice cloning, 17 langs)'
        status['supported_languages'] = ['en', 'de', 'fr', 'es', 'it', 'pl', 'pt', 'zh', 'ja', 'ko', 'ar', 'hi', 'tr', 'fi']
        status['license_notice'] = 'CPML License: Non-commercial use only'
    return jsonify(status)


@app.route('/tts/generate_beep', methods=['POST'])
def generate_beep_route():
    """Generate a beep sound using ffmpeg, streaming directly."""
    try:
        import io
        
        data = request.get_json() or {}
        duration_ms = data.get('duration_ms', 400)
        freq = data.get('freq', 1000)
        
        # Create in-memory buffer
        buffer = io.BytesIO()
        
        # Generate beep (pipe to stdout instead of file)
        proc = subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i',
            f'sine=frequency={freq}:duration={duration_ms/1000}',
            '-c:a', 'libmp3lame',
            '-'  # Output to stdout
        ], capture_output=True, check=True)
        
        buffer.write(proc.stdout)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='beep.mp3'
        )
        
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Beep generation failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/debug/sessions')
def debug_session_count():
    """Counts only. Disabled (404) unless ATA_DEBUG_TOKEN is set and sent as X-Debug-Token."""
    expected = os.getenv('ATA_DEBUG_TOKEN')
    given = request.headers.get('X-Debug-Token', '')
    if not expected or not hmac.compare_digest(given.encode(), expected.encode()):
        return api_error('not_found', 'Not found.', 404)
    return jsonify(JOB_STORE.stats())

# Start background cleaner (non-blocking)

if __name__ == '__main__':
    use_https = os.getenv('USE_HTTPS', 'true').lower() == 'true'
    
    if use_https:
        cert_to_use = None
        key_to_use = None
        
        # Priority 1: Check if Certbot/production cert exists
        if os.path.exists(FLASK_SSL_CERT) and os.path.exists(FLASK_SSL_KEY):
            cert_to_use = FLASK_SSL_CERT
            key_to_use = FLASK_SSL_KEY
            logger.info(f"Using production certificate from config.sh")
            logger.info(f"  Domain:   {DOMAIN_NAME}")
            logger.info(f"  SSL cert: {cert_to_use}")
            logger.info(f"  SSL key:  {key_to_use}")
        
        # Priority 2: Fallback to self-signed if Certbot cert not found
        else:
            logger.warning(f"Production certificate not found at {FLASK_SSL_CERT}")
            logger.warning("Falling back to self-signed certificate (development mode)")
            
            cert_to_use, key_to_use = create_self_signed_cert()
            
            if not cert_to_use or not os.path.exists(cert_to_use):
                logger.error("CRITICAL: Self-signed certificate generation failed!")
                logger.error("Cannot start Flask without valid SSL certificate.")
                sys.exit(1)
            
            logger.info(f"  SSL cert: {cert_to_use}")
            logger.info(f"  SSL key:  {key_to_use}")
        
        # Verify files are readable before starting
        if not os.access(cert_to_use, os.R_OK) or not os.access(key_to_use, os.R_OK):
            logger.error(f"Certificate files not readable!")
            logger.error(f"  Cert: {cert_to_use}")
            logger.error(f"  Key:  {key_to_use}")
            sys.exit(1)
        
        logger.info(f"Starting Flask with HTTPS on {FLASK_BIND_HOST}:{FLASK_BIND_PORT}")
        
        app.run(
            debug=False,
            host=FLASK_BIND_HOST,
            port=FLASK_BIND_PORT,
            ssl_context=(cert_to_use, key_to_use),
            threaded=True
        )
    else:
        logger.warning("Starting Flask with HTTP (USE_HTTPS=false) — NOT recommended for medical data")
        app.run(debug=False, host=FLASK_BIND_HOST, port=FLASK_BIND_PORT, threaded=True)