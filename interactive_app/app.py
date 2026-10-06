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
from audio_utils import AudioBeepReplacer
from io import BytesIO
from dotenv import load_dotenv
import json
import requests


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
        AnonymizationEngine
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

# Import Anonymization engine
global_anonymizer = None
try:
    from process import AnonymizationEngine, MODEL_FOLDER
    global_anonymizer = AnonymizationEngine(
        method="local_mmbert",
        level="standard",
        model_path=MODEL_FOLDER / "multilingual_DialogPII_NER"
    )
    logger.info(f"✅ AnonymizationEngine pre-loaded successfully. Method: {global_anonymizer.method}")
except Exception as e:
    logger.error(f"⚠️ Failed to pre-load AnonymizationEngine: {e}")
    global_anonymizer = None

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
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Store batch metadata
app.bulk_batches = {}

# --- Model Pre-loading ---
try:
    load_models()
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

SURROGATES = {
    "EN": {
        "PERSON": [
            "John Smith", "Emma Johnson", "Michael Brown", "Sophia Miller",
            "Daniel Wilson", "Olivia Moore", "James Taylor", "Emily Davis",
            "Benjamin Clark", "Charlotte White", "Henry Walker", "Mia Harris",
            "Alexander Hall", "Amelia Young", "David Allen", "Ella King",
            "Joseph Wright", "Grace Scott", "Samuel Green", "Lily Baker"
        ],
        "CITY": [
            "Berlin", "London", "New York", "Chicago", "Boston",
            "Seattle", "Munich", "Hamburg", "Paris", "Vienna",
            "Toronto", "Dublin", "Leeds", "Bristol", "Manchester",
            "Frankfurt", "Cologne", "Zurich", "Amsterdam", "Prague"
        ],
        "ZIP": [
            "10001", "20095", "75008", "10115", "50667",
            "80331", "SW1A1AA", "94105", "60601", "33101",
            "70173", "01067", "28195", "4000", "8001",
            "1010", "2000", "80333", "04109", "90402"
        ],
        "AGE": [
            "21", "24", "27", "30", "33",
            "36", "39", "42", "45", "48",
            "51", "54", "57", "60", "63",
            "66", "69", "72", "75", "78"
        ],
        "EMAIL": [
            "john@example.com", "emma@test.com", "michael@mail.com",
            "sophia@demo.org", "daniel@company.net", "olivia@example.org",
            "james@sample.com", "emily@test.org", "ben@demo.net",
            "charlotte@mail.org", "henry@example.net", "mia@test.com",
            "alex@sample.org", "amelia@demo.com", "david@mail.net",
            "ella@example.org", "joseph@test.net", "grace@sample.com",
            "samuel@demo.org", "lily@mail.com"
        ],
        "PHONE": [
            "+1 202 555 0101", "+1 202 555 0102", "+1 202 555 0103",
            "+44 20 7946 0001", "+44 20 7946 0002",
            "+49 30 123456", "+49 40 987654",
            "+33 1 23456789", "+41 44 1234567", "+43 1 987654",
            "+1 303 555 1212", "+1 404 555 2323", "+49 89 456789",
            "+44 161 555 1000", "+33 4 11111111",
            "+1 212 555 8888", "+49 221 987654",
            "+43 662 123456", "+41 31 7654321", "+1 617 555 0909"
        ],
        "PROFESSION": [
            "doctor", "teacher", "engineer", "lawyer", "designer",
            "developer", "nurse", "scientist", "manager", "architect",
            "chef", "journalist", "consultant", "photographer", "pilot",
            "researcher", "pharmacist", "electrician", "mechanic", "writer"
        ],
        "ORGANISATION": [
	    "Acme Corporation",
	    "Global Tech Solutions",
	    "Green Valley Hospital",
	    "Sunrise Medical Center",
	    "Northbridge University",
	    "Blue River Consulting",
	    "United Logistics Group",
	    "Pioneer Software Ltd.",
	    "Metro Insurance Services",
	    "Future Energy Systems",
	    "City Health Network",
	    "Bright Education Trust",
	    "National Research Institute",
	    "Western Manufacturing Inc.",
	    "Community Care Foundation",
	    "Summit Financial Partners",
	    "Digital Innovation Labs",
	    "Central Public Library",
	    "Evergreen Construction",
	    "International Trade Association"
	],
        "URL": [
            "https://example.com", "https://test.org", "https://demo.net",
            "https://sample.com", "https://mywebsite.org",
            "https://company.net", "https://homepage.com",
            "https://service.org", "https://product.net",
            "https://info.com", "https://data.org",
            "https://project.net", "https://alpha.com",
            "https://beta.org", "https://gamma.net",
            "https://delta.com", "https://epsilon.org",
            "https://zeta.net", "https://theta.com",
            "https://lambda.org"
        ],
        "PRODUCT": [
            "iPhone", "ThinkPad", "Galaxy Tablet", "MacBook",
            "Surface Pro", "PlayStation", "AirPods", "Kindle",
            "GoPro", "Fitbit", "Nikon Camera", "Dell Monitor",
            "HP Printer", "Canon Lens", "Dyson Vacuum",
            "Sony Headphones", "Bose Speaker", "Apple Watch",
            "Nintendo Switch", "Pixel Phone"
        ],
        "STREET": [
            "Main Street", "Oak Avenue", "Maple Road", "Pine Street",
            "Cedar Lane", "Elm Street", "Washington Avenue",
            "Lakeview Drive", "Hillcrest Road", "Sunset Boulevard",
            "Park Avenue", "River Road", "King Street",
            "Queen Street", "Church Lane", "Station Road",
            "Mill Road", "Victoria Street", "Bridge Street", "High Street"
        ]
    },

    "DE": {
        "PERSON": [
            "Max Müller", "Anna Schmidt", "Peter Weber", "Laura Fischer",
            "Thomas Wagner", "Julia Becker", "Lukas Hoffmann", "Sarah Koch",
            "Felix Bauer", "Leonie Richter", "Jonas Klein", "Marie Wolf",
            "Paul Schröder", "Lisa Neumann", "Tim Braun", "Nina Hartmann",
            "David Lange", "Sophie Krüger", "Jan Meier", "Eva Schulz"
        ],
        "CITY": [
            "Berlin", "Hamburg", "München", "Köln", "Frankfurt",
            "Stuttgart", "Dresden", "Leipzig", "Bremen", "Hannover",
            "Düsseldorf", "Dortmund", "Essen", "Bonn", "Mannheim",
            "Nürnberg", "Augsburg", "Karlsruhe", "Potsdam", "Freiburg"
        ],
        "ZIP": [
            "10115", "20095", "80331", "50667", "60311",
            "70173", "01067", "04109", "28195", "30159",
            "40213", "44135", "45127", "53111", "68159",
            "90402", "86150", "76133", "14467", "79098"
        ],
        "AGE": [
            "18", "22", "25", "28", "31",
            "34", "37", "40", "43", "46",
            "49", "52", "55", "58", "61",
            "64", "67", "70", "73", "76"
        ],
        "EMAIL": [
            "max@beispiel.de", "anna@test.de", "peter@mail.de",
            "laura@firma.de", "thomas@demo.de", "julia@beispiel.org",
            "lukas@test.org", "sarah@mail.org", "felix@demo.net",
            "leonie@firma.com", "jonas@test.com", "marie@example.de",
            "paul@beispiel.com", "lisa@demo.org", "tim@mail.net",
            "nina@test.net", "david@firma.org", "sophie@example.com",
            "jan@beispiel.net", "eva@test.de"
        ],
        "PHONE": [
            "+49 30 123456", "+49 40 987654", "+49 89 456789",
            "+49 221 111111", "+49 711 222222",
            "+49 351 333333", "+49 341 444444", "+49 421 555555",
            "+49 511 666666", "+49 211 777777",
            "+49 231 888888", "+49 201 999999",
            "+49 228 123123", "+49 621 321321",
            "+49 911 456456", "+49 821 654654",
            "+49 761 789789", "+49 331 987987",
            "+49 69 135791", "+49 731 246810"
        ],
        "PROFESSION": [
            "Arzt", "Lehrer", "Ingenieur", "Anwalt", "Designer",
            "Entwickler", "Krankenpfleger", "Wissenschaftler", "Manager",
            "Architekt", "Koch", "Journalist", "Berater", "Fotograf",
            "Pilot", "Forscher", "Apotheker", "Elektriker",
            "Mechaniker", "Schriftsteller"
        ],
        "ORGANISATION": [
	    "Universitätsklinikum Berlin",
	    "Technische Universität München",
	    "Stadtwerke Hamburg",
	    "Muster GmbH",
	    "Beispiel AG",
	    "Forschungszentrum Leipzig",
	    "Klinikum Stuttgart",
	    "Deutsches Institut für Informatik",
	    "Berliner Verkehrsbetriebe",
	    "Münchner Versicherungsgruppe",
	    "Norddeutsche Logistik GmbH",
	    "Gesundheitszentrum Köln",
	    "Innovationslabor Dresden",
	    "Rhein-Main Consulting",
	    "Bildungswerk Frankfurt",
	    "Sozialverband Deutschland",
	    "Energieversorgung Bayern",
	    "MediCare Krankenhausverbund",
	    "Industrieverband Nordrhein",
	    "Wissenschaftsakademie Freiburg"
	],
        "URL": [
            "https://beispiel.de", "https://test.org", "https://demo.net",
            "https://firma.de", "https://webseite.org",
            "https://projekt.net", "https://daten.de",
            "https://service.org", "https://produkt.net",
            "https://info.de", "https://alpha.org",
            "https://beta.net", "https://gamma.de",
            "https://delta.org", "https://epsilon.net",
            "https://zeta.de", "https://theta.org",
            "https://lambda.net", "https://omega.de",
            "https://portal.org"
        ],
        "PRODUCT": [
            "iPhone", "ThinkPad", "Galaxy Tablet", "MacBook",
            "Surface Pro", "PlayStation", "AirPods", "Kindle",
            "GoPro", "Fitbit", "Nikon Kamera", "Dell Monitor",
            "HP Drucker", "Canon Objektiv", "Dyson Staubsauger",
            "Sony Kopfhörer", "Bose Lautsprecher", "Apple Watch",
            "Nintendo Switch", "Pixel Smartphone"
        ],
        "STREET": [
            "Hauptstraße", "Bahnhofstraße", "Gartenweg", "Schillerstraße",
            "Goethestraße", "Bergstraße", "Dorfstraße", "Mühlenweg",
            "Kirchstraße", "Lindenweg", "Parkstraße", "Wiesenweg",
            "Waldstraße", "Ringstraße", "Schulstraße", "Mozartstraße",
            "Lessingstraße", "Friedhofsweg", "Birkenweg", "Ahornstraße"
        ]
    }
}

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
    # flatten offsets + build reference string
    offsets = []
    s1_tokens = []

    for item in offs:
        for w in item["words"]:
            word = w["word"]
            s1_tokens.append(word)
            offsets.append([w["start"], w["end"], word.lower()])

    s1 = " ".join(s1_tokens).lower()

    # normalize s2
    s2 = re.sub(r"SPEAKER_[0-9]+:", "", s2)
    s2 = re.sub(r"\n", "", s2).lower()


    w1 = s1.split()
    w2 = s2.split()

    matcher = difflib.SequenceMatcher(None, w1, w2)

    result = [
        {
            "type": tag,
            "s1_words": w1[i1:i2],
            "s1_range": (i1, i2),
            "s2_words": w2[j1:j2],
            "s2_range": (j1, j2),
        }
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]

    time_span_remove = []


    for r in result:

        start_i, end_i = r["s1_range"]

        # end boundary (use next token start if possible)
        if end_i < len(offsets):
            x_en = float(offsets[end_i][0])
        else:
            x_en = float(offsets[-1][0])

        # start boundary (use previous token end if possible)
        if start_i - 1 >= 0:
            x_st = float(offsets[start_i - 1][1])
        else:
            x_st = float(offsets[0][1])

        time_span_remove.append({
            "start": x_st,
            "end": x_en,
            "word": r["s1_words"],
        })

    return time_span_remove
    
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

@app.route('/anonymize', methods=['POST'])
def anonymize():
    try:
        data = request.get_json()
        transcript_text = data.get('transcript', '')
        include_tags = data.get('include_tags', None)
        
        if not transcript_text:
            logger.warning("No transcript provided to /anonymize")
            return jsonify({'error': 'No transcript provided'}), 400
        
        # =========================================
        # INSERT CLEANUP LOGIC HERE (after validation, before save)
        # =========================================
        
        # Check if global_anonymizer was loaded successfully
        if global_anonymizer is None or not global_anonymizer.method:
            # Try to create one on-demand
            try:
                global_anonymizer = AnonymizationEngine(
                    method="local_mmbert",
                    level="standard",
                    model_path=MODEL_FOLDER / "multilingual_DialogPII_NER",
                    include_tags=include_tags
                )
                if not global_anonymizer.method:
                    raise ValueError("Anonymization engine failed to initialize")
            except Exception as e:
                logger.error(f"Failed to initialize AnonymizationEngine: {e}")
                return jsonify({
                    'error': 'Anonymization service unavailable',
                    'details': str(e)
                }), 500
        
        # Call .anonymize() directly (returns 5 values)
        result_text, success, msg, entity_map, surrogate_registry = global_anonymizer.anonymize(
            transcript_text,
            use_surrogates=False
        )
        
        logger.info(f"Anonymization result: success={success}, text_len={len(result_text) if result_text else 0}")
        
        if not success or result_text is None:
            logger.error(f"Anonymization FAILED: {msg}")
            return jsonify({
                'error': 'Anonymization failed',
                'details': msg,
                'original_length': len(transcript_text)
            }), 500
        
        # =========================================
        # SESSION EXPIRATION CLEANUP LOGIC
        # Add this block BEFORE saving the new session
        # =========================================
        MAX_SESSION_AGE_SECONDS = 3600  # 1 hour
        
        if hasattr(app, 'session_entity_maps'):
            current_time = time.time()
            expired_sessions = [
                sid for sid, record in app.session_entity_maps.items()
                if current_time - record.get('timestamp', 0) > MAX_SESSION_AGE_SECONDS
            ]
            for sid in expired_sessions:
                del app.session_entity_maps[sid]
                logger.debug(f"Cleaned up expired session: {sid[:8]}...")
        
        # Now save the new session
        session_id = request.headers.get('X-Session-ID', str(int(time.time())))
        if not hasattr(app, 'session_entity_maps'):
            app.session_entity_maps = {}
        
        app.session_entity_maps[session_id] = {
            'entity_map': entity_map,
            'timestamp': time.time(),
            'original_length': len(transcript_text)
        }
        
        # Rest of the route remains unchanged...
        logger.info(f"Anonymization SUCCESS: {len(transcript_text)} → {len(result_text)} chars")
        logger.info(f"Session ID: {session_id} | Entity map entries: {sum(len(v) for v in entity_map.values())}")
        
        return jsonify({
            'status': 'success',
            'original': transcript_text,
            'anonymized': result_text,
            'chars_original': len(transcript_text),
            'chars_anonymized': len(result_text),
            'session_id': session_id,
            'entity_count': sum(len(v) for v in entity_map.values()),
        })
    
    except TypeError as e:
        logger.error(f"Anonymization tuple unpacking error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server configuration error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Anonymization error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def replace_surrogates(text, lang="EN"):
    lang = lang.upper()

    if lang not in SURROGATES:
        lang = "EN"

    values = SURROGATES[lang]

    # keep track of recently used replacements per tag
    used = {k: [] for k in values}

    def repl(match):
        tag = match.group(1)

        if tag not in values:
            return match.group(0)

        choices = values[tag]

        # try to avoid immediate repetitions
        available = [x for x in choices if x not in used[tag]]

        if not available:
            # reset if everything was already used
            used[tag] = []
            available = choices

        value = random.choice(available)
        used[tag].append(value)

        return value

    return re.sub(r"\[([A-Z_]+)\]", repl, text)
    


# ============================================================================
# GLOBAL ERROR HANDLERS (Return JSON instead of HTML)
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error.description)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error.description)}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error', 'message': 'An unexpected error occurred'}), 500

@app.errorhandler(Exception)
def handle_exception(error):
    """Catch-all for unhandled exceptions."""
    logger.error(f"Unhandled exception: {error}", exc_info=True)
    return jsonify({'error': 'Server error', 'message': str(error)}), 500
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file upload too large errors with JSON response."""
    logger.warning(f"File too large: {request.content_length} bytes")
    return jsonify({
        'error': 'File too large',
        'message': 'The uploaded audio file exceeds the 5 GB size limit.',
        'max_size_gb': 5
    }), 413

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
    # DEBUG: Print to server console
    print(f"DEBUG: WHISPER_MODELS content: {list(WHISPER_MODELS.items())}")
    
    return jsonify({
        'whisper_models': list(WHISPER_MODELS.items()),
        'default_whisper_model': 'base'
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

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        language = request.form.get('language', None)
        if not language or language == 'auto':
            language = None
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Transcribing file: {filename} in language: {language}")
            transcription, wordOffS = transcribe_audio_locally(filepath, language=language)
            
            # Check if transcription resulted in an error message
            if not isinstance(transcription, str):
                logger.error(f"Unexpected transcription type: {type(transcription)}")
                return jsonify({'error': 'Transcription service returned invalid data'}), 500
            
            # Check for error messages
            if transcription.startswith("Error:") or transcription.startswith("Transcription failed:"):
                logger.error(f"Transcription returned error: {transcription}")
                return jsonify({'error': transcription}), 500
            
            # Ensure wordOffS is always a list
            if not isinstance(wordOffS, list):
                wordOffS = []
            
            return jsonify({
                'success': True,
                'transcription': transcription if transcription else "",
                'offsets': wordOffS,
                'audio_name': filepath
            })
        
        return jsonify({'error': 'Invalid file format'}), 400
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in upload: {e}", exc_info=True)
        return jsonify({'error': f'Response parsing failed: {str(e)}'}), 500
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL error during transcription: {e}", exc_info=True)
        return jsonify({'error': 'Transcription service SSL error. Check VPN connection.'}), 503
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during transcription: {e}", exc_info=True)
        return jsonify({'error': 'Transcription service timeout. Please try again.'}), 504
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error during transcription: {e}", exc_info=True)
        return jsonify({'error': 'Transcription service unavailable. Check network connection.'}), 503
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)  # Added exc_info for traceback
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/transcribe_recording', methods=['POST'])
def transcribe_recording():
    temp_file_path = None
    converted_file_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio recording provided'}), 400
        
        audio_blob = request.files['audio']
        language = request.form.get('language', None)
        if not language or language == 'auto':
            language = None
        
        logger.info(f"Received audio file: filename='{audio_blob.filename}', language='{language}'")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            audio_blob.save(temp_file.name)
            temp_file_path = temp_file.name
        
        file_size = os.path.getsize(temp_file_path)
        if file_size == 0:
            return jsonify({'error': 'Empty audio file saved'}), 400
        
        try:
            converted_file_path = temp_file_path.replace('.webm', '.wav')
            subprocess.run([
                'ffmpeg', '-i', temp_file_path, 
                '-ar', '16000', '-ac', '1', '-f', 'wav', '-y',
                converted_file_path
            ], check=True, capture_output=True, text=True)
            
            if os.path.getsize(converted_file_path) > 0:
                audio_file_to_transcribe = converted_file_path
            else:
                audio_file_to_transcribe = temp_file_path
        except Exception as e:
            logger.warning(f"FFmpeg conversion failed: {e}. Using original file.")
            audio_file_to_transcribe = temp_file_path
        
        logger.info(f"Starting transcription of: {audio_file_to_transcribe}")
        transcription, wordOffS = transcribe_audio_locally(audio_file_to_transcribe, language=language)
        
        # Check if transcription resulted in an error message
        if isinstance(transcription, str) and ("Error:" in transcription or "failed" in transcription.lower()):
            logger.error(f"Transcription returned error: {transcription}")
            return jsonify({'error': transcription}), 500
        
        return jsonify({
            'success': True,
            'transcription': transcription if transcription else "",
            'offsets': wordOffS if wordOffS else [],
            'audio_name': audio_file_to_transcribe
        })
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in transcribe_recording: {e}", exc_info=True)
        return jsonify({'error': f'Response parsing failed: {str(e)}'}), 500
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL error during transcription: {e}", exc_info=True)
        return jsonify({'error': 'Transcription service SSL error. Check VPN connection.'}), 503
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during transcription: {e}", exc_info=True)
        return jsonify({'error': 'Transcription service timeout. Please try again.'}), 504
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error during transcription: {e}", exc_info=True)
        return jsonify({'error': 'Transcription service unavailable. Check network connection.'}), 503
    except Exception as e:
        logger.error(f"Error processing recording: {str(e)}", exc_info=True)  # Added exc_info for traceback
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if converted_file_path and os.path.exists(converted_file_path):
            os.unlink(converted_file_path)
        return jsonify({'error': f'Error processing recording: {str(e)}'}), 500

@app.route('/surrogate_text', methods=['POST'])
def surrogate_text():
    """
    Apply surrogate substitution using persisted entity_map from /anonymize.
    STRICT MODE: Requires valid session_id; rejects if session expired/missing.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        lang = (data.get('lang') or 'EN').upper()
        session_id = data.get('session_id')
        seed = data.get('seed')  # Optional reproducibility
        locale = FAKER_LOCALE_MAP.get(lang, 'en_US')

        if not apply_surrogate_substitution or not HAS_FAKER:
            return jsonify({
                'error': 'Faker library not available on server',
                'hint': 'Install with: pip install faker'
            }), 500

        # ==========================================
        # CRITICAL: Validate session_id exists
        # ==========================================
        if not session_id:
            return jsonify({
                'error': 'Missing session_id',
                'hint': 'Call /anonymize first and use the returned session_id here.',
                'required': True
            }), 400

        # Retrieve persisted entity_map
        entity_maps = getattr(app, 'session_entity_maps', {})
        
        if session_id not in entity_maps:
            return jsonify({
                'error': 'Session expired or invalid',
                'hint': 'Run /anonymize again to create a new session.',
                'expired': True
            }), 401

        entity_map_record = entity_maps[session_id]
        entity_map = entity_map_record['entity_map']
        original_length = entity_map_record.get('original_length', 0)
        created_at = entity_map_record.get('timestamp', 0)

        # Optional: Enforce session expiration (e.g., 1 hour)
        MAX_SESSION_AGE_SECONDS = 3600
        if time.time() - created_at > MAX_SESSION_AGE_SECONDS:
            # Cleanup expired session
            del entity_maps[session_id]
            return jsonify({
                'error': 'Session expired',
                'hint': f'Session older than {MAX_SESSION_AGE_SECONDS/60:.0f} minutes. Run /anonymize again.',
                'expired': True,
                'age_seconds': int(time.time() - created_at)
            }), 401

        # Validate entity_map is not empty
        total_entities = sum(len(v) for v in entity_map.values())
        if total_entities == 0:
            return jsonify({
                'error': 'Empty entity map',
                'hint': 'Anonymization produced no entities. No surrogates needed.'
            }), 400

        # Count placeholders in text vs entity_map entries (alignment check)
        placeholder_count = sum(len(v) for v in entity_map.values())
        actual_placeholders = sum(
            len(re.findall(rf'\[{tag}\]', text)) 
            for tag in entity_map.keys()
        )

        alignment_error = abs(placeholder_count - actual_placeholders) > 0
        if alignment_error:
            logger.warning(
                f"Placeholder mismatch: map={placeholder_count}, text={actual_placeholders}. "
                f"User may have edited transcript after anonymization."
            )
            # Still proceed but log warning; decision to abort is yours.
            # Uncomment below to make this a hard failure:
            # return jsonify({
            #     'error': 'Transcript mismatch',
            #     'hint': 'Placeholder count differs from anonymization output. User edits may have broken alignment.',
            #     'expected': placeholder_count,
            #     'found': actual_placeholders
            # }), 400

        # Apply surrogate substitution
        s_text, registry = apply_surrogate_substitution(
            text,
            entity_map,
            use_surrogates=True,
            seed=seed,
            locales=locale
        )

        # Optional: Clean up used sessions to prevent memory bloat
        # del entity_maps[session_id]

        logger.info(
            f"Surrogate substitution complete: {sum(len(v) for v in registry.values())} entities replaced. "
            f"Session: {session_id[:8]}... Engine: faker-consistent"
        )

        return jsonify({
            'success': True,
            'surrogated_text': s_text,
            'entities_replaced': sum(len(v) for v in registry.values()),
            'surrogate_registry': registry,
            'engine': 'faker-consistent',  # Identity guaranteed
            'tts_available': TTS_AVAILABLE,
            'tts_backend': TTS_BACKEND,
        })

    except Exception as e:
        logger.error(f"Error in surrogate_text route: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error generating surrogates: {str(e)}'}), 500

@app.route('/rephrase_text', methods=['POST'])
def rephrase_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        la = data['lang']

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
        
    
@app.route('/generate_org_audio', methods=['POST'])
def generate_org_audio_route():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        offS = data['offSet']
        audio_file_name = data['audioName']

        if not text:
            return jsonify({'error': 'Empty text'}), 400
        
        if not TTS_AVAILABLE:
            return jsonify({'error': 'TTS not available. Enable in .env with TTS_ENABLED=true'}), 500
            
        offsets = diff_relevant_offsets(offS, text)

        output_wav="speech_output_beeped.mp3"

        audio_path = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), output_wav)

        replacer = AudioBeepReplacer()
        output_file = replacer.replace_offsets_with_beeps(audio_file_name,offsets,audio_path)

        if audio_path and os.path.exists(audio_path):
            audio_filename = os.path.basename(audio_path)
            return jsonify({
                'success': True,
                'audio_file': audio_filename,
                'audio_url': f'/download_speech/{audio_filename}',
                'message': 'Speech generated successfully!'
            })
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return jsonify({'error': f'Error generating audio: {str(e)}'}), 500


@app.route('/generate_speech', methods=['POST'])
def generate_speech_api():
    """Generate speech from text with optional multi-speaker randomization."""
    try:
        from pipeline.tts.tts_engine import generate_multi_speaker_tts
        
        data = request.get_json()
        text = data.get('text', '').strip()
        lang = data.get('lang', 'en')
        randomize_voices = data.get('randomize_voices', True)  # Always randomize per spec
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"🎵 Speech request: {len(text)} chars, multi-speaker=True")
        
        # Generate multi-speaker TTS
        output_path, voice_mapping = generate_multi_speaker_tts(
            transcript_text=text,
            word_offsets=[],  # Would pass from earlier transcription if available
            audio_source_path=None
        )
        
        if output_path is None:
            logger.error("Speech generation failed")
            return jsonify({'error': 'Speech generation failed'}), 500
        
        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'multi_speaker_tts_{int(time.time())}.wav'
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
        
        # Check if certificates already exist
        if SSL_CERT_FILE.exists() and SSL_KEY_FILE.exists():
            # Check if self-generated
            if SSL_MARKER_FILE.exists():
                print(f"INFO: Self-generated certs found at {SSL_CERT_DIR}")
                print(f"      Will not overwrite (unless deleted manually)")
                # Optionally backup before regenerating
                if "--force-regen" in sys.argv:
                    print("WARNING: --force-regen flag detected, backing up...")
                    backup_dir = SSL_CERT_DIR / ".backups"
                    backup_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shutil.copy2(SSL_CERT_FILE, backup_dir / f"server_{timestamp}.crt")
                    shutil.copy2(SSL_KEY_FILE, backup_dir / f"server_{timestamp}.key")
            else:
                print(f"INFO: User-provided certs found at {SSL_CERT_DIR}")
                print(f"      Respecting user certificates (not overwriting)")
            
            # Return existing paths
            return str(SSL_CERT_FILE), str(SSL_KEY_FILE)
        
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
                x509.DNSName("127.0.0.1"),
                x509.DNSName("transcriber.cloud.cci.charite.de"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Save certificates
        with open(SSL_CERT_FILE, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(SSL_KEY_FILE, "wb") as f:
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
        model_key = data.get('model', 'medgemma') 
        enabled = data.get('enabled', True)

        if not enabled:
            return jsonify({'error': 'LLM rewriting is disabled'}), 400

        if not CHAT_AI_API_KEY:
            return jsonify({'error': 'LLM API Key not configured on server'}), 500

        model_id = AVAILABLE_MODELS.get(model_key, model_key)
        
        # DEBUG: Log what we're actually sending
        logger.info(f"Web Interface: Model key={model_key}")
        logger.info(f"Web Interface: Mapped model_id={model_id}")
        logger.info(f"Web Interface: Endpoint={CHAT_AI_ENDPOINT}")
        logger.info(f"Web Interface: API Key length={len(CHAT_AI_API_KEY) if CHAT_AI_API_KEY else 0}")
        logger.info(f"Web Interface: Text length={len(text)}")
        
        rewritten_text, status = call_llm_rewriter(text, model_id)
        
        logger.info(f"DEBUG: call_llm_rewriter returned: status={status}, len(text)={len(rewritten_text) if rewritten_text else 0}")
        
        if not rewritten_text:
            return jsonify({'error': f'LLM rewrite failed: {status}'}), 500
        
        return jsonify({
            'success': True,
            'rewritten_text': rewritten_text,
            'model_used': model_id
        })

    except Exception as e:
        logger.error(f"Error in LLM rewrite route: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error rewriting text: {str(e)}'}), 500

# --- NEW ROUTES FOR EDITING & BULK UPLOAD ---

@app.route('/save_transcription', methods=['POST'])
def save_transcription():
    """Saves edited transcription text to a temporary session-like storage (in-memory for simplicity)"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
    
        if not hasattr(app, 'active_edits'):
            app.active_edits = {}
        
        # Generate a simple ID based on timestamp for this session's edit
        edit_id = request.headers.get('X-Session-ID', str(int(time.time())))
        app.active_edits[edit_id] = data['text']
        
        return jsonify({'success': True, 'message': 'Transcription saved'})
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_bulk', methods=['POST'])
def upload_bulk():
    """Handles bulk upload of audio files - registers batch for later transcription."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        batch_id = f"batch_{int(time.time())}"
        batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        audio_files = []
        text_files = []
        processed_count = 0
        
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(batch_dir, filename)
            file.save(filepath)
            
            if filename.endswith(tuple(ALLOWED_EXTENSIONS)):
                audio_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size': os.path.getsize(filepath)
                })
                processed_count += 1
                logger.info(f"Bulk upload: {filename} ({os.path.getsize(filepath)} bytes)")
            
            elif filename.endswith('.txt'):
                text_files.append(filename)
                processed_count += 1
            
            elif filename.endswith('.zip'):
                # Extract and process
                extract_dir = os.path.join(batch_dir, f"extracted_{batch_id}")
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                os.remove(filepath)
                
                for root, dirs, files_in_zip in os.walk(extract_dir):
                    for f in files_in_zip:
                        if f.endswith(tuple(ALLOWED_EXTENSIONS)):
                            full_path = os.path.join(root, f)
                            audio_files.append({
                                'filename': f,
                                'filepath': full_path,
                                'size': os.path.getsize(full_path)
                            })
                            processed_count += 1
        
        # Register batch in app
        app.bulk_batches[batch_id] = {
            'created': time.time(),
            'audio_files': audio_files,
            'text_files': text_files,
            'processed': False,
            'results': []
        }
        
        return jsonify({
            'success': True,
            'processed_count': processed_count,
            'audio_count': len(audio_files),
            'text_count': len(text_files),
            'batch_id': batch_id,
            'message': f'{processed_count} file(s) uploaded. Audio files registered for transcription.'
        })
    
    except Exception as e:
        logger.error(f"Bulk upload error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/list_batches', methods=['GET'])
def list_batches():
    """List all uploaded batches."""
    try:
        batches = []
        for batch_id, batch_data in app.bulk_batches.items():
            batches.append({
                'batch_id': batch_id,
                'created': batch_data['created'],
                'audio_count': len(batch_data['audio_files']),
                'text_count': len(batch_data['text_files']),
                'processed': batch_data.get('processed', False),
                'result_count': len(batch_data.get('results', []))
            })
        return jsonify({'batches': batches})
    except Exception as e:
        logger.error(f"Error listing batches: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def _transcribe_single_file(audio_file, language=None):
    """Delegate transcription to process.py - returns dict result."""
    from process import transcribe_audio_locally
    
    try:
        logger.info(f"Transcribing: {audio_file['filename']}")
        
        transcription, wordOffS = transcribe_audio_locally(
            audio_file['filepath'],
            language=language
        )
        
        if isinstance(transcription, str) and ("Error:" in transcription or "failed" in transcription.lower()):
            return {
                'filename': audio_file['filename'],
                'status': 'failed',
                'error': transcription
            }
        
        return {
            'filename': audio_file['filename'],
            'status': 'success',
            'transcription': transcription if transcription else "",
            'offsets': wordOffS if isinstance(wordOffS, list) else [],
            'audio_path': audio_file['filepath']
        }
    
    except Exception as e:
        logger.error(f"Error transcribing {audio_file['filename']}: {e}", exc_info=True)
        return {
            'filename': audio_file['filename'],
            'status': 'failed',
            'error': f'Transcription error: {str(e)}'
        }

def _transcribe_batch(batch_id, language=None):
    """Transcribe all files in a batch using process.py - returns results dict."""
    if batch_id not in app.bulk_batches:
        return {'error': 'Batch not found'}, 404
    
    batch = app.bulk_batches[batch_id]
    results = []
    success_count = 0
    failed_count = 0
    
    for idx, audio_file in enumerate(batch['audio_files'], 1):
        result = _transcribe_single_file(audio_file, language=language)
        results.append(result)
        
        if result['status'] == 'success':
            success_count += 1
        else:
            failed_count += 1
    
    # Update batch state
    batch['processed'] = True
    batch['results'] = results
    
    return {
        'success': True,
        'batch_id': batch_id,
        'success_count': success_count,
        'failed_count': failed_count,
        'results': results
    }

@app.route('/process_batch/<batch_id>', methods=['POST'])
def process_batch(batch_id):
    """Transcribe all audio files in a batch."""
    try:
        language = request.form.get('language', 'auto')
        if language == 'auto' or not language:
            language = None
        
        result = _transcribe_batch(batch_id, language=language)
        
        if 'error' in result:
            return jsonify(result), 404
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/process_all_batches', methods=['POST'])
def process_all_batches():
    """Transcribe all unprocessed batches sequentially."""
    try:
        language = request.form.get('language', 'auto')
        if language == 'auto' or not language:
            language = None
        
        batches_to_process = [
            bid for bid, data in app.bulk_batches.items()
            if not data.get('processed', False)
        ]
        
        if not batches_to_process:
            return jsonify({'success': True, 'message': 'No batches to process'})
        
        all_results = {}
        total_success = 0
        total_failed = 0
        
        for batch_id in batches_to_process:
            try:
                result = _transcribe_batch(batch_id, language=language)
                if result.get('success'):
                    total_success += result.get('success_count', 0)
                    total_failed += result.get('failed_count', 0)
                all_results[batch_id] = result
            except Exception as e:
                logger.error(f"Error processing batch {batch_id}: {e}")
                all_results[batch_id] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'batches_processed': len(batches_to_process),
            'total_success': total_success,
            'total_failed': total_failed,
            'results': all_results
        })
    
    except Exception as e:
        logger.error(f"Error in process_all_batches: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download_text/<file_type>/<filename>')
def download_text(file_type, filename):
    """Downloads text files (original, anonymized, LLM)"""
    try:
        # Map file types to folders
        folder_map = {
            'original': app.config['UPLOAD_FOLDER'], # Assuming original is saved here or in a specific folder
            'bert': ANONYM_FOLDER,
            'llm': LLM_ANONYM_FOLDER
        }
        
        if file_type not in folder_map:
            return jsonify({'error': 'Invalid file type'}), 400
            
        base_path = folder_map[file_type]
        file_path = os.path.join(base_path, secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(file_path, as_attachment=True, download_name=filename)
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


def cleanup_expired_sessions():
    """Background task to remove expired sessions periodically."""
    MAX_SESSION_AGE_SECONDS = int(os.getenv('SESSION_MAX_AGE_SECONDS', 3600))
    CLEANUP_INTERVAL_SECONDS = int(os.getenv('SESSION_CLEANUP_INTERVAL', 300))  # ← Use env var
    
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS) 
        
        try:
            if not hasattr(app, 'session_entity_maps'):
                continue
            
            current_time = time.time()
            expired_sessions = [
                sid for sid, record in app.session_entity_maps.items()
                if current_time - record.get('timestamp', 0) > MAX_SESSION_AGE_SECONDS
            ]
            
            if expired_sessions:
                count = len(expired_sessions)
                for sid in expired_sessions:
                    del app.session_entity_maps[sid]
                
                logger.info(f"Background cleanup: removed {count} expired session(s)")
        
        except Exception as e:
            logger.error(f"Error in background session cleanup: {e}")

@app.route('/debug/sessions')
def debug_session_count():
    """Healthcheck endpoint to monitor session memory usage."""
    session_count = len(getattr(app, 'session_entity_maps', {}))
    total_memory_mb = sum(
        sys.getsizeof(record['entity_map']) / (1024 * 1024)
        for record in getattr(app, 'session_entity_maps', {}).values()
    )
    return jsonify({
        'active_sessions': session_count,
        'estimated_memory_mb': round(total_memory_mb, 2)
    })

# Start background cleaner (non-blocking)

if __name__ == '__main__':
    use_https = os.getenv('USE_HTTPS', 'true').lower() == 'true'
    import threading
    session_cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    session_cleanup_thread.start()
    logger.info("Session cleanup thread started (every 5 minutes)")
    
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