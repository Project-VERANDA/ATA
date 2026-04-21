import os
import sys
import tempfile
import time
import re
import subprocess
from collections import OrderedDict
import logging
from pathlib import Path
import ipaddress
from datetime import datetime, timezone, timedelta

# --- PATH SETUP ---
current_script_dir = Path(__file__).resolve().parent
main_folder = current_script_dir.parent

# 1. Add 'pipeline' to sys.path
pipeline_path = main_folder / "pipeline"
if str(pipeline_path) not in sys.path:
    sys.path.insert(0, str(pipeline_path))

# 2. Add 'pipeline/model' to sys.path (for anonymizer)
model_folder_path = pipeline_path / "model"
if str(model_folder_path) not in sys.path:
    sys.path.insert(0, str(model_folder_path))

# 3. Add 'ModelTraining' to sys.path (legacy support)
model_training_path = main_folder / "ModelTraining"
if str(model_training_path) not in sys.path:
    sys.path.insert(0, str(model_training_path))

# --- IMPORTS ---
try:
    from process import (
        transcribe_audio_locally, 
        load_models, 
        call_llm_rewriter, 
        AVAILABLE_LLM_MODELS, 
        CHAT_AI_API_KEY, 
        CHAT_AI_ENDPOINT,
        anonymize_text_locally
    )
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported shared functions from process.py")
except ImportError as e:
    logger.critical(f"CRITICAL: Could not import from process.py: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logger.error("WhisperX is not available. Exiting.")
    sys.exit("Exiting: WhisperX is a required dependency.")

try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# --- CONFIGURATION ---
BERT_ANONYMIZER_AVAILABLE = True
DEFAULT_MODEL = 'bert-base-ner'
TTS_AVAILABLE = GTTS_AVAILABLE

import torch
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
from dotenv import load_dotenv

load_dotenv()

if WHISPERX_AVAILABLE:
    logger.info("WhisperX is available - using for transcription with speaker diarization.")
else:
    logger.error("WhisperX is not installed. The application cannot run without it.")

if GTTS_AVAILABLE:
    logger.info("Google TTS (gTTS) available for German text-to-speech.")
else:
    logger.warning("gTTS not available. German text-to-speech will be disabled.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- PRE-LOAD MODELS ---
try:
    load_models()
    logger.info("Interactive App: Models pre-loaded from process.py")
except Exception as e:
    logger.error(f"Interactive App: Failed to pre-load models: {e}")

# --- CONFIGURATION ---
CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://llm.cloud.cci.charite.de/v1')

# Model mappings for the UI
AVAILABLE_MODELS = {
    'medgemma': 'medgemma-1.5-4b-it',
    'medgemma27b': 'medgemma-27b-it',
    'gpt-oss-120b': 'gpt-oss-120b',
    'Qwen3.5-27B': 'Qwen3.5-27B',
    'Qwen3.5-397B-A17B': 'Qwen3.5-397B-A17B',
    'qwen3-asr-1.7b': 'Qwen3-ASR-1.7B',
    'cle-Kimi-K2.5': 'Kimi-K2.5',
    'cle-Qwen3.5-397B-A17B-FP8': 'Qwen3.5-397B-A17B-FP8'
}

WHISPER_MODELS = OrderedDict([
    ('tiny', '1: Tiny (Fastest)'),
    ('base', '2: Base (Default)'),
    ('small', '3: Small'),
    ('medium', '4: Medium'),
    ('large', '5: Large (Best)')
])

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_beep(duration_ms=400, freq=1000):
    return Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-12)

def generate_speech(text, voice_settings=None, language='de'):
    if not GTTS_AVAILABLE or not PYDUB_AVAILABLE:
        logger.error("gTTS or pydub not available, cannot generate speech with beeps.")
        return None, None

    try:
        tag_regex = re.compile(r'(\[[A-Z_]+\])')
        text_parts = tag_regex.split(text)
        beep_sound = generate_beep()
        final_audio = AudioSegment.empty()

        for part in text_parts:
            if not part:
                continue
            if tag_regex.match(part):
                final_audio += beep_sound
            elif part.strip():
                try:
                    tts = gTTS(text=part.strip(), lang=language, slow=False)
                    with io.BytesIO() as fp:
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        speech_segment = AudioSegment.from_mp3(fp)
                        final_audio += speech_segment
                except Exception as e:
                    logger.error(f"gTTS failed for segment: '{part[:30]}...'. Error: {e}")
        
        if len(final_audio) == 0:
            logger.warning("No audio was generated (all segments were tags or empty).")
            return None, None

        audio_filename = f"speech_output_{int(time.time())}.mp3"
        audio_path = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), audio_filename)
        final_audio.export(audio_path, format="mp3")
        logger.info(f"Speech with beeps generated successfully: {audio_path}")
        return audio_path, 'gtts_with_beeps'
    except Exception as e:
        logger.error(f"Speech generation with beeps failed: {e}")
        return None, None

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
    return jsonify({
        'whisper_models': list(WHISPER_MODELS.items()),
        'default_whisper_model': 'base'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        language = request.form.get('language', 'de')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Transcribing file: {filename} in language: {language}")
            transcription = transcribe_audio_locally(filepath, language=language)
            
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'transcription': transcription
            })
        
        return jsonify({'error': 'Invalid file format'}), 400
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/transcribe_recording', methods=['POST'])
def transcribe_recording():
    temp_file_path = None
    converted_file_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio recording provided'}), 400
        
        audio_blob = request.files['audio']
        language = request.form.get('language', 'de')
        
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
        except Exception:
            audio_file_to_transcribe = temp_file_path
        
        logger.info(f"Starting transcription of: {audio_file_to_transcribe}")
        transcription = transcribe_audio_locally(audio_file_to_transcribe, language=language)
            
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if converted_file_path and os.path.exists(converted_file_path):
            os.unlink(converted_file_path)
            
        return jsonify({
            'success': True,
            'transcription': transcription
        })
    
    except Exception as e:
        logger.error(f"Error processing recording: {str(e)}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if converted_file_path and os.path.exists(converted_file_path):
            os.unlink(converted_file_path)
        return jsonify({'error': f'Error processing recording: {str(e)}'}), 500

@app.route('/anonymize', methods=['POST'])
def anonymize_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not BERT_ANONYMIZER_AVAILABLE:
            return jsonify({'error': 'BERT anonymizer not available. Please check server logs.'}), 500
        
        anonymized_text, success, error_msg = anonymize_text_locally(text)
        
        if not success:
            logger.error(f"Anonymization failed: {error_msg}")
            return jsonify({'error': f'Anonymization failed: {error_msg}'}), 500
        
        return jsonify({
            'success': True, 
            'anonymized_text': anonymized_text, 
            'model_used': 'Local BERT', 
            'tts_available': GTTS_AVAILABLE
        })
    
    except Exception as e:
        logger.error(f"Error in anonymize route: {str(e)}")
        return jsonify({'error': f'Error anonymizing text: {str(e)}'}), 500

@app.route('/generate_speech', methods=['POST'])
def generate_speech_route():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text'}), 400
        
        if not GTTS_AVAILABLE:
            return jsonify({'error': 'TTS not available'}), 500
        
        voice_settings = data.get('voice_settings', {'voice': 'de'})
        language = voice_settings.get('voice', 'de')
        
        audio_path, tts_engine = generate_speech(text, voice_settings, language)
        
        if audio_path and os.path.exists(audio_path):
            audio_filename = os.path.basename(audio_path)
            return jsonify({
                'success': True,
                'audio_file': audio_filename,
                'tts_engine': tts_engine,
                'audio_url': f'/download_speech/{audio_filename}',
                'message': 'Speech generated successfully!'
            })
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        return jsonify({'error': f'Error generating speech: {str(e)}'}), 500

@app.route('/download_speech/<filename>')
def download_speech(filename):
    try:
        if not filename.startswith('speech_output_') or '..' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading speech file: {str(e)}")
        return jsonify({'error': f'Error downloading speech file: {str(e)}'}), 500

@app.route('/available_voices')
def get_available_voices():
    try:
        voices = []
        if GTTS_AVAILABLE:
            voices = [{'id': 'de', 'name': 'Deutsch (Google TTS)', 'language': ['de'], 'engine': 'gtts'}]
        return jsonify({'voices': voices, 'tts_available': GTTS_AVAILABLE})
    except Exception as e:
        return jsonify({'error': f'Error getting voices: {str(e)}'}), 500

@app.route('/health')
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'transcription_available': True,
            'transcription_type': 'shared-engine',
            'speaker_diarization': True,
            'whisperx_available': WHISPERX_AVAILABLE,
            'chat_ai_configured': CHAT_AI_API_KEY is not None,
            'default_model': DEFAULT_MODEL,
            'available_models': list(AVAILABLE_MODELS.keys()),
            'tts_available': GTTS_AVAILABLE,
            'tts_engine': 'gtts' if GTTS_AVAILABLE else None
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

def create_self_signed_cert():
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Localhost"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Speech Anonymizer"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        now = datetime.now(timezone.utc)
        cert = x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(private_key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now).not_valid_after(now + timedelta(days=365)).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        with open("cert.pem", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open("key.pem", "wb") as f:
            f.write(private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()))
        
        return "cert.pem", "key.pem"
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
        logger.info(f"Web Interface: Requesting LLM rewrite with model {model_id}")

        rewritten_text, status = call_llm_rewriter(text, model_id)
        
        if not rewritten_text:
            return jsonify({'error': f'LLM rewrite failed: {status}'}), 500
        
        return jsonify({
            'success': True,
            'rewritten_text': rewritten_text,
            'model_used': model_id
        })

    except Exception as e:
        logger.error(f"Error in LLM rewrite route: {str(e)}")
        return jsonify({'error': f'Error rewriting text: {str(e)}'}), 500

if __name__ == '__main__':
    use_https = os.getenv('USE_HTTPS', 'true').lower() == 'true'

    if use_https:
        try:
            cert_file, key_file = create_self_signed_cert()
            if cert_file and key_file:
                logger.info("Starting server with HTTPS (self-signed certificate)")
                app.run(debug=False, host='0.0.0.0', port=5001, ssl_context=(cert_file, key_file))
            else:
                logger.info("Starting server with HTTPS (ad-hoc certificate)")
                app.run(debug=False, host='0.0.0.0', port=5001, ssl_context='adhoc')
        except Exception as e:
            logger.error(f"Failed to start HTTPS server: {e}")
            logger.info("Falling back to HTTP")
            app.run(debug=False, host='0.0.0.0', port=5001)
    else:
        logger.info("Starting server with HTTP")
        app.run(debug=False, host='0.0.0.0', port=5001)