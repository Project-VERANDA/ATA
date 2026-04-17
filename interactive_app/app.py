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

try:
    import whisperx
    from whisperx import diarize
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logging.error("WhisperX is not available. Please install it to use this application.")
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

try:
    from bert_anonymizer.anonymizer import anonymize_text_with_bert
    BERT_ANONYMIZER_AVAILABLE = True
except (ImportError, OSError) as e:
    BERT_ANONYMIZER_AVAILABLE = False
    logging.warning(f"BERT Anonymizer not available. Error: {e}")

try:
    from spacy_anonymizer.anonymizer import anonymize_text_with_spacy, is_spacy_model_available
    SPACY_ANONYMIZER_AVAILABLE = is_spacy_model_available()
except (ImportError, OSError) as e:
    SPACY_ANONYMIZER_AVAILABLE = False
    logging.warning(f"spaCy Anonymizer not available. Error: {e}")

try:
    from ensemble_anonymizer.anonymizer import anonymize_text_with_ensemble
    ENSEMBLE_ANONYMIZER_AVAILABLE = BERT_ANONYMIZER_AVAILABLE and SPACY_ANONYMIZER_AVAILABLE
except (ImportError, OSError) as e:
    ENSEMBLE_ANONYMIZER_AVAILABLE = False
    logging.warning(f"Ensemble Anonymizer not available. Error: {e}")

TTS_AVAILABLE = GTTS_AVAILABLE

import torch
import gc
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- SHARED ENGINE IMPORT (CRITICAL CHANGE) ---
# Add pipeline to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

# Import the shared functions from process.py
try:
    from process import transcribe_audio_locally, load_models
    logger.info("Successfully imported shared transcription engine from process.py")
    
    # Pre-load models at startup (optional but recommended for speed)
    try:
        load_models()
        logger.info("Interactive App: Models pre-loaded from process.py")
    except Exception as e:
        logger.error(f"Interactive App: Failed to pre-load models: {e}")
        # We continue anyway; lazy loading will happen on first request
except ImportError as e:
    logger.critical(f"CRITICAL: Could not import from process.py: {e}")
    logger.critical("Ensure process.py is in the 'pipeline' folder and defines 'transcribe_audio_locally' and 'load_models'.")
    sys.exit(1)

# Wrapper for backward compatibility in routes
def transcribe_audio(audio_path, language='de'):
    """Calls the shared engine from process.py"""
    return transcribe_audio_locally(audio_path, language)

# --- CONFIGURATION ---
CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://chat-ai.academiccloud.de/v1')
DEFAULT_MODEL = os.getenv('CHAT_AI_MODEL', 'llama-3.1-8b-instruct')

AVAILABLE_MODELS = {
    'llama-3.1-8b-instruct': 'Meta Llama 3.1 8B Instruct',
    'gemma-3-27b-it': 'Google Gemma 3 27B Instruct',
    'internvl2.5-8b-mpo': 'OpenGVLab InternVL2.5 8B MPO',
    'qwen-3-235b-a22b': 'Alibaba Qwen 3 235B A22B',
    'qwen-3-32b': 'Alibaba Qwen 3 32B',
    'qwq-32b': 'Alibaba Qwen QwQ 32B',
    'deepseek-r1': 'DeepSeek R1',
    'deepseek-r1-distill-llama-70b': 'DeepSeek R1 Distill Llama 70B',
    'llama-3.3-70b-instruct': 'Meta Llama 3.3 70B Instruct',
    'llama-3.1-sauerkrautlm-70b-instruct': 'VAGOsolutions Llama 3.1 SauerkrautLM 70B Instruct',
    'mistral-large-instruct': 'Mistral Large Instruct',
    'codestral-22b': 'Mistral Codestral 22B',
    'e5-mistral-7b-instruct': 'E5 Mistral 7B Instruct',
    'qwen-2.5-vl-72b-instruct': 'Alibaba Qwen 2.5 VL 72B Instruct',
    'qwen-2.5-coder-32b-instruct': 'Alibaba Qwen 2.5 Coder 32B Instruct'
}

WHISPER_MODELS = OrderedDict([
    ('tiny', '1: Tiny (Fastest, lowest accuracy)'),
    ('base', '2: Base (Default, good balance)'),
    ('small', '3: Small (More accurate, slower)'),
    ('medium', '4: Medium (High accuracy, very slow)'),
    ('large', '5: Large (Best accuracy, slowest)')
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
            logger.warning("No audio was generated.")
            return None, None

        audio_filename = f"speech_output_{int(time.time())}.mp3"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
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
    models_to_send = AVAILABLE_MODELS.copy()
    local_models = {}
    if BERT_ANONYMIZER_AVAILABLE:
        local_models['bert-base-ner'] = 'Local BERT Model'
    if SPACY_ANONYMIZER_AVAILABLE:
        local_models['spacy-de-ner'] = 'Local spaCy Model'
    if ENSEMBLE_ANONYMIZER_AVAILABLE:
        local_models['ensemble-spacy-bert'] = 'Local Ensemble (spaCy + BERT)'

    if local_models:
        combined_models = {'local_models': local_models, **models_to_send}
    else:
        combined_models = models_to_send

    return jsonify({
        'available_models': combined_models,
        'default_model': DEFAULT_MODEL
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
        # Note: model_size is ignored now as we use the fixed local model from process.py
        # But we keep the param for UI compatibility
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Transcribing file: {filename} in language: {language}")
            transcription = transcribe_audio(filepath, language=language)
            
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
        transcription = transcribe_audio(audio_file_to_transcribe, language=language)
            
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
        anonymization_level = data.get('level', 'standard')
        selected_model = data.get('model', DEFAULT_MODEL)

        # Local Models
        if selected_model == 'bert-base-ner':
            if not BERT_ANONYMIZER_AVAILABLE:
                return jsonify({'error': 'BERT anonymizer not available'}), 500
            anonymized_text, _ = anonymize_text_with_bert(text)
            return jsonify({'success': True, 'anonymized_text': anonymized_text, 'model_used': 'Local BERT', 'tts_available': GTTS_AVAILABLE})
        
        elif selected_model == 'spacy-de-ner':
            if not SPACY_ANONYMIZER_AVAILABLE:
                return jsonify({'error': 'spaCy anonymizer not available'}), 500
            anonymized_text, _ = anonymize_text_with_spacy(text)
            return jsonify({'success': True, 'anonymized_text': anonymized_text, 'model_used': 'Local spaCy', 'tts_available': GTTS_AVAILABLE})
        
        elif selected_model == 'ensemble-spacy-bert':
            if not ENSEMBLE_ANONYMIZER_AVAILABLE:
                return jsonify({'error': 'Ensemble anonymizer not available'}), 500
            anonymized_text = anonymize_text_with_ensemble(text)
            return jsonify({'success': True, 'anonymized_text': anonymized_text, 'model_used': 'Local Ensemble', 'tts_available': GTTS_AVAILABLE})

        # Remote LLM
        if selected_model not in AVAILABLE_MODELS:
            return jsonify({'error': f'Invalid model: {selected_model}'}), 400
        
        if not CHAT_AI_API_KEY:
            return jsonify({'error': 'Chat AI API key not configured'}), 500
        
        prompts = {
            'basic': "Anonymize PII (names, phones, emails, addresses) with specific tags. Keep speaker tags. Return ONLY text.",
            'standard': "Anonymize PII (names, professions, dates, ages, locations) with specific tags. Keep speaker tags. Return ONLY text.",
            'strict': "Thoroughly anonymize ALL PII with specific tags. Keep speaker tags. Return ONLY text."
        }
        
        prompt = prompts.get(anonymization_level, prompts['standard'])
        model_name = AVAILABLE_MODELS.get(selected_model, selected_model)
        
        headers = {'Authorization': f'Bearer {CHAT_AI_API_KEY}', 'Content-Type': 'application/json'}
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Text to anonymize:\n\n{text}"}
            ],
            "max_tokens": 8000,
            "temperature": 0.1
        }
        
        for attempt in range(3):
            try:
                response = requests.post(f"{CHAT_AI_ENDPOINT}/chat/completions", headers=headers, json=payload, timeout=360)
                break
            except requests.exceptions.Timeout:
                if attempt < 2: continue
                return jsonify({'error': 'Timeout'}), 504
            except requests.exceptions.RequestException as e:
                return jsonify({'error': f'Connection error: {str(e)}'}), 503
        
        if response.status_code != 200:
            return jsonify({'error': f'API error: {response.status_code}'}), 500
        
        response_data = response.json()
        anonymized_text = response_data['choices'][0]['message']['content'].strip()
        anonymized_text = re.sub(r'<thought>.*?</thought>', '', anonymized_text, flags=re.DOTALL).strip()
        
        return jsonify({
            'success': True,
            'anonymized_text': anonymized_text,
            'level': anonymization_level,
            'model_used': model_name,
            'tts_available': GTTS_AVAILABLE
        })
    
    except Exception as e:
        logger.error(f"Error anonymizing text: {str(e)}")
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
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
    # Since we rely on process.py, we check if the import worked
    try:
        from process import load_models
        # We can't easily check model state without calling it, so we assume healthy if import succeeded
        return jsonify({
            'status': 'healthy',
            'transcription_available': True,
            'transcription_type': 'shared-engine',
            'transcription_description': 'Using shared process.py engine',
            'speaker_diarization': True, # Assumed if process.py loaded
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
        # Ensure timedelta is imported here if not at top level, but top level is better
        from datetime import datetime, timezone, timedelta 
        
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Localhost"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Speech Anonymizer"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        # Fixed: Use imported classes directly
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

if __name__ == '__main__':
    use_https = os.getenv('USE_HTTPS', 'true').lower() == 'true'
    #use_https = False

    if use_https:
        try:
            cert_file, key_file = create_self_signed_cert()
            if cert_file and key_file:
                logger.info("Starting server with HTTPS (self-signed certificate)")
                app.run(debug=True, host='0.0.0.0', port=5001, ssl_context=(cert_file, key_file))
            else:
                logger.info("Starting server with HTTPS (ad-hoc certificate)")
                app.run(debug=True, host='0.0.0.0', port=5001, ssl_context='adhoc')
        except Exception as e:
            logger.error(f"Failed to start HTTPS server: {e}")
            logger.info("Falling back to HTTP")
            app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.info("Starting server with HTTP")
        app.run(debug=True, host='0.0.0.0', port=5001)