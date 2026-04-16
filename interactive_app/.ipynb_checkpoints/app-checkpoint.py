import os
import sys
import tempfile
import time
import re
import subprocess
from collections import OrderedDict
import logging

# Simplified imports: only use whisperx
try:
    import whisperx
    from whisperx import diarize
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logging.error("WhisperX is not available. Please install it to use this application.")
    # Exit if whisperx is not available, as it's a core dependency
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
    # The ensemble is only available if both its components are available
    ENSEMBLE_ANONYMIZER_AVAILABLE = BERT_ANONYMIZER_AVAILABLE and SPACY_ANONYMIZER_AVAILABLE
except (ImportError, OSError) as e:
    ENSEMBLE_ANONYMIZER_AVAILABLE = False
    logging.warning(f"Ensemble Anonymizer not available. Error: {e}")

# Only use gTTS for German TTS
TTS_AVAILABLE = GTTS_AVAILABLE

import torch
import gc
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import requests
from dotenv import load_dotenv
import ssl

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if WHISPERX_AVAILABLE:
    logger.info("WhisperX is available - using for transcription with speaker diarization.")
else:
    # This part is now redundant due to the sys.exit above, but kept for clarity
    logger.error("WhisperX is not installed. The application cannot run without it.")

if GTTS_AVAILABLE:
    logger.info("Google TTS (gTTS) available for German text-to-speech.")
else:
    logger.warning("gTTS not available. German text-to-speech will be disabled.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Simplified model initialization
whisperx_model = None
diarize_model = None
current_model_size = None

# Available whisper models by size, ordered from worst to best quality
WHISPER_MODELS = OrderedDict([
    ('tiny', '1: Tiny (Fastest, lowest accuracy)'),
    ('base', '2: Base (Default, good balance)'),
    ('small', '3: Small (More accurate, slower)'),
    ('medium', '4: Medium (High accuracy, very slow)'),
    ('large', '5: Large (Best accuracy, slowest)')
])

def load_transcription_model(model_name="base"):
    """Dynamically loads a WhisperX transcription model by size."""
    global whisperx_model, current_model_size

    if model_name not in WHISPER_MODELS:
        logger.warning(f"Invalid model name '{model_name}'. Falling back to 'base'.")
        model_name = "base"

    if model_name == current_model_size:
        logger.info(f"WhisperX model '{model_name}' is already loaded.")
        return

    logger.info(f"Loading WhisperX transcription model: '{model_name}'...")

    # Clear existing models from memory to free up resources
    whisperx_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        device = "cpu"
        compute_type = "float32"
        whisperx_model = whisperx.load_model(model_name, device, compute_type=compute_type)
        current_model_size = model_name
        logger.info(f"WhisperX '{model_name}' model loaded successfully.")
    except Exception as e:
        logger.error(f"Could not load WhisperX model '{model_name}': {e}")
        logger.error("The application may not function correctly without a transcription model.")
        current_model_size = None

# Load the default model and the diarization model at startup
load_transcription_model("base")

try:
    logger.info("Loading diarization model...")
    diarize_model = diarize.DiarizationPipeline(use_auth_token=os.getenv("HUGGING_FACE_TOKEN"), device="cpu")
    logger.info("Diarization model loaded successfully.")
except Exception as e:
    logger.warning(f"Failed to load diarization model: {e}")
    logger.warning("Speaker diarization will be skipped.")
    diarize_model = None

# Configure Chat AI API (SAIA platform)
CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://chat-ai.academiccloud.de/v1')
DEFAULT_MODEL = os.getenv('CHAT_AI_MODEL', 'llama-3.1-8b-instruct')

# Available models for anonymization (exact names from GWDG Chat AI - Open Source Only)
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

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_beep(duration_ms=400, freq=1000):
    """Generates a beep sound using pydub."""
    return Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-12) # Lower volume

def generate_speech(text, voice_settings=None, language='de'):
    """Generate speech from text using Google TTS, replacing bracketed tags with beeps."""
    if not GTTS_AVAILABLE or not PYDUB_AVAILABLE:
        logger.error("gTTS or pydub not available, cannot generate speech with beeps.")
        return None, None

    try:
        logger.info(f"Generating speech with beeps for text length: {len(text)} characters in language: {language}")

        # Regex to find all bracketed tags like [NAME_PATIENT]
        tag_regex = re.compile(r'(\[[A-Z_]+\])')

        # Split the text by the tags, keeping the tags as delimiters
        text_parts = tag_regex.split(text)

        # Generate a standard beep sound
        beep_sound = generate_beep()

        # Initialize an empty audio segment for the final audio
        final_audio = AudioSegment.empty()

        for part in text_parts:
            if not part:
                continue

            if tag_regex.match(part):
                # This part is a tag, so append a beep
                logger.info(f"Replacing tag '{part}' with a beep.")
                final_audio += beep_sound
            elif part.strip():
                # This is a text part, generate speech for it
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
            logger.warning("No audio was generated. The text might be empty or contain only tags.")
            return None, None

        # Generate a unique filename and path for the combined audio
        audio_filename = f"speech_output_{int(time.time())}.mp3"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        
        # Export the final combined audio to an MP3 file
        final_audio.export(audio_path, format="mp3")
        
        logger.info(f"Speech with beeps generated successfully: {audio_path}")
        return audio_path, 'gtts_with_beeps'

    except Exception as e:
        logger.error(f"Speech generation with beeps failed: {e}")
        return None, None

def merge_consecutive_speaker_segments(segments):
    """Merge consecutive segments from the same speaker"""
    merged_segments = []
    prev_segment = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"]

        if prev_segment and prev_segment["speaker"] == speaker:
            prev_segment["text"] += " " + text
        else:
            if prev_segment:
                merged_segments.append(prev_segment)
            prev_segment = {"speaker": speaker, "text": text}

    if prev_segment:
        merged_segments.append(prev_segment)

    return merged_segments

def transcribe_with_speakers(audio_path, language='de'):
    """Transcribe audio with speaker identification using WhisperX"""
    try:
        logger.info(f"Processing audio file: {audio_path} with WhisperX in language '{language}'")
        
        if whisperx_model is None:
            raise RuntimeError("WhisperX model is not loaded.")
        
        if diarize_model is None:
            logger.warning("Diarization model not loaded. Transcribing without speaker identification.")
            return transcribe_basic(audio_path, language)

        audio = whisperx.load_audio(audio_path)
        logger.info(f"Audio loaded, duration: {len(audio)/16000:.2f} seconds")
        
        result = whisperx_model.transcribe(audio, batch_size=32, language=language)
        logger.info(f"Transcription completed, detected language: {result.get('language', 'unknown')}")

        try:
            language_code = language if language else result.get("language", "de")
            if not language_code:
                raise Exception("Language not detected, cannot align.")

            logger.info(f"Loading alignment model for language: {language_code}...")
            align_model, metadata = whisperx.load_align_model(language_code=language_code, device="cpu")
            logger.info("Alignment model loaded successfully")

            result = whisperx.align(result["segments"], align_model, metadata, audio, "cpu", return_char_alignments=False)
            logger.info("Alignment completed")
        
        except Exception as e:
            logger.warning(f"Failed to align transcription: {e}. Diarization may be less accurate.")
        
        logger.info("Starting speaker diarization...")
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=4)
        logger.info("Diarization completed")
        
        logger.info("Assigning speakers to segments...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker assignment completed.")
        
        result["segments"] = merge_consecutive_speaker_segments(result["segments"])
        
        speaker_text = [f"{segment.get('speaker', 'Unknown')}: {segment['text'].strip()}" for segment in result["segments"] if segment['text'].strip()]
        
        gc.collect()
        
        final_result = "\n".join(speaker_text) if speaker_text else "No speech detected."
        logger.info(f"Final transcription length: {len(final_result)} characters")
        return final_result
            
    except Exception as e:
        logger.error(f"Transcription with speakers failed: {e}")
        return f"Transcription failed: {e}"

def transcribe_basic(audio_path, language='de'):
    """Basic transcription using WhisperX without speaker identification"""
    try:
        logger.info(f"Processing audio file: {audio_path} with WhisperX (basic) in language '{language}'")

        if whisperx_model is None:
            raise RuntimeError("WhisperX model is not loaded.")
            
        audio = whisperx.load_audio(audio_path)
        result = whisperx_model.transcribe(audio, batch_size=16, language=language)
        return result.get("text", "No speech detected.")
                
    except Exception as e:
        logger.error(f"Basic transcription failed: {e}")
        return f"Transcription failed: {e}"

def transcribe_audio(audio_path, language='de'):
    """Main transcription function - uses speaker diarization when available"""
    if WHISPERX_AVAILABLE and diarize_model is not None:
        return transcribe_with_speakers(audio_path, language=language)
    else:
        return transcribe_basic(audio_path, language=language)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def get_models():
    # Start with the base list of models from the environment
    models_to_send = AVAILABLE_MODELS.copy()
    
    local_models = {}
    # If the local BERT model is available, add it
    if BERT_ANONYMIZER_AVAILABLE:
        local_models['bert-base-ner'] = 'Local BERT Model'
    
    # If the local spaCy model is available, add it
    if SPACY_ANONYMIZER_AVAILABLE:
        local_models['spacy-de-ner'] = 'Local spaCy Model'

    # If the local Ensemble model is available, add it
    if ENSEMBLE_ANONYMIZER_AVAILABLE:
        local_models['ensemble-spacy-bert'] = 'Local Ensemble (spaCy + BERT)'

    # Combine the model lists
    if local_models:
        # Use a more descriptive key to avoid clashes and for clarity
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
        
        # Get selected language and model size
        language = request.form.get('language', 'de')
        model_size = request.form.get('model_size', 'base')
        load_transcription_model(model_size)
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Transcribe audio
            logger.info(f"Transcribing file: {filename} in language: {language}")
            transcription = transcribe_audio(filepath, language=language)
            
            # Clean up uploaded file
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
        
        # Get selected language and model size
        language = request.form.get('language', 'de')
        model_size = request.form.get('model_size', 'base')
        load_transcription_model(model_size)

        logger.info(f"Received audio file: filename='{audio_blob.filename}', content_type='{audio_blob.content_type}', language='{language}'")
        
        # Save the recorded audio to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            audio_blob.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Check file size after saving
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Saved audio file: {temp_file_path}, size: {file_size} bytes")
        
        if file_size == 0:
            return jsonify({'error': 'Empty audio file saved - no audio data received from browser'}), 400
        
        if file_size < 100:  # Very small file, probably not real audio
            return jsonify({'error': f'Audio file too small ({file_size} bytes) - recording may have failed'}), 400
        
        # Try to convert webm to wav for better compatibility
        try:
            converted_file_path = temp_file_path.replace('.webm', '.wav')
            
            # Use ffmpeg to convert webm to wav
            result = subprocess.run([
                'ffmpeg', '-i', temp_file_path, 
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # mono
                '-f', 'wav',     # WAV format
                '-y',            # overwrite output
                converted_file_path
            ], check=True, capture_output=True, text=True)
            
            logger.info(f"Converted to WAV: {converted_file_path}")
            
            # Check converted file size
            converted_size = os.path.getsize(converted_file_path)
            logger.info(f"Converted file size: {converted_size} bytes")
            
            if converted_size > 0:
                audio_file_to_transcribe = converted_file_path
            else:
                logger.warning("Converted file is empty, using original")
                audio_file_to_transcribe = temp_file_path
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg conversion failed: {e}")
            logger.warning(f"FFmpeg stderr: {e.stderr}")
            logger.info("Trying direct transcription of webm file")
            audio_file_to_transcribe = temp_file_path
        except FileNotFoundError:
            logger.warning("FFmpeg not found, using original webm file")
            audio_file_to_transcribe = temp_file_path
        except Exception as e:
            logger.warning(f"Audio conversion error: {e}")
            audio_file_to_transcribe = temp_file_path
        
        # Transcribe the recording
        logger.info(f"Starting transcription of: {audio_file_to_transcribe}")
        transcription = transcribe_audio(audio_file_to_transcribe, language=language)
            
        # Clean up temporary files
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
        
        # Clean up temporary files
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
        anonymization_level = data.get('level', 'standard')  # basic, standard, strict
        selected_model = data.get('model', DEFAULT_MODEL)  # Get selected model or use default

        # Handle local BERT model anonymization
        if selected_model == 'bert-base-ner':
            if not BERT_ANONYMIZER_AVAILABLE:
                return jsonify({'error': 'BERT anonymizer is not available on the server.'}), 500
            
            logger.info("Anonymizing text with local BERT model")
            try:
                anonymized_text, _ = anonymize_text_with_bert(text)
                return jsonify({
                    'success': True,
                    'anonymized_text': anonymized_text,
                    'level': 'n/a', # Anonymization level is not applicable for BERT
                    'model_used': 'Local BERT Model (German NER)',
                    'tts_available': GTTS_AVAILABLE
                })
            except Exception as e:
                logger.error(f"Error during BERT anonymization: {str(e)}")
                return jsonify({'error': f'BERT anonymization failed: {str(e)}'}), 500
        
        # Handle local spaCy model anonymization
        elif selected_model == 'spacy-de-ner':
            if not SPACY_ANONYMIZER_AVAILABLE:
                return jsonify({'error': 'spaCy anonymizer is not available on the server.'}), 500

            logger.info("Anonymizing text with local spaCy model")
            try:
                anonymized_text, _ = anonymize_text_with_spacy(text)
                return jsonify({
                    'success': True,
                    'anonymized_text': anonymized_text,
                    'level': 'n/a', # Anonymization level is not applicable
                    'model_used': 'Local spaCy Model',
                    'tts_available': GTTS_AVAILABLE
                })
            except Exception as e:
                logger.error(f"Error during spaCy anonymization: {str(e)}")
                return jsonify({'error': f'spaCy anonymization failed: {str(e)}'}), 500

        # Handle local Ensemble model anonymization
        elif selected_model == 'ensemble-spacy-bert':
            if not ENSEMBLE_ANONYMIZER_AVAILABLE:
                return jsonify({'error': 'Ensemble anonymizer is not available on the server.'}), 500
            
            logger.info("Anonymizing text with local Ensemble model")
            try:
                # The new ensemble function directly returns the final anonymized text
                anonymized_text = anonymize_text_with_ensemble(text)
                return jsonify({
                    'success': True,
                    'anonymized_text': anonymized_text,
                    'level': 'n/a', # Anonymization level is not applicable
                    'model_used': 'Local Ensemble (spaCy + BERT)',
                    'tts_available': GTTS_AVAILABLE
                })
            except Exception as e:
                logger.error(f"Error during Ensemble anonymization: {str(e)}")
                return jsonify({'error': f'Ensemble anonymization failed: {str(e)}'}), 500

        # Validate the selected LLM model
        if selected_model not in AVAILABLE_MODELS:
            return jsonify({'error': f'Invalid model selected: {selected_model}'}), 400
        
        if not CHAT_AI_API_KEY:
            return jsonify({'error': 'Chat AI API key not configured'}), 500
        
        # Define anonymization prompts based on level
        prompts = {
            'basic': """
            Anonymize the following text by replacing only the most obvious personally identifiable information (PII) using these specific labels:
            - Patient names with [NAME_PATIENT]
            - Doctor names with [NAME_DOCTOR]
            - Phone numbers with [CONTACT_PHONE]
            - Email addresses with [CONTACT_EMAIL]
            - Street addresses with [LOCATION_STREET]
            - Cities with [LOCATION_CITY]
            
            Keep the text as natural and readable as possible. Only replace obvious PII.
            
            CRITICAL INSTRUCTIONS: 
            - Do NOT anonymize or modify speaker identification tags like SPEAKER_00, SPEAKER_01, etc. These must be preserved exactly as they appear.
            - Do NOT translate any text. Keep ALL text in its original language exactly as it appears.
            - Preserve the original grammar, sentence structure, and language completely.
            - Only replace the specific PII elements mentioned above, leave everything else unchanged.
            
            IMPORTANT: Return ONLY the anonymized text. Do not include any explanations, reasoning, or additional commentary.
            """,
            'standard': """
            Anonymize the following text by replacing personally identifiable information (PII) using these specific labels:
            - Patient names with [NAME_PATIENT]
            - Doctor names with [NAME_DOCTOR]
            - Relative names with [NAME_RELATIVE]
            - Other names with [NAME_OTHER]
            - Professions with [PROFESSION]
            - Phone numbers with [CONTACT_PHONE]
            - Email addresses with [CONTACT_EMAIL]
            - Fax numbers with [CONTACT_FAX]
            - Street addresses with [LOCATION_STREET]
            - Cities with [LOCATION_CITY]
            - ZIP codes with [LOCATION_ZIP]
            - Hospitals with [LOCATION_HOSPITAL]
            - Organizations with [LOCATION_ORGANISATION]
            - Dates with [DATE]
            - Ages with [AGE]
            - ID numbers with [ID]
            
            Keep the text readable while ensuring privacy protection.
            
            CRITICAL INSTRUCTIONS: 
            - Do NOT anonymize or modify speaker identification tags like SPEAKER_00, SPEAKER_01, etc. These must be preserved exactly as they appear.
            - Do NOT translate any text. Keep ALL text in its original language exactly as it appears.
            - Preserve the original grammar, sentence structure, and language completely.
            - Only replace the specific PII elements mentioned above, leave everything else unchanged.
            
            IMPORTANT: Return ONLY the anonymized text. Do not include any explanations, reasoning, or additional commentary.
            """,
            'strict': """
            Thoroughly anonymize the following text by replacing all personally identifiable information (PII) using these specific labels:
            - Patient names with [NAME_PATIENT]
            - Doctor names with [NAME_DOCTOR]
            - Relative names with [NAME_RELATIVE]
            - External names with [NAME_EXT]
            - Usernames with [NAME_USERNAME]
            - Other names with [NAME_OTHER]
            - Titles with [NAME_TITLE]
            - Professions with [PROFESSION]
            - Dates with [DATE]
            - Ages with [AGE]
            - Street addresses with [LOCATION_STREET]
            - Cities with [LOCATION_CITY]
            - ZIP codes with [LOCATION_ZIP]
            - Countries with [LOCATION_COUNTRY]
            - States with [LOCATION_STATE]
            - Hospitals with [LOCATION_HOSPITAL]
            - Organizations with [LOCATION_ORGANISATION]
            - Other locations with [LOCATION_OTHER]
            - ID numbers with [ID]
            - Phone numbers with [CONTACT_PHONE]
            - Email addresses with [CONTACT_EMAIL]
            - Fax numbers with [CONTACT_FAX]
            - URLs with [CONTACT_URL]
            - Other contact information with [CONTACT_OTHER]
            
            Be thorough in removing all potentially identifying information using the appropriate specific labels.
            
            CRITICAL INSTRUCTIONS: 
            - Do NOT anonymize or modify speaker identification tags like SPEAKER_00, SPEAKER_01, etc. These must be preserved exactly as they appear.
            - Do NOT translate any text. Keep ALL text in its original language exactly as it appears.
            - Preserve the original grammar, sentence structure, and language completely.
            - Only replace the specific PII elements mentioned above, leave everything else unchanged.
            
            IMPORTANT: Return ONLY the anonymized text. Do not include any explanations, reasoning, or additional commentary.
            """
        }
        
        prompt = prompts.get(anonymization_level, prompts['standard'])
        
        model_name = AVAILABLE_MODELS.get(selected_model, selected_model)
        logger.info(f"Anonymizing text with {anonymization_level} level using {model_name}")
        
        # Prepare request to Chat AI API
        headers = {
            'Authorization': f'Bearer {CHAT_AI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Text to anonymize:\n\n{text}"}
            ],
            "max_tokens": 8000,  # Increased from 2000 to 8000 for much larger outputs
            "temperature": 0.1
        }
        
        # Try the request with retries and much longer timeout for large texts
        max_retries = 3
        timeout = 360  # Increased timeout to 6 minutes for very large text processing
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{CHAT_AI_ENDPOINT}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                break  # Success, exit retry loop
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Request timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    logger.error("All retry attempts failed due to timeout")
                    return jsonify({'error': 'The anonymization service is currently slow to respond. Please try again in a moment.'}), 504
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                return jsonify({'error': f'Connection error: {str(e)}'}), 503
        
        if response.status_code != 200:
            logger.error(f"Chat AI API error: {response.status_code} - {response.text}")
            return jsonify({'error': f'Chat AI API error: {response.status_code}'}), 500
        
        response_data = response.json()
        anonymized_text = response_data['choices'][0]['message']['content'].strip()
        
        # Remove any <think> tags and their content
        anonymized_text = re.sub(r'<think>.*?</think>', '', anonymized_text, flags=re.DOTALL).strip()
        
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
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Check if TTS is available
        if not GTTS_AVAILABLE:
            return jsonify({'error': 'Text-to-speech not available. Please install gtts.'}), 500
        
        # Get voice settings from request
        voice_settings = data.get('voice_settings', {
            'rate': 150,
            'volume': 0.8,
            'voice': 'de'
        })
        
        # Extract language from voice settings
        language = voice_settings.get('voice', 'de')
        
        logger.info(f"Generating speech for anonymized text with settings: {voice_settings}, language: {language}")
        
        # Generate speech
        audio_path, tts_engine = generate_speech(text, voice_settings, language)
        
        if audio_path and os.path.exists(audio_path):
            # Get the filename for the response
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
        # Security: only allow files in uploads directory with specific pattern
        if not filename.startswith('speech_output_') or '..' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Serve the audio file
        from flask import send_file
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error downloading speech file: {str(e)}")
        return jsonify({'error': f'Error downloading speech file: {str(e)}'}), 500

@app.route('/available_voices')
def get_available_voices():
    """Get German TTS voice"""
    try:
        # Only provide German gTTS voice
        if GTTS_AVAILABLE:
            voices = [
                {
                    'id': 'de', 
                    'name': 'Deutsch (Google TTS)', 
                    'language': ['de'], 
                    'engine': 'gtts'
                }
            ]
            logger.info("German TTS voice available")
        else:
            voices = []
            logger.warning("Google TTS not available")
        
        return jsonify({
            'voices': voices,
            'tts_available': GTTS_AVAILABLE,
            'gtts_available': GTTS_AVAILABLE
        })
    
    except Exception as e:
        logger.error(f"Error getting available voices: {str(e)}")
        return jsonify({'error': f'Error getting available voices: {str(e)}'}), 500

@app.route('/health')
def health_check():
    # Determine transcription capabilities
    if WHISPERX_AVAILABLE and whisperx_model is not None:
        if diarize_model is not None:
            transcription_type = 'whisperx-with-speakers'
            transcription_desc = 'WhisperX with speaker diarization'
        else:
            transcription_type = 'whisperx-basic'
            transcription_desc = 'WhisperX without speaker diarization'
    elif whisperx_model is not None: # No diarization model, so it's just basic WhisperX
        transcription_type = 'whisperx-basic'
        transcription_desc = 'WhisperX without speaker diarization'
    else:
        transcription_type = 'none'
        transcription_desc = 'No whisper model available'
    
    return jsonify({
        'status': 'healthy',
        'transcription_available': whisperx_model is not None,
        'transcription_type': transcription_type,
        'transcription_description': transcription_desc,
        'speaker_diarization': WHISPERX_AVAILABLE and diarize_model is not None,
        'whisperx_available': WHISPERX_AVAILABLE,
        'faster_whisper_available': False, # Removed as faster-whisper is no longer used
        'chat_ai_configured': CHAT_AI_API_KEY is not None,
        'default_model': DEFAULT_MODEL,
        'available_models': list(AVAILABLE_MODELS.keys()),
        'tts_available': GTTS_AVAILABLE,
        'tts_engine': 'gtts' if GTTS_AVAILABLE else None
    })

def create_self_signed_cert():
    """Create a self-signed certificate for HTTPS"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Localhost"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Speech Anonymizer"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress("127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Write certificate and key to files
        with open("cert.pem", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open("key.pem", "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        return "cert.pem", "key.pem"
    except ImportError:
        logger.warning("cryptography package not available, using ad-hoc SSL context")
        return None, None

if __name__ == '__main__':
    # Try to use HTTPS for microphone access
    use_https = os.getenv('USE_HTTPS', 'true').lower() == 'true'
    
    if use_https:
        try:
            cert_file, key_file = create_self_signed_cert()
            if cert_file and key_file:
                logger.info("Starting server with HTTPS (self-signed certificate)")
                logger.info("⚠️  You may need to accept the security warning in your browser")
                logger.info("🌐 Access the app at: https://localhost:5001")
                app.run(debug=True, host='0.0.0.0', port=5001, ssl_context=(cert_file, key_file))
            else:
                logger.info("Starting server with HTTPS (ad-hoc certificate)")
                logger.info("⚠️  You may need to accept the security warning in your browser")
                logger.info("🌐 Access the app at: https://localhost:5001")
                app.run(debug=True, host='0.0.0.0', port=5001, ssl_context='adhoc')
        except Exception as e:
            logger.error(f"Failed to start HTTPS server: {e}")
            logger.info("Falling back to HTTP (microphone may not work)")
            logger.info("🌐 Access the app at: http://localhost:5001")
            app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.info("Starting server with HTTP")
        logger.info("🌐 Access the app at: http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001) 