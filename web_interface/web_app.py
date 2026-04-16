#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
import threading
import time
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# --- Dynamic Path Resolution ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Pipeline paths (Relative to project root)
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
VIDEOS_FOLDER = PIPELINE_DIR / "videos"
AUDIOS_FOLDER = PIPELINE_DIR / "audios"
TRANSCRIPTS_FOLDER = PIPELINE_DIR / "transcripts"
ANNONYM_FOLDER = PIPELINE_DIR / "annonym"
MODEL_FOLDER = PIPELINE_DIR / "model"

# Web interface paths
TEMPLATES_DIR = SCRIPT_DIR / "templates"
STATIC_DIR = SCRIPT_DIR / "static"
LOGS_DIR = PROJECT_ROOT / "logs"
UPLOAD_FOLDER = SCRIPT_DIR / "uploads"

# Ensure directories exist
for folder in [VIDEOS_FOLDER, AUDIOS_FOLDER, TRANSCRIPTS_FOLDER, ANNONYM_FOLDER, LOGS_DIR, UPLOAD_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# --- Logging Configuration ---
log_file = LOGS_DIR / "web_app.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Import Pipeline Helpers ---
sys.path.insert(0, str(PIPELINE_DIR))
try:
    from pipeline import sanitize_filename, validate_path
    logger.info("Successfully imported pipeline helpers.")
except ImportError as e:
    logger.critical(f"Failed to import pipeline helpers: {e}")
    sys.exit(1)

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Track pipeline status
pipeline_status = {
    'running': False,
    'current_step': None,
    'progress': 0,
    'last_updated': None,
    'errors': []
}

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mp3', 'mkv'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/')
def index():
    # Note: You will need to create a simple index.html for this later or reuse the interactive one
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({'error': 'Unsupported file format'}), 400
        
        filename = secure_filename(file.filename)
        filename = sanitize_filename(filename)
        target_path = VIDEOS_FOLDER / filename
        
        if not validate_path(target_path, VIDEOS_FOLDER):
            return jsonify({'error': 'Invalid file path'}), 400
        
        file.save(str(target_path))
        logger.info(f"Video uploaded: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(target_path)
        })
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    global pipeline_status
    
    if pipeline_status['running']:
        return jsonify({'error': 'Pipeline already running'}), 409
    
    def run_pipeline_async():
        global pipeline_status
        pipeline_status['running'] = True
        pipeline_status['current_step'] = 'Initializing'
        pipeline_status['progress'] = 0
        pipeline_status['errors'] = []
        
        try:
            from pipeline import process_videos, process_audios
            
            pipeline_status['current_step'] = 'Extracting Audio'
            pipeline_status['progress'] = 25
            process_videos()
            
            pipeline_status['current_step'] = 'Transcribing & Diarizing'
            pipeline_status['progress'] = 50
            process_audios()
            
            pipeline_status['progress'] = 100
            pipeline_status['current_step'] = 'Complete'
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            pipeline_status['errors'].append(str(e))
            pipeline_status['current_step'] = 'Error'
        finally:
            pipeline_status['running'] = False
            pipeline_status['last_updated'] = time.time()
    
    thread = threading.Thread(target=run_pipeline_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Pipeline started in background'
    })

@app.route('/pipeline_status')
def get_pipeline_status():
    return jsonify(pipeline_status)

@app.route('/get_transcripts')
def get_transcripts():
    try:
        if not TRANSCRIPTS_FOLDER.exists():
            return jsonify({'transcripts': []})
        
        transcripts = []
        for file in TRANSCRIPTS_FOLDER.iterdir():
            if file.suffix == '.txt':
                transcripts.append({
                    'filename': file.name,
                    'path': str(file),
                    'size': file.stat().st_size
                })
        
        return jsonify({'transcripts': transcripts})
    
    except Exception as e:
        logger.error(f"Error listing transcripts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_transcript/<filename>')
def download_transcript(filename):
    try:
        file_path = TRANSCRIPTS_FOLDER / filename
        
        if not validate_path(file_path, TRANSCRIPTS_FOLDER):
            return jsonify({'error': 'Invalid file path'}), 400
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(str(file_path), as_attachment=True)
    
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'pipeline_base': str(PIPELINE_DIR),
        'pipeline_running': pipeline_status['running']
    })

if __name__ == '__main__':
    logger.info("Starting VERANDA Batch Pipeline Interface")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Pipeline Base: {PIPELINE_DIR}")
    # Port 5001 for Batch Mode
    app.run(debug=True, host='0.0.0.0', port=7890)