"""
pipeline/tts/backend.py
Unified TTS backend abstraction supporting Piper and Coqui TTS
Updated for NumPy 2.x compatibility and Python 3.13
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TTSError(Exception):
    pass


class TTSBackend(ABC):
    """Abstract base class for all TTS engines"""

    def __init__(self, config: Dict):
        self.config = config
        self.backend_name = config.get('TTS_BACKEND', 'piper').lower()
        self._current_lang: Optional[str] = None
        self._initialize()

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def synthesize(self, text: str, output_path: str, speaker_id: int = 0,
                   language: str = 'en', **kwargs) -> str:
        pass

    @abstractmethod
    def get_available_voices(self, language: str = 'en') -> List[Dict]:
        pass

    @abstractmethod
    def switch_language(self, language: str):
        """Switch the loaded model to a different language if supported."""
        pass


class PiperBackend(TTSBackend):
    """Piper TTS via direct binary execution."""

    # Voice model directory structure: {lang}_{region}-{voice}-{quality}
    # e.g. en_US-ryan-high, de_DE-thorsten-high
    DEFAULT_VOICES = {
        'en': 'en_US-lessac-medium',  # Updated from ryan-high to match installer
        'de': 'de_DE-thorsten-high',
        'fr': 'fr_FR-siwis-medium',
        'es': 'es_ES-carlfm-x_low',
        'nl': 'nl_NL-mls-medium',
        'it': 'it_IT-riccardo-x_low',
        'pl': 'pl_PL-gosia-medium',
        'pt': 'pt_BR-faber-medium',
        'fi': 'fi_Finnish-medium',
        'ar': 'ar_arabic-medium',
        'hi': 'hi_hindi-medium',
        'tr': 'tr_turkish-medium',
    }

    def _initialize(self):
        logger.info("Initializing Piper TTS backend...")

        self.bin_path = Path(self.config.get(
            'TTS_BIN_PATH',
            './pipeline/tts/bin/piper'
        ))
        self.voice_dir = Path(self.config.get(
            'TTS_VOICE_DIR',
            './pipeline/model/piper-voices'  # Updated path to match installer
        ))
        self.sample_rate = int(self.config.get('TTS_SAMPLE_RATE', 22050))

        # Check if configured voice path takes precedence
        configured_voice = Path(self.config.get('TTS_VOICE_PATH', ''))
        if configured_voice.exists():
            self._voice_path = configured_voice
            logger.info(f"Using configured Piper voice: {configured_voice}")
        elif not self.bin_path.exists():
            logger.warning(f"Piper binary not found: {self.bin_path}")
            # Don't fail immediately - will check at synthesis time

        # Resolve voice for default language
        default_lang = self.config.get('TTS_DEFAULT_LANG', 'en')
        self.switch_language(default_lang)

    def switch_language(self, language: str):
        lang_key = language.lower()[:2]
        voice_name = self.DEFAULT_VOICES.get(lang_key, self.DEFAULT_VOICES['en'])
        voice_path = self.voice_dir / f"{voice_name}.onnx"

        if not voice_path.exists():
            # Try the config-specified voice as fallback
            configured = Path(self.config.get('TTS_VOICE_PATH', ''))
            if configured.exists():
                voice_path = configured
                logger.info(f"Falling back to configured voice: {configured.name}")
            else:
                raise TTSError(
                    f"Voice model not found: {voice_path}. "
                    f"Download from https://huggingface.co/rhasspy/piper-voices"
                )

        self._voice_path = voice_path
        self._current_lang = lang_key
        logger.info(f"Piper voice set: {voice_path.name} (lang={lang_key})")

    def synthesize(self, text: str, output_path: str, speaker_id: int = 0,
                language: str = 'en', **kwargs) -> str:
        import subprocess

        # Check binary availability
        if not self.bin_path.exists():
            # Try alternative binary paths
            alt_paths = [
                Path('./pipeline/tts/bin/piper-linux-x86_64'),
                Path('/usr/local/bin/piper'),
                Path('/usr/bin/piper'),
            ]
            for alt in alt_paths:
                if alt.exists():
                    self.bin_path = alt
                    logger.info(f"Found Piper binary at: {alt}")
                    break
            else:
                raise TTSError(f"Piper binary not found at {self.bin_path} or alternatives")

        # Switch language if needed
        lang_key = language.lower()[:2]
        if self._current_lang != lang_key:
            try:
                self.switch_language(lang_key)
            except TTSError as e:
                logger.warning(f"Language switch failed: {e}. Using current model ({self._current_lang}).")

        output_path = str(Path(output_path).absolute())
        sample_rate = int(self.config.get('TTS_SAMPLE_RATE', 22050))
        cmd = [
            str(self.bin_path),
            '-m', str(self._voice_path),
            '-o', output_path,
            '--sample_rate', str(sample_rate),
        ]

        # Add speaker ID if supported
        if speaker_id > 0 and hasattr(self, '_voice_config'):
            cmd.extend(['--speaker_id', str(speaker_id)])

        try:
            proc = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                check=True,
                timeout=300  # 5 minute timeout for long texts
            )
            logger.debug(f"Piper synthesized: {text[:50]}... -> {output_path}")
            return output_path
        except subprocess.TimeoutExpired:
            raise TTSError("Piper synthesis timed out (text too long?)")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors='replace') if e.stderr else "unknown error"
            raise TTSError(f"Piper error: {stderr}") from e

    def get_available_voices(self, language: str = 'en') -> List[Dict]:
        lang_key = language.lower()[:2]
        voices = []
        for onnx_file in sorted(self.voice_dir.glob(f"{lang_key}_*.onnx")):
            voices.append({
                'id': onnx_file.stem,
                'name': onnx_file.stem.replace('_', ' ').title(),
                'language': [lang_key],
                'engine': 'piper',
            })
        if not voices:
            # Return default voice info
            voice_name = self.DEFAULT_VOICES.get(lang_key, 'en_US-lessac-medium')
            voices.append({
                'id': voice_name,
                'name': f'{lang_key.upper()} Default (Piper)',
                'language': [lang_key],
                'engine': 'piper',
            })
        return voices


class CoquiXTTSBackend(TTSBackend):
    """Coqui XTTS v2 implementation with multilingual support."""

    MODEL_REPOS = {
        'en': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'de': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'fr': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'es': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'it': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'pl': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'pt': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'tr': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'zh': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'ja': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'ko': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'ar': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'hi': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'fi': 'tts_models/multilingual/multi-dataset/xtts_v2',
    }

    SPEAKER_MAP = {
        'default': 'default',
    }

    def _initialize(self):
        logger.info("Initializing Coqui XTTS v2 backend...")
        
        # Set environment variable before importing TTS
        os.environ['COQUI_TOS_AGREED'] = '1'

        try:
            from TTS.api import TTS
            self.TTSClass = TTS
        except ImportError as e:
            raise TTSError(
                "TTS (Coqui) not installed. Run: "
                "pip install TTS>=0.27.0"
            ) from e

        self._model = None
        self._model_path = None
        
        # Load model from configured path or download
        model_path = self.config.get('XTTS_Model_Path', '')
        if model_path and Path(model_path).exists():
            self._model_path = Path(model_path)
            logger.info(f"Using existing Coqui model at: {model_path}")
        else:
            logger.info("Downloading Coqui XTTS v2 model (~2GB)...")
            self._model = self.TTSClass('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)
            self._model_path = Path(self._model.model_path)

        default_lang = self.config.get('TTS_DEFAULT_LANG', 'en')
        self.switch_language(default_lang)

    def switch_language(self, language: str):
        lang_key = language.lower()[:2]
        
        if lang_key not in self.MODEL_REPOS:
            logger.warning(f"Coqui: No dedicated model for '{lang_key}', using multilingual.")
        
        self._current_lang = lang_key
        logger.info(f"Coqui XTTS ready for language: {lang_key}")

    def synthesize(self, text: str, output_path: str, speaker_id: int = 0,
                   language: str = 'en', speed: float = 1.0, **kwargs) -> str:
        import torch
        
        # Ensure model is loaded
        if self._model is None:
            try:
                from TTS.api import TTS
                self._model = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)
            except Exception as e:
                raise TTSError(f"Failed to load Coqui model: {e}") from e

        # Switch language if needed (Coqui uses multilingual model)
        lang_key = language.lower()[:2]
        if self._current_lang != lang_key:
            self.switch_language(lang_key)

        output_path = str(Path(output_path).absolute())

        # Get reference audio for voice cloning (optional)
        ref_audio = self.config.get('XTTS_Reference_Audio_Path', '')
        speaker_wav = ref_audio if ref_audio and Path(ref_audio).exists() else None

        try:
            # Synthesize with Coqui
            self._model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=lang_key,
                speed=speed,
            )
            logger.debug(f"Coqui XTTS synthesized: {text[:50]}... -> {output_path}")
            return output_path
        except Exception as e:
            raise TTSError(f"Coqui XTTS synthesis failed: {e}") from e

    def get_available_voices(self, language: str = 'en') -> List[Dict]:
        # Coqui uses voice cloning, so we return a generic response
        lang_key = language.lower()[:2]
        return [{
            'id': 'default',
            'name': f'{lang_key.upper()} (Coqui XTTS v2 - Voice Cloning Enabled)',
            'language': [lang_key],
            'engine': 'coqui_xtts',
        }]


def get_tts_backend(config: Optional[Dict] = None, config_path: str = '.env') -> TTSBackend:
    """Factory: load TTS backend from .env or config dict."""
    if config is None:
        try:
            from dotenv import dotenv_values
            config = dotenv_values(config_path)
        except ImportError:
            config = {}
        # Merge with environment variables
        config.update({k: v for k, v in os.environ.items() if k.startswith('TTS_')})

    backend_name = config.get('TTS_BACKEND', 'piper').lower()

    backends = {
        'piper': PiperBackend,
        'coqui_xtts': CoquiXTTSBackend,  # Updated from melotts to coqui_xtts
        'melotts': None,  # Deprecated - removed
    }

    if backend_name not in backends:
        raise TTSError(f"Unknown TTS backend: {backend_name}. Choices: ['piper', 'coqui_xtts']")
    
    if backends[backend_name] is None:
        raise TTSError(f"TTS backend '{backend_name}' has been deprecated. Use 'piper' or 'coqui_xtts'.")

    logger.info(f"Loading TTS backend: {backend_name}")
    return backends[backend_name](config)


def init_tts(backend: str, **kwargs) -> TTSBackend:
    """Direct initialization with inline config."""
    config = {'TTS_BACKEND': backend, **kwargs}
    classes = {
        'piper': PiperBackend, 
        'coqui_xtts': CoquiXTTSBackend,
    }
    if backend not in classes:
        raise TTSError(f"Unknown backend: {backend}. Choices: ['piper', 'coqui_xtts']")
    return classes[backend](config)