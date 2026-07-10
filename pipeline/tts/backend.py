"""
pipeline/tts/backend.py
Unified TTS backend abstraction supporting Piper and MeloTTS
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
        'en': 'en_US-ryan-high',
        'de': 'de_DE-thorsten-high',
        'fr': 'fr_FR-siwis-medium',
        'es': 'es_ES-carlfm-x_low',
        'nl': 'nl_NL-mls-medium',
        'it': 'it_IT-riccardo-x_low',
        'pl': 'pl_PL-gosia-medium',
        'pt': 'pt_BR-faber-medium',
    }

    def _initialize(self):
        logger.info("Initializing Piper TTS backend...")

        self.bin_path = Path(self.config.get(
            'TTS_BIN_PATH',
            './pipeline/tts/bin/piper'
        ))
        self.voice_dir = Path(self.config.get(
            'TTS_VOICE_DIR',
            './pipeline/tts/voices'
        ))
        self.sample_rate = int(self.config.get('TTS_SAMPLE_RATE', 22050))

        if not self.bin_path.exists():s
            raise TTSError(f"Piper binary not found: {self.bin_path}")

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

        # Switch language if needed
        lang_key = language.lower()[:2]
        if self._current_lang != lang_key:
            try:
                self.switch_language(lang_key)
            except TTSError as e:
                logger.warning(f"Language switch failed: {e}. Using current model ({self._current_lang}).")

        output_path = str(Path(output_path).absolute())

        cmd = [
            str(self.bin_path),
            '-m', str(self._voice_path),
            '-o', output_path,
            '--sample_rate', str(self.sample_rate),
        ]

        try:
            proc = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                check=True
            )
            logger.debug(f"Piper synthesized: {text[:50]}... -> {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise TTSError(f"Piper error: {e.stderr.decode(errors='replace')}") from e

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
            voice_name = self.DEFAULT_VOICES.get(lang_key, 'en_US-ryan-high')
            voices.append({
                'id': voice_name,
                'name': f'{lang_key.upper()} Default (Piper)',
                'language': [lang_key],
                'engine': 'piper',
            })
        return voices


class MeloTTSBackend(TTSBackend):
    """MeloTTS implementation with multilingual support."""

    # MeloTTS model repos per language
    MODEL_REPOS = {
        'en': 'myshell-ai/MeloTTS-English',
        'de': 'myshell-ai/MeloTTS-German',
        'fr': 'myshell-ai/MeloTTS-French',
        'es': 'myshell-ai/MeloTTS-Spanish',
        'zh': 'myshell-ai/MeloTTS-Chinese',
        'ja': 'myshell-ai/MeloTTS-Japanese',
        'ko': 'myshell-ai/MeloTTS-Korean',
    }

    # Speaker name mapping per language
    SPEAKER_MAP = {
        'en': {'0': 'EN-US', '1': 'EN-GB', '2': 'EN-India', 'default': 'EN-Default'},
        'de': {'0': 'DE', 'default': 'DE'},
        'fr': {'0': 'FR', 'default': 'FR'},
        'es': {'0': 'ES', 'default': 'ES'},
        'zh': {'0': 'ZH', 'default': 'ZH'},
        'ja': {'0': 'JP', 'default': 'JP'},
        'ko': {'0': 'KO', 'default': 'KO'},
    }

    def _initialize(self):
        logger.info("Initializing MeloTTS backend...")

        try:
            from melotts import MeloTTS
            self.MeloTTSClass = MeloTTS
        except ImportError as e:
            raise TTSError(
                "melotts not installed. Run: "
                "pip install git+https://github.com/myshell-ai/MeloTTS.git"
            ) from e

        self._models: Dict[str, object] = {}  # Cache loaded models per language
        self._current_model = None

        # Load default language model
        default_lang = self.config.get('TTS_DEFAULT_LANG', 'en')
        self.switch_language(default_lang)

    def switch_language(self, language: str):
        lang_key = language.lower()[:2]

        if lang_key not in self.MODEL_REPOS:
            logger.warning(f"MeloTTS: No model for '{lang_key}', falling back to English.")
            lang_key = 'en'

        # Return cached model if already loaded
        if lang_key in self._models:
            self._current_model = self._models[lang_key]
            self._current_lang = lang_key
            return

        model_name = self.config.get('TTS_MODEL_NAME') or self.MODEL_REPOS[lang_key]

        # If a specific model name is configured and doesn't match the language repo,
        # use it only for the default language
        if lang_key != self.config.get('TTS_DEFAULT_LANG', 'en')[:2]:
            model_name = self.MODEL_REPOS[lang_key]

        try:
            logger.info(f"MeloTTS: Loading model for {lang_key}: {model_name}")
            model = self.MeloTTSClass.from_pretrained(model_name)
            self._models[lang_key] = model
            self._current_model = model
            self._current_lang = lang_key
            logger.info(f"✅ MeloTTS loaded: {lang_key}")
        except Exception as e:
            raise TTSError(f"Failed to load MeloTTS for {lang_key}: {e}") from e

    def _get_speaker_key(self, speaker_id: int, language: str) -> str:
        lang_key = language.lower()[:2]
        speakers = self.SPEAKER_MAP.get(lang_key, self.SPEAKER_MAP['en'])
        key = str(speaker_id) if str(speaker_id) in speakers else 'default'
        return speakers.get(key, speakers['default'])

    def synthesize(self, text: str, output_path: str, speaker_id: int = 0,
                   language: str = 'en', speed: float = 1.0, **kwargs) -> str:
        # Switch language if needed
        lang_key = language.lower()[:2]
        if self._current_lang != lang_key:
            try:
                self.switch_language(lang_key)
            except TTSError as e:
                logger.warning(f"Language switch failed: {e}. Using current model ({self._current_lang}).")

        if not self._current_model:
            raise TTSError("No MeloTTS model loaded.")

        output_path = str(Path(output_path).absolute())
        speaker_key = self._get_speaker_key(speaker_id, language)

        # Get the actual speaker ID from the model
        speaker_ids = getattr(self._current_model, 'speakers', {})
        sid = speaker_ids.get(speaker_key, 0)

        try:
            self._current_model.tts_to_file(
                text=text,
                speaker_id=sid,
                file_path=output_path,
                speed=speed,
            )
            logger.debug(f"MeloTTS synthesized: {text[:50]}... -> {output_path}")
            return output_path
        except Exception as e:
            raise TTSError(f"MeloTTS synthesis failed: {e}") from e

    def get_available_voices(self, language: str = 'en') -> List[Dict]:
        lang_key = language.lower()[:2]
        speakers = self.SPEAKER_MAP.get(lang_key, self.SPEAKER_MAP['en'])
        voices = []
        for speaker_id, speaker_name in speakers.items():
            if speaker_id == 'default':
                continue  # Skip internal 'default' key
            voices.append({
                'id': speaker_id,
                'name': f"{speaker_name} (MeloTTS)",
                'language': [lang_key],
                'engine': 'melotts',
            })
        return voices


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
        'melotts': MeloTTSBackend,
    }

    if backend_name not in backends:
        raise TTSError(f"Unknown TTS backend: {backend_name}. Choices: {list(backends.keys())}")

    logger.info(f"Loading TTS backend: {backend_name}")
    return backends[backend_name](config)


def init_tts(backend: str, **kwargs) -> TTSBackend:
    """Direct initialization with inline config."""
    config = {'TTS_BACKEND': backend, **kwargs}
    classes = {'piper': PiperBackend, 'melotts': MeloTTSBackend}
    if backend not in classes:
        raise TTSError(f"Unknown backend: {backend}")
    return classes[backend](config)