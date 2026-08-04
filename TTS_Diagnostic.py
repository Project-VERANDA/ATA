#!/usr/bin/env python3
"""
TTS Diagnostic Script for Dialogue Anonymizer
Tests Piper and Coqui XTTS functionality systematically.
Run on Ubuntu VM: python3 tts_diagnostic.py
"""

import os
import sys
import subprocess
import json
import wave
import io
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TSSEngineDiagnostic:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'overall_status': 'PASS',
            'recommendations': []
        }
        
        # Configuration paths (matching tts_engine.py)
        self.pipeline_dir = PROJECT_ROOT / "pipeline"
        self.tts_dir = self.pipeline_dir / "tts"
        self.piper_executable = self.tts_dir / "bin" / "piper"
        self.piper_voice_dir = self.tts_dir / "voices"
        self.voice_mapping = self.piper_voice_dir / "VOICE_MAPPING.txt"
        
        # Test parameters
        self.test_text = "Hello, this is a TTS diagnostic test. Testing audio synthesis."
        self.sample_rate = 22050
        
    def log_test(self, name, status, details="", duration=None):
        """Record test result."""
        test_entry = {
            'name': name,
            'status': status,  # PASS, WARN, FAIL
            'details': details,
            'duration_sec': duration
        }
        self.results['tests'].append(test_entry)
        
        if status == 'FAIL':
            self.results['overall_status'] = 'FAIL'
            self.results['recommendations'].append(
                f"{name}: {details}"
            )
            
        print(f"{'✅' if status == 'PASS' else '⚠️' if status == 'WARN' else '❌'} {name}: {status}")
        if details:
            print(f"   {details}")
            
    def test_1_piper_binary_exists(self):
        """Check if Piper executable exists and has correct permissions."""
        print("\n[TEST 1] Piper Binary Existence")
        
        if not self.piper_executable.exists():
            self.log_test(
                "Piper Executable Found", 
                'FAIL', 
                f"Not found at {self.piper_executable}"
            )
            self.log_test(
                "Installer Fix Needed",
                'WARN',
                "Re-run Installer.sh or manually download from https://github.com/rhasspy/piper/releases"
            )
            return False
            
        if not os.access(self.piper_executable, os.X_OK):
            self.log_test(
                "Piper Execute Permissions",
                'WARN',
                "Binary exists but not executable"
            )
            try:
                os.chmod(self.piper_executable, 0o755)
                self.log_test("Fixed Permissions", 'PASS')
            except Exception as e:
                self.log_test("Permission Fix", 'FAIL', str(e))
                return False
                
        file_info = subprocess.run(['file', str(self.piper_executable)], 
                                   capture_output=True, text=True)
        
        self.log_test(
            "Binary File Type", 
            'PASS' if 'ELF' in file_info.stdout else 'WARN',
            file_info.stdout.strip() if file_info.returncode == 0 else "Unknown"
        )
        
        return True
        
    def test_2_voice_models_exist(self):
        """Verify voice .onnx models are downloaded and valid."""
        print("\n[TEST 2] Voice Models")
        
        if not self.piper_voice_dir.exists():
            self.log_test(
                "Voice Directory Exists",
                'FAIL',
                f"Path not found: {self.piper_voice_dir}"
            )
            return False
            
        onnx_files = list(self.piper_voice_dir.glob("*.onnx"))
        valid_voices = []
        
        for onnx_file in onnx_files:
            size_mb = onnx_file.stat().st_size / (1024 * 1024)
            if size_mb >= 40:  # Valid ONNX model should be >40MB
                valid_voices.append({
                    'name': onnx_file.stem,
                    'size_mb': round(size_mb, 1)
                })
            else:
                self.log_test(
                    f"Corrupt Model: {onnx_file.stem}",
                    'WARN',
                    f"Only {size_mb:.1f} MB (should be >40MB)"
                )
                
        self.log_test(
            f"Valid Voices ({len(valid_voices)})",
            'PASS' if len(valid_voices) >= 2 else 'FAIL',
            f"Found {len(valid_voices)} valid voice model(s)"
        )
        
        for voice in valid_voices[:3]:  # Show first 3
            print(f"   • {voice['name']} ({voice['size_mb']} MB)")
            
        if len(valid_voices) < 2:
            self.log_test(
                "Voice Download Script",
                'WARN',
                "Need minimum 2 voices for multi-speaker TTS. Re-run installer with Piper backend."
            )
            return False
            
        return True
        
    def test_3_library_dependencies(self):
        """Check shared library dependencies with ldd."""
        print("\n[TEST 3] Library Dependencies")
        
        if not self.piper_executable.exists():
            self.log_test("LDd Check", 'SKIP', "Binary not found")
            return False
            
        result = subprocess.run(['ldd', str(self.piper_executable)], 
                                capture_output=True, text=True)
        
        lines = result.stdout.split('\n')
        missing_libs = [line.strip() for line in lines if 'not found' in line]
        
        if missing_libs:
            self.log_test(
                "Shared Libraries",
                'FAIL',
                f"Missing: {', '.join(missing_libs)}"
            )
            self.log_test(
                "Fix Missing Libs",
                'WARN',
                "Install with: sudo apt-get install libportaudio2 libsndfile1"
            )
        else:
            self.log_test(
                "Shared Libraries",
                'PASS',
                "All dependencies resolved"
            )
            
        return len(missing_libs) == 0
        
    def test_4_direct_synthesis_command(self):
        """Test Piper synthesis using direct CLI invocation."""
        print("\n[TEST 4] Direct Piper Synthesis")
        
        # Find a valid voice model
        onnx_files = list(self.piper_voice_dir.glob("*.onnx"))
        if not onnx_files:
            self.log_test("Direct Synthesis", 'FAIL', "No voice models found")
            return False
            
        voice_path = onnx_files[0]
        output_path = Path('/tmp/piper_test.wav')
        
        cmd = [
            str(self.piper_executable),
            '-m', str(voice_path),
            '-o', str(output_path),
            '--sample_rate', str(self.sample_rate)
        ]
        
        env = os.environ.copy()
        bin_dir = str(self.tts_dir / 'bin')
        env['LD_LIBRARY_PATH'] = bin_dir + ':' + env.get('LD_LIBRARY_PATH', '')
        
        try:
            proc = subprocess.run(
                cmd,
                input=self.test_text,
                text=True,
                capture_output=True,
                timeout=30,
                env=env
            )
            
            if proc.returncode != 0:
                self.log_test(
                    "CLI Invocation",
                    'FAIL',
                    f"Exit code {proc.returncode}: {proc.stderr.decode()[:200]}"
                )
                return False
                
            if not output_path.exists():
                self.log_test(
                    "Output File Created",
                    'FAIL',
                    "No WAV file generated"
                )
                return False
                
            size_kb = output_path.stat().st_size / 1024
            if size_kb < 10:  # Too small, likely empty
                self.log_test(
                    "Audio Output Validity",
                    'FAIL',
                    f"File too small: {size_kb:.1f} KB"
                )
                return False
                
            # Verify WAV headers
            try:
                with wave.open(str(output_path), 'rb') as wav:
                    channels = wav.getnchannels()
                    sampwidth = wav.getsampwidth()
                    framerate = wav.getframerate()
                    duration_sec = wav.getnframes() / framerate
                    
                self.log_test(
                    "Audio Quality",
                    'PASS',
                    f"{channels}ch, {sampwidth*8}bit, {framerate}Hz, {duration_sec:.2f}s"
                )
            except Exception as e:
                self.log_test(
                    "WAV Validation",
                    'FAIL',
                    f"Invalid WAV: {e}"
                )
                return False
                
            # Cleanup
            output_path.unlink()
            return True
            
        except subprocess.TimeoutExpired:
            self.log_test("CLI Timeout", 'FAIL', "Command exceeded 30s")
            return False
        except FileNotFoundError:
            self.log_test("CLI Executable", 'FAIL', "Piper binary not found in PATH")
            return False
        except Exception as e:
            self.log_test("CLI Error", 'FAIL', str(e)[:200])
            return False
            
    def test_5_python_piper_package(self):
        """Test Python Piper package as alternative."""
        print("\n[TEST 5] Python Piper Package")
        
        try:
            from piper import PiperVoice
            self.log_test(
                "Import piper module",
                'PASS'
            )
        except ImportError as e:
            self.log_test(
                "Import piper module",
                'WARN',
                f"Not installed: {e}"
            )
            self.log_test(
                "Install piper-tts",
                'INFO',
                "pip install piper-tts"
            )
            return False
            
        onnx_files = list(self.piper_voice_dir.glob("*.onnx"))
        if not onnx_files:
            self.log_test("Voice Model Available", 'FAIL', "No .onnx files found")
            return False
            
        voice_path = onnx_files[0]
        print(f"   Testing voice: {voice_path.stem}")
        
        try:
            voice = PiperVoice.load(str(voice_path))
            self.log_test(
                "Voice Loading",
                'PASS'
            )
        except Exception as e:
            self.log_test(
                "Voice Loading",
                'FAIL',
                str(e)[:200]
            )
            return False
            
        try:
            chunks = list(voice.synthesize(self.test_text))
            if not chunks:
                self.log_test("Synthesis Output", 'FAIL', "No audio chunks produced")
                return False
                
            self.log_test(
                "Python Synthesis",
                'PASS',
                f"Generated {len(chunks)} chunk(s)"
            )
            
            # Test WAV writing
            buffer = io.BytesIO()
            sample_rate = self.sample_rate
            
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(chunks[0].sample_channels)
                wav.setsampwidth(chunks[0].sample_width)
                wav.setframerate(sample_rate)
                for chunk in chunks:
                    wav.writeframes(chunk.audio_int16_bytes)
                    
            self.log_test(
                "WAV Serialization",
                'PASS',
                f"Buffer size: {len(buffer.getvalue()) / 1024:.1f} KB"
            )
            
        except Exception as e:
            self.log_test(
                "Synthesis Execution",
                'FAIL',
                str(e)[:200]
            )
            return False
            
        return True
        
    def test_6_coqui_xtts_availability(self):
        """Check Coqui XTTS as fallback option."""
        print("\n[TEST 6] Coqui XTTS Fallback")
        
        try:
            from TTS.api import TTS
            self.log_test(
                "Import TTS",
                'PASS'
            )
        except ImportError:
            self.log_test(
                "Import TTS",
                'WARN',
                "Coqui not installed"
            )
            self.log_test(
                "Install Coqui XTTS",
                'INFO',
                "pip install TTS"
            )
            self.results['recommendations'].append(
                "Consider enabling Coqui XTTS in installer for alternative backend"
            )
            return False
            
        try:
            # Quick availability check (don't actually download model yet)
            available = TTS.list_models()
            self.log_test(
                "TTS Module Initialized",
                'PASS',
                f"TTS version ready"
            )
        except Exception as e:
            self.log_test(
                "TTS Initialization",
                'WARN',
                str(e)[:200]
            )
            
        # Note: Actual model download would take minutes, skipping for diagnostic
        self.log_test(
            "Full Coqui Test",
            'SKIP',
            "Model download not tested in diagnostic mode"
        )
        
        self.results['recommendations'].append(
            "Coqui XTTS available for installation if Piper fails persistently"
        )
        
    def test_7_ld_library_path(self):
        """Check LD_LIBRARY_PATH configuration."""
        print("\n[TEST 7] Environment Variables")
        
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        bin_dir = str(self.tts_dir / 'bin')
        
        if bin_dir in ld_path:
            self.log_test(
                "LD_LIBRARY_PATH Includes TTS Bin",
                'PASS'
            )
        else:
            self.log_test(
                "LD_LIBRARY_PATH Includes TTS Bin",
                'WARN',
                f"Should include: {bin_dir}"
            )
            self.log_test(
                "Fix Recommendation",
                'INFO',
                "Export: export LD_LIBRARY_PATH=/path/to/ATA/pipeline/tts/bin:$LD_LIBRARY_PATH"
            )
            
        piper_env = os.environ.get('Piper_Voice_Path', 'Not set')
        if piper_env != 'Not set':
            self.log_test(
                "Piper_Voice_Path Env Var",
                'PASS'
            )
        else:
            self.log_test(
                "Piper_Voice_Path Env Var",
                'WARN',
                "Not configured in environment"
            )
            
    def test_8_integration_import(self):
        """Test importing from actual tts_engine.py module."""
        print("\n[TEST 8] Integration with tts_engine.py")
        
        try:
            # Need to add pipeline/tts to path
            tts_path = self.tts_dir
            if str(tts_path) not in sys.path:
                sys.path.insert(0, str(tts_path))
                
            from tts_engine import get_tts_status, generate_speech, SPEAKER_VOICE_MAP
            self.log_test(
                "Import tts_engine module",
                'PASS'
            )
            
            status = get_tts_status()
            self.log_test(
                "get_tts_status()",
                'PASS',
                f"Backend: {status.get('backend', 'unknown')}, Enabled: {status.get('enabled', False)}"
            )
            
            if not status.get('available_voices'):
                self.log_test(
                    "Available Voices",
                    'FAIL',
                    "No voices found by engine"
                )
            else:
                self.log_test(
                    "Available Voices",
                    'PASS',
                    f"{len(status['available_voices'])} voice(s) registered"
                )
                
        except ImportError as e:
            self.log_test(
                "Import tts_engine module",
                'FAIL',
                str(e)
            )
        except Exception as e:
            self.log_test(
                "Integration Test",
                'WARN',
                str(e)[:200]
            )
            
    def run_all_tests(self):
        """Execute all diagnostic tests sequentially."""
        print("=" * 60)
        print("TTS DIAGNOSTIC SCRIPT FOR DIALOGUE ANONYMIZER")
        print(f"Started: {self.results['timestamp']}")
        print(f"Project Root: {PROJECT_ROOT}")
        print("=" * 60)
        
        # Run tests in order (some depend on previous passing)
        self.test_1_piper_binary_exists()
        self.test_2_voice_models_exist()
        self.test_3_library_dependencies()
        self.test_4_direct_synthesis_command()
        self.test_5_python_piper_package()
        self.test_6_coqui_xtts_availability()
        self.test_7_ld_library_path()
        self.test_8_integration_import()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for t in self.results['tests'] if t['status'] == 'PASS')
        warned = sum(1 for t in self.results['tests'] if t['status'] == 'WARN')
        failed = sum(1 for t in self.results['tests'] if t['status'] == 'FAIL')
        skipped = sum(1 for t in self.results['tests'] if t['status'] == 'SKIP')
        
        total = len(self.results['tests'])
        
        print(f"\nOverall Status: {self.results['overall_status']}")
        print(f"Results: {passed}/{total} passed, {failed} failed, {warned} warnings, {skipped} skipped")
        
        if self.results['recommendations']:
            print(f"\nRecommendations ({len(self.results['recommendations'])}):")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save results
        results_path = PROJECT_ROOT / "tts_diagnostic_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nFull results saved to: {results_path}")
        
        return self.results['overall_status'] == 'PASS'


def main():
    """Entry point."""
    diagnostic = TSSEngineDiagnostic()
    success = diagnostic.run_all_tests()
    
    if not success:
        print("\n⚠️  Some diagnostics failed. Review recommendations above.")
        print("\nNext Steps:")
        print("  1. If Piper binary missing: Re-run ./Installer.sh")
        print("  2. If libraries missing: sudo apt-get install libportaudio2 libsndfile1")
        print("  3. If voice models missing: Check network, re-run installer")
        print("  4. Consider Coqui XTTS fallback if Piper persists failing")
        sys.exit(1)
    else:
        print("\n✅ All diagnostics passed! TTS should be functional.")
        sys.exit(0)


if __name__ == '__main__':
    main()