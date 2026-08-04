#!/usr/bin/env bash
# ============================================================================
# TTS Infrastructure Verification Script
# Run from inside the ATA folder
# ============================================================================

echo "=============================================="
echo "TTS INFRASTRUCTURE VERIFICATION"
echo "=============================================="
echo "Current Directory: $(pwd)"
echo ""

# Path configuration (matches your actual structure)
TTS_DIR="./pipeline/tts"
VOICE_DIR="${TTS_DIR}/voices"
BIN_DIR="${TTS_DIR}/bin"
PIPERS_BIN="${BIN_DIR}/piper"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counter
PASSED=0
FAILED=0

check_file() {
    local file=$1
    local desc=$2
    
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo -e "${GREEN}✓${NC} ${desc}: ${file} (${size})"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} ${desc}: ${file} NOT FOUND"
        ((FAILED++))
        return 1
    fi
}

check_dir() {
    local dir=$1
    local desc=$2
    
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        count=$(ls -1 "$dir" | wc -l)
        echo -e "${GREEN}✓${NC} ${desc}: ${dir} (${count} items)"
        ((PASSED++))
        return 0
    elif [ -d "$dir" ]; then
        echo -e "${YELLOW}⚠${NC} ${desc}: ${dir} EXISTS but EMPTY"
        ((FAILED++))
        return 1
    else
        echo -e "${RED}✗${NC} ${desc}: ${dir} NOT FOUND"
        ((FAILED++))
        return 1
    fi
}

echo "[1] DIRECTORY STRUCTURE"
echo "----------------------"
check_dir "$TTS_DIR" "TTS Directory"
check_dir "$BIN_DIR" "Binary Directory"
check_dir "$VOICE_DIR" "Voice Directory"
echo ""

echo "[2] PIPER BINARY"
echo "----------------"
if [ -f "$PIPERS_BIN" ]; then
    check_file "$PIPERS_BIN" "Piper Executable"
    
    # Check file type
    file_type=$(file "$PIPERS_BIN")
    echo "   Type: ${file_type}"
    
    # Check permissions
    if [ -x "$PIPERS_BIN" ]; then
        echo -e "   ${GREEN}✓${NC} Execute permission: YES"
        ((PASSED++))
    else
        echo -e "   ${RED}✗${NC} Execute permission: NO"
        chmod +x "$PIPERS_BIN"
        echo -e "   ${GREEN}✓${NC} Fixed execute permission"
        ((PASSED++))
    fi
    
    # Check if it runs (may fail due to missing libs)
    echo ""
    echo "   Testing binary execution..."
    if "$PIPERS_BIN" --help &>/dev/null; then
        echo -e "   ${GREEN}✓${NC} Binary responds to --help"
        ((PASSED++))
    else
        echo -e "   ${YELLOW}⚠${NC} Binary exists but --help failed (likely missing shared libraries)"
        echo -e "   ${YELLOW}   Fix:${NC} sudo apt-get install libportaudio2 libsndfile1"
    fi
else
    check_file "$PIPERS_BIN" "Piper Executable"
fi
echo ""

echo "[3] VOICE MODELS"
echo "----------------"
VOICE_COUNT=$(find "$VOICE_DIR" -name "*.onnx" -type f 2>/dev/null | wc -l)
VALID_VOICES=0

if [ "$VOICE_COUNT" -gt 0 ]; then
    echo "Found ${VOICE_COUNT} .onnx file(s):"
    
    for voice_file in "$VOICE_DIR"/*.onnx; do
        if [ -f "$voice_file" ]; then
            size_bytes=$(stat -c%s "$voice_file" 2>/dev/null || echo 0)
            size_mb=$((size_bytes / 1024 / 1024))
            basename=$(basename "$voice_file")
            
            if [ "$size_mb" -ge 40 ]; then
                echo -e "   ${GREEN}✓${NC} ${basename} (${size_mb} MB) - VALID"
                ((VALID_VOICES++))
            else
                echo -e "   ${RED}✗${NC} ${basename} (${size_mb} MB) - TOO SMALL (corrupted)"
            fi
        fi
    done
    
    if [ "$VALID_VOICES" -ge 2 ]; then
        echo -e "\n${GREEN}✓${NC} Voice models: ${VALID_VOICES} valid model(s) ready"
        ((PASSED++))
    else
        echo -e "\n${RED}✗${NC} Voice models: Need minimum 2 valid voices for multi-speaker"
        ((FAILED++))
    fi
else
    echo -e "${RED}✗${NC} No voice models found in ${VOICE_DIR}"
    echo ""
    echo "Voice models may not have been downloaded during installation."
    echo "Run this quick-download script:"
    echo ""
    echo "  bash << 'DOWNLOAD_EOF'"
    echo "  cd pipeline/tts/voices"
    echo "  curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
    echo "  curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    echo "  DOWNLOAD_EOF"
    ((FAILED++))
fi
echo ""

echo "[4] VOICE MAPPING FILE"
echo "----------------------"
if [ -f "${VOICE_DIR}/VOICE_MAPPING.txt" ]; then
    check_file "${VOICE_DIR}/VOICE_MAPPING.txt" "Voice Mapping"
    echo "   Contents preview:"
    head -6 "${VOICE_DIR}/VOICE_MAPPING.txt" | sed 's/^/   /'
else
    check_file "${VOICE_DIR}/VOICE_MAPPING.txt" "Voice Mapping"
fi
echo ""

echo "[5] TTS_ENGINE MODULE"
echo "---------------------"
if [ -f "${TTS_DIR}/tts_engine.py" ]; then
    check_file "${TTS_DIR}/tts_engine.py" "TTS Engine Python Module"
    
    # Check if it can be imported
    echo ""
    echo "   Testing Python import..."
    if python3 -c "import sys; sys.path.insert(0, '${TTS_DIR}'); from tts_engine import SPEAKER_VOICE_MAP" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} Python import successful"
        ((PASSED++))
        
        # Show speaker mapping
        echo ""
        echo "   Speaker-to-Voice Mapping:"
        python3 -c "
import sys
sys.path.insert(0, '${TTS_DIR}')
from tts_engine import SPEAKER_VOICE_MAP
for k, v in SPEAKER_VOICE_MAP.items():
    print(f'   ${GREEN}●${NC} ${k} → ${v}')
" 2>/dev/null || echo "   (Could not display mapping)"
    else
        echo -e "   ${RED}✗${NC} Python import failed"
        ((FAILED++))
    fi
else
    check_file "${TTS_DIR}/tts_engine.py" "TTS Engine Python Module"
fi
echo ""

echo "[6] LD_LIBRARY_PATH"
echo "-------------------"
current_ld_path=$LD_LIBRARY_PATH
echo "   Current: ${current_ld_path:0:60}..."

if [[ "$current_ld_path" == *"${BIN_DIR}"* ]]; then
    echo -e "   ${GREEN}✓${NC} TTS bin directory in LD_LIBRARY_PATH"
    ((PASSED++))
else
    echo -e "   ${YELLOW}⚠${NC} TTS bin directory NOT in LD_LIBRARY_PATH"
    echo "   ${YELLOW}   To fix:${NC}"
    echo "   export LD_LIBRARY_PATH=${PWD}/${BIN_DIR}:\$LD_LIBRARY_PATH"
    ((PASSED++))  # Not fatal, can be fixed at runtime
fi
echo ""

echo "[7] PYTHON PIPER PACKAGE"
echo "------------------------"
if python3 -c "from piper import PiperVoice" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} piper Python package installed"
    pip3 show piper-tts 2>/dev/null | grep -E "^(Name|Version):" | sed 's/^/   /'
    ((PASSED++))
else
    echo -e "   ${YELLOW}⚠${NC} piper Python package NOT installed"
    echo "   ${YELLOW}   Install:${NC} pip install piper-tts"
fi
echo ""

echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ All critical checks passed! TTS infrastructure is healthy.${NC}"
else
    echo -e "${RED}❌ Some checks failed.${NC}"
    echo ""
    if [ "$VALID_VOICES" -lt 2 ]; then
        echo "🔧 RECOMMENDED FIX: Download voice models"
        echo "   bash << 'DOWNLOAD_EOF'"
        echo "   mkdir -p pipeline/tts/voices"
        echo "   cd pipeline/tts/voices"
        echo ""
        echo "   # Download 4 voices for speaker differentiation"
        echo "   for voice in en_US-amy-medium en_US-lessac-medium en_US-kusal-medium en_US-ryan-medium; do"
        echo "       name=\${voice##*_}"
        echo "       parts=(\$voice)"
        echo "       group=\${parts[1]}\${parts[2]:0:3}"
        echo "       quality=\${parts[2]}"
        echo "       url=\"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/\${group}/\${quality}/\${voice}.onnx\""
        echo "       echo \"Downloading \$voice...\""
        echo "       curl -LO \"\$url\""
        echo "   done"
        echo ""
        echo "   cd ../../../"
        echo "   DOWNLOAD_EOF"
    fi
    
    if ! [[ "$current_ld_path" == *"${BIN_DIR}"* ]]; then
        echo ""
        echo "🔧 RECOMMENDED FIX: Update LD_LIBRARY_PATH"
        echo "   echo 'export LD_LIBRARY_PATH=${PWD}/pipeline/tts/bin:\$LD_LIBRARY_PATH' >> ~/.bashrc"
        echo "   source ~/.bashrc"
    fi
fi

echo ""
exit $FAILED