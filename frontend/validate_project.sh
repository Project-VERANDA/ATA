#!/bin/bash
# validate_project.sh - Check React+TypeScript project structure and content

echo "=========================================="
echo "  BIH Speech Anonymizer - Validation"
echo "=========================================="
echo ""

PASS=0
FAIL=0

check_file() {
  local file=$1
  local desc=$2
  if [[ -f "$file" ]]; then
    echo "✓ $desc: $file"
    ((PASS++))
    return 0
  else
    echo "✗ MISSING: $desc: $file"
    ((FAIL++))
    return 1
  fi
}

check_no_javascript() {
  local file=$1
  if [[ "$file" == *.jsx ]] || [[ "$file" == *.js ]]; then
    echo "✗ WRONG EXTENSION: Should be .tsx: $file"
    ((FAIL++))
    return 1
  fi
  return 0
}

echo "[1] Checking Project Root Files"
echo "--------------------------------"
check_file "index.html" "Main HTML"
check_file "package.json" "Package Config"
check_file "vite.config.ts" "Vite Config"
check_file "tsconfig.json" "TypeScript Config"
echo ""

echo "[2] Checking src/ Structure"
echo "----------------------------"
check_file "src/main.tsx" "Entry Point (TSX)"
check_file "src/App.tsx" "App Component (TSX)"
check_file "src/index.css" "Global Styles (CSS)"
check_file "src/types.ts" "Type Definitions"
echo ""

echo "[3] Checking Components Directory"
echo "----------------------------------"
for comp in Layout BIHLogo SurveyPopup InfoPopup; do
  check_file "src/components/${comp}.tsx" "${comp} Component"
done
echo ""

echo "[4] Checking Pages Directory"
echo "-----------------------------"
for page in LandingPage SignInPage SubmitJobPage DashboardPage JobDetailPage; do
  check_file "src/pages/${page}.tsx" "${page} Page"
done
echo ""

echo "[5] Checking Contexts Directory"
echo "--------------------------------"
check_file "src/contexts/AuthContext.tsx" "Auth Context"
check_file "src/contexts/ThemeContext.tsx" "Theme Context"
echo ""

echo "[6] Detecting Wrong File Extensions"
echo "------------------------------------"
BAD_EXT=$(find src -name "*.jsx" -o -name "*.js" 2>/dev/null)
if [[ -z "$BAD_EXT" ]]; then
  echo "✓ No .jsx/.js files found (correct for TSX project)"
  ((PASS+=5))
else
  echo "✗ Found .jsx/.js files that should be .tsx:"
  echo "$BAD_EXT" | while read f; do
    echo "  → $f"
  done
  ((FAIL+=$(echo "$BAD_EXT" | wc -l)))
fi
echo ""

echo "[7] Checking index.html Script Reference"
echo "-----------------------------------------"
SCRIPT_SRC=$(grep -o 'src="/src/[^"]*"' index.html 2>/dev/null | head -1)
if [[ "$SCRIPT_SRC" == *".tsx\"" ]]; then
  echo "✓ Script reference correct: main.tsx"
  ((PASS++))
elif [[ "$SCRIPT_SRC" == *".jsx\"" ]]; then
  echo "✗ Script reference WRONG: References .jsx but file is .tsx"
  ((FAIL++))
else
  echo "⚠ Could not parse script src from index.html"
fi
echo ""

echo "[8] Checking for Common Import Issues"
echo "--------------------------------------"
# Look for .tsx extensions in imports (should be omitted)
EXT_IMPORTS=$(grep -r "from.*\.tsx['\"]" src 2>/dev/null | wc -l)
if [[ $EXT_IMPORTS -gt 0 ]]; then
  echo "⚠ Found $EXT_IMPORTS imports with explicit .tsx extensions (can be omitted)"
  grep -rn "from.*\.tsx['\"]" src 2>/dev/null | head -3
else
  echo "✓ Imports use correct extensionless paths"
  ((PASS++))
fi
echo ""

echo "[9] Checking Key Content in Critical Files"
echo "-------------------------------------------"

# Check index.css has CSS variables
if grep -q ":root" src/index.css 2>/dev/null; then
  echo "✓ index.css has CSS root variables"
  ((PASS++))
else
  echo "✗ index.css missing :root CSS variables"
  ((FAIL++))
fi

# Check App.tsx has Routes
if grep -q "Routes" src/App.tsx 2>/dev/null; then
  echo "✓ App.tsx has routing set up"
  ((PASS++))
else
  echo "✗ App.tsx missing Routes configuration"
  ((FAIL++))
fi

# Check AuthContext exports
if grep -q "export.*AuthProvider" src/contexts/AuthContext.tsx 2>/dev/null; then
  echo "✓ AuthContext exports Provider"
  ((PASS++))
else
  echo "✗ AuthContext missing Provider export"
  ((FAIL++))
fi

# Check ThemeContext exports
if grep -q "export.*ThemeProvider" src/contexts/ThemeContext.tsx 2>/dev/null; then
  echo "✓ ThemeContext exports Provider"
  ((PASS++))
else
  echo "✗ ThemeContext missing Provider export"
  ((FAIL++))
fi
echo ""

echo "[10] Running TypeScript Compiler (Dry Run)"
echo "------------------------------------------"
if command -v npx &> /dev/null; then
  echo "Checking TypeScript syntax..."
  if command -v npx if npx tsc --noEmit 2>&1 | head -20; then> /dev/null && npx tsc --noEmit > /dev/null 2>&1; then
    echo "✓ No critical TypeScript errors (first 20 lines shown above if any)"
    ((PASS++))
  fi
fi
echo ""

echo "=========================================="
echo "  VALIDATION SUMMARY"
echo "=========================================="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""

if [[ $FAIL -eq 0 ]]; then
  echo "✅ All checks passed! Ready to run 'npm run dev'"
else
  echo "⚠ $FAIL issue(s) found. Review above and fix before running."
fi
echo ""

# Fix suggestions
if [[ -f "index.html" ]] && grep -q "\.jsx\"" index.html; then
  echo "QUICK FIX: Update index.html line with script src:"
  echo "  Replace: src=\"/src/main.jsx\""
  echo "  With:    src=\"/src/main.tsx\""
  echo ""
fi

if find src -name "*.jsx" 2>/dev/null | grep -q .; then
  echo "QUICK FIX: Rename .jsx files to .tsx:"
  find src -name "*.jsx" -exec sh -c 'mv "$1" "${1%.jsx}.tsx"' _ {} \;
  echo "Done renaming."
  echo ""
fi
