#!/usr/bin/env bash
#===============================================================================
# ATA Deployment Validation Script
# Verifies all required files exist with correct permissions before Docker deployment
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# CONFIGURATION
#-------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATA_PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Track results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

#-------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#-------------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((TOTAL_CHECKS++))
    ((PASSED_CHECKS++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((TOTAL_CHECKS++))
    ((FAILED_CHECKS++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    ((WARNINGS++))
}

info() {
    echo -e "${CYAN}ℹ️  INFO${NC}: $1"
}

check_file_exists() {
    local file="$1"
    local description="$2"
    
    if [[ -f "$file" ]]; then
        pass "$description"
        return 0
    else
        fail "$description (missing: $file)"
        return 1
    fi
}

check_dir_exists() {
    local dir="$1"
    local description="$2"
    
    if [[ -d "$dir" ]]; then
        pass "$description"
        return 0
    else
        fail "$description (missing: $dir)"
        return 1
    fi
}

check_permissions() {
    local file="$1"
    local expected="$2"
    local description="$3"
    
    if [[ ! -f "$file" ]]; then
        fail "$description (file not found: $file)"
        return 1
    fi
    
    local actual=$(stat -c "%a" "$file" 2>/dev/null || stat -f "%Lp" "$file" 2>/dev/null)
    
    if [[ "$actual" == "$expected" ]]; then
        pass "$description (permissions: $actual)"
        return 0
    else
        fail "$description (permissions: expected $expected, got $actual)"
        return 1
    fi
}

check_content_pattern() {
    local file="$1"
    local pattern="$2"
    local description="$3"
    
    if [[ ! -f "$file" ]]; then
        fail "$description (file not found)"
        return 1
    fi
    
    if grep -q "$pattern" "$file" 2>/dev/null; then
        pass "$description"
        return 0
    else
        fail "$description (pattern not found: $pattern)"
        return 1
    fi
}

#-------------------------------------------------------------------------------
# SECTION: PROJECT STRUCTURE
#-------------------------------------------------------------------------------

check_project_structure() {
    print_header "1. Project Structure"
    
    check_dir_exists "$ATA_PROJECT_ROOT" "Project root directory"
    check_file_exists "${ATA_PROJECT_ROOT}/docker-compose.yml" "Docker Compose (development)"
    check_file_exists "${ATA_PROJECT_ROOT}/docker-compose.prod.yml" "Docker Compose (production)"
    check_file_exists "${ATA_PROJECT_ROOT}/.gitignore" ".gitignore file"
    check_file_exists "${ATA_PROJECT_ROOT}/.dockerignore" ".dockerignore file"
}

#-------------------------------------------------------------------------------
# SECTION: SECRETS (CRITICAL)
#-------------------------------------------------------------------------------

check_secrets() {
    print_header "2. Secrets Configuration (CRITICAL)"
    
    local secrets_dir="${ATA_PROJECT_ROOT}/secrets"
    
    # Check directory exists
    if [[ ! -d "$secrets_dir" ]]; then
        fail "Secrets directory missing: $secrets_dir"
        return 1
    fi
    
    pass "Secrets directory exists"
    
    # Check permissions on directory
    local dir_perms=$(stat -c "%a" "$secrets_dir" 2>/dev/null || stat -f "%Lp" "$secrets_dir" 2>/dev/null)
    if [[ "$dir_perms" == "700" ]]; then
        pass "Secrets directory permissions: 700"
    else
        fail "Secrets directory permissions: expected 700, got $dir_perms"
    fi
    
    # Check each secret file
    check_permissions "${secrets_dir}/postgres_password.txt" "600" "PostgreSQL password file"
    check_permissions "${secrets_dir}/keycloak_admin_password.txt" "600" "Keycloak admin password file"
    check_permissions "${secrets_dir}/jwt_secret.txt" "600" "JWT secret file"
    check_permissions "${secrets_dir}/encryption_key.txt" "600" "Encryption key file"
    
    # Check files have content (not empty)
    for file in postgres_password.txt jwt_secret.txt encryption_key.txt; do
        if [[ -s "${secrets_dir}/${file}" ]]; then
            pass "${file} has content (not empty)"
        else
            fail "${file} is empty"
        fi
    done
    
    # Verify .gitignore excludes secrets
    if grep -q "^secrets/" "${ATA_PROJECT_ROOT}/.gitignore" 2>/dev/null; then
        pass ".gitignore excludes secrets/"
    else
        fail ".gitignore does NOT exclude secrets/ (CRITICAL!)"
    fi
}

#-------------------------------------------------------------------------------
# SECTION: DOCKER CONFIGURATION
#-------------------------------------------------------------------------------

check_docker_config() {
    print_header "3. Docker Configuration"
    
    local docker_dir="${ATA_PROJECT_ROOT}/docker"
    
    check_dir_exists "${docker_dir}/backend" "Docker backend directory"
    check_dir_exists "${docker_dir}/frontend" "Docker frontend directory"
    check_dir_exists "${docker_dir}/nginx" "Docker nginx directory"
    check_dir_exists "${docker_dir}/postgres" "Docker postgres directory"
    check_dir_exists "${docker_dir}/keycloak" "Docker keycloak directory"
    
    # Backend files
    check_file_exists "${docker_dir}/backend/Dockerfile" "Backend Dockerfile"
    check_file_exists "${docker_dir}/backend/requirements.txt" "Backend requirements.txt"
    check_file_exists "${docker_dir}/backend/.dockerignore" "Backend .dockerignore"
    
    # Frontend files
    check_file_exists "${docker_dir}/frontend/Dockerfile" "Frontend Dockerfile"
    check_file_exists "${docker_dir}/frontend/package.json" "Frontend package.json"
    check_file_exists "${docker_dir}/frontend/.dockerignore" "Frontend .dockerignore"
    
    # Nginx files
    check_file_exists "${docker_dir}/nginx/nginx.conf" "Nginx main config"
    check_file_exists "${docker_dir}/nginx/security.headers" "Nginx security headers"
    check_file_exists "${docker_dir}/nginx/sites-available/speech-anonymizer" "Nginx site config"
    check_file_exists "${docker_dir}/nginx/seccomp-profile.json" "Nginx seccomp profile"
    
    # Postgres files
    check_file_exists "${docker_dir}/postgres/init-db.sql" "PostgreSQL init script"
    
    # Keycloak files
    check_file_exists "${docker_dir}/keycloak/realm-export.json" "Keycloak realm configuration"
}

#-------------------------------------------------------------------------------
# SECTION: SSL CERTIFICATES
#-------------------------------------------------------------------------------

check_ssl_certificates() {
    print_header "4. SSL Certificates"
    
    local ssl_dir="${ATA_PROJECT_ROOT}/ssl_certs"
    
    if [[ ! -d "$ssl_dir" ]]; then
        warn "SSL certificates directory not found: $ssl_dir"
        warn "Run ./ssl_certs/setup_ssl.sh to generate certificates"
        return 1
    fi
    
    pass "SSL certificates directory exists"
    
    check_file_exists "${ssl_dir}/setup_ssl.sh" "SSL setup script"
    
    # Check if production certs exist (may not exist in dev)
    if [[ -f "${ssl_dir}/server.crt" ]] && [[ -f "${ssl_dir}/server.key" ]]; then
        pass "Organization SSL certificates present"
    else
        warn "Organization SSL certificates not found (will use self-signed in dev)"
    fi
    
    # Verify .GitIgnore in ssl_certs
    check_file_exists "${ssl_dir}/.GitIgnore" "SSL .GitIgnore file"
    
    # Verify .GitIgnore contains certificate patterns
    if grep -q "\.key" "${ssl_dir}/.GitIgnore" 2>/dev/null; then
        pass "SSL .GitIgnore excludes key files"
    else
        warn "SSL .GitIgnore may not exclude key files"
    fi
}

#-------------------------------------------------------------------------------
# SECTION: CONFIG FILE CONTENTS
#-------------------------------------------------------------------------------

check_config_contents() {
    print_header "5. Configuration Contents"
    
    # Check nginx.conf includes conf.d
    if check_content_pattern "${ATA_PROJECT_ROOT}/docker/nginx/nginx.conf" "include /etc/nginx/conf.d" \
       "Nginx config includes conf.d directory"; then
        :
    fi
    
    # Check speech-anonymizer uses HTTPS backend
    if check_content_pattern "${ATA_PROJECT_ROOT}/docker/nginx/sites-available/speech-anonymizer" \
       "proxy_pass https://" "Nginx site config uses HTTPS backend"; then
        :
    fi
    
    # Check speech-anonymizer uses correct SSL path
    if check_content_pattern "${ATA_PROJECT_ROOT}/docker/nginx/sites-available/speech-anonymizer" \
       "/etc/nginx/ssl/server.crt" "Nginx site config uses correct SSL path"; then
        :
    fi
    
    # Check backend Dockerfile has HEALTHCHECK
    if check_content_pattern "${ATA_PROJECT_ROOT}/docker/backend/Dockerfile" "HEALTHCHECK" \
       "Backend Dockerfile includes HEALTHCHECK"; then
        :
    fi
    
    # Check docker-compose mounts SSL certs
    if check_content_pattern "${ATA_PROJECT_ROOT}/docker-compose.yml" "/etc/nginx/ssl" \
       "Docker Compose mounts SSL certificates"; then
        :
    fi
}

#-------------------------------------------------------------------------------
# SECTION: ENVIRONMENT FILE
#-------------------------------------------------------------------------------

check_env_file() {
    print_header "6. Environment File"
    
    local env_file="${ATA_PROJECT_ROOT}/.env"
    
    if [[ ! -f "$env_file" ]]; then
        warn ".env file not found (will use defaults from docker-compose)"
        warn "Create .env from .env.example if needed"
        return 1
    fi
    
    pass ".env file exists"
    
    # Check .env doesn't contain placeholder text (indicates it wasn't configured)
    if grep -q "changeme\|your_api_key_here\|replace-with-secure" "$env_file" 2>/dev/null; then
        warn ".env contains placeholder values - review and update"
    else
        pass ".env appears to have actual values configured"
    fi
    
    # Check gitignore excludes .env
    if grep -q "^\.env" "${ATA_PROJECT_ROOT}/.gitignore" 2>/dev/null; then
        pass ".gitignore excludes .env"
    else
        warn ".gitignore may not exclude .env file"
    fi
}

#-------------------------------------------------------------------------------
# SECTION: GITIGNORE VALIDATION
#-------------------------------------------------------------------------------

check_gitignore() {
    print_header "7. Git Ignore Validation"
    
    local gitignore="${ATA_PROJECT_ROOT}/.gitignore"
    
    # List critical patterns that should be ignored
    local critical_patterns=(
        "secrets/"
        ".env"
        "ssl_certs/*.key"
        "ssl_certs/server.key"
        "pipeline/model/"
        "uploads/"
        "logs/"
    )
    
    for pattern in "${critical_patterns[@]}"; do
        if grep -q "$pattern" "$gitignore" 2>/dev/null; then
            pass ".gitignore includes: $pattern"
        else
            warn ".gitignore may not exclude: $pattern"
        fi
    done
}

#-------------------------------------------------------------------------------
# SUMMARY AND RESULTS
#-------------------------------------------------------------------------------

print_summary() {
    print_header "Validation Summary"
    
    echo ""
    echo -e "${BLUE}Total Checks:${NC}   $TOTAL_CHECKS"
    echo -e "${GREEN}Passed:${NC}        $PASSED_CHECKS"
    echo -e "${RED}Failed:${NC}         $FAILED_CHECKS"
    echo -e "${YELLOW}Warnings:${NC}      $WARNINGS"
    echo ""
    
    if [[ $FAILED_CHECKS -eq 0 ]]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✅ ALL CRITICAL CHECKS PASSED!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "Your environment is ready for Docker deployment!"
        echo ""
        echo "Next steps:"
        echo "  1. Review warnings above (if any)"
        echo "  2. Run: docker-compose up -d --build"
        echo "  3. Check health: curl -kf https://localhost/health"
        echo ""
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}❌ $FAILED_CHECKS CHECK(S) FAILED!${NC}"
        echo -e "${RED}========================================${NC}"
        echo ""
        echo "Fix failed checks before deploying:"
        echo "  • Check secrets/ directory and permissions"
        echo "  • Verify docker/ directory structure"
        echo "  • Ensure .gitignore excludes sensitive files"
        echo ""
        exit 1
    fi
    
    if [[ $WARNINGS -gt 0 ]]; then
        echo -e "${YELLOW}⚠️  $WARNINGS WARNING(S) detected.${NC}"
        echo "These are not blocking but should be reviewed."
        echo ""
    fi
}

#-------------------------------------------------------------------------------
# MAIN EXECUTION
#-------------------------------------------------------------------------------

main() {
    print_header "ATA Deployment Validator"
    echo ""
    echo -e "${CYAN}Project Root:${NC} $ATA_PROJECT_ROOT"
    echo -e "${CYAN}Timestamp:${NC}     $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo ""
    
    # Run all checks
    check_project_structure
    check_secrets
    check_docker_config
    check_ssl_certificates
    check_config_contents
    check_env_file
    check_gitignore
    
    # Print summary
    print_summary
}

# Run main function
main "$@"
