#!/usr/bin/env bash
#===============================================================================
# ATA Secret Generation and Configuration Script
# Generates cryptographically strong secrets and updates Keycloak configuration
# MUST be run before first Docker deployment
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# CONFIGURATION
#-------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATA_PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SECRETS_DIR="${ATA_PROJECT_ROOT}/secrets"
REALM_FILE="${ATA_PROJECT_ROOT}/docker/keycloak/realm-export.json"
ENV_FILE="${ATA_PROJECT_ROOT}/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

#-------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#-------------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_step() {
    echo -e "\n${BLUE}▶ Step $1${NC}: $2"
}

#-------------------------------------------------------------------------------
# PRE-FLIGHT CHECKS
#-------------------------------------------------------------------------------

check_prerequisites() {
    print_step "1" "Checking prerequisites..."
    
    if ! command -v openssl &> /dev/null; then
        print_error "openssl is required but not installed!"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        print_warning "jq not found (will use sed for JSON manipulation)"
        USE_JQ=false
    else
        USE_JQ=true
        print_success "jq found"
    fi
    
    # Check we have write access
    if [[ ! -w "$ATA_PROJECT_ROOT" ]]; then
        print_error "Cannot write to ATA project root: $ATA_PROJECT_ROOT"
        print_info "Run this script from within the ATA directory"
        exit 1
    fi
    
    print_success "All prerequisites met"
}

#-------------------------------------------------------------------------------
# CREATE SECRETS DIRECTORY
#-------------------------------------------------------------------------------

create_secrets_directory() {
    print_step "2" "Creating secrets directory..."
    
    mkdir -p "$SECRETS_DIR"
    
    # Set restrictive permissions on directory
    chmod 700 "$SECRETS_DIR"
    
    print_success "Secrets directory created with permissions 700"
}

#-------------------------------------------------------------------------------
# GENERATE RANDOM VALUES
#-------------------------------------------------------------------------------

generate_random_base64() {
    openssl rand -base64 32 | tr -d '\n'
}

generate_random_hex() {
    openssl rand -hex 32
}

generate_jwt_secret() {
    # JWT secrets should be at least 256 bits
    openssl rand -hex 64
}

#-------------------------------------------------------------------------------
# GENERATE ALL SECRETS
#-------------------------------------------------------------------------------

generate_secrets() {
    print_step "3" "Generating cryptographic secrets..."
    
    local postgres_password=$(generate_random_base64)
    local keycloak_admin_password=$(generate_random_base64)
    local jwt_secret=$(generate_jwt_secret)
    local encryption_key=$(generate_random_base64)
    
    # Write to files
    echo -n "$postgres_password" > "${SECRETS_DIR}/postgres_password.txt"
    echo -n "$keycloak_admin_password" > "${SECRETS_DIR}/keycloak_admin_password.txt"
    echo -n "$jwt_secret" > "${SECRETS_DIR}/jwt_secret.txt"
    echo -n "$encryption_key" > "${SECRETS_DIR}/encryption_key.txt"
    
    # Set restrictive permissions (CRITICAL!)
    chmod 600 "${SECRETS_DIR}/postgres_password.txt"
    chmod 600 "${SECRETS_DIR}/keycloak_admin_password.txt"
    chmod 600 "${SECRETS_DIR}/jwt_secret.txt"
    chmod 600 "${SECRETS_DIR}/encryption_key.txt"
    
    # Store in variables for later use
    export GENERATED_POSTGRES_PASSWORD="$postgres_password"
    export GENERATED_KEYCLOAK_ADMIN_PASSWORD="$keycloak_admin_password"
    export GENERATED_JWT_SECRET="$jwt_secret"
    export GENERATED_ENCRYPTION_KEY="$encryption_key"
    
    print_success "Generated PostgreSQL password: ${postgres_password:0:8}***"
    print_success "Generated Keycloak admin password: ${keycloak_admin_password:0:8}***"
    print_success "Generated JWT secret: ${jwt_secret:0:16}***"
    print_success "Generated encryption key: ${encryption_key:0:8}***"
}

#-------------------------------------------------------------------------------
# UPDATE KEYCLOAK REALM CONFIGURATION
#-------------------------------------------------------------------------------

update_keycloak_realm() {
    print_step "4" "Updating Keycloak realm configuration..."
    
    if [[ ! -f "$REALM_FILE" ]]; then
        print_error "Keycloak realm file not found: $REALM_FILE"
        print_info "Create the file first or restore from template"
        exit 1
    fi
    
    # Update admin password in realm file
    if [[ "$USE_JQ" == "true" ]]; then
        # Using jq (more reliable)
        local temp_file=$(mktemp)
        
        jq --arg password "$GENERATED_KEYCLOAK_ADMIN_PASSWORD" \
           '(.users[] | select(.username == "admin") | .credentials[0].value) = $password' \
           "$REALM_FILE" > "$temp_file"
        
        mv "$temp_file" "$REALM_FILE"
    else
        # Using sed (fallback)
        local escaped_password=$(echo "$GENERATED_KEYCLOAK_ADMIN_PASSWORD" | sed 's/[&/\]/\\&/g')
        sed -i "s|CHANGE_ADMIN_PASSWORD_HERE|$escaped_password|g" "$REALM_FILE"
        sed -i "s|CHANGE_THIS_SECRET_IN_PRODUCTION|$escaped_password|g" "$REALM_FILE"
    fi
    
    # Update backend client secret
    local backend_client_secret=$(generate_random_base64)
    
    if [[ "$USE_JQ" == "true" ]]; then
        jq --arg secret "$backend_client_secret" \
           '(.clients[] | select(.clientId == "ata-backend-api") | .secret) = $secret' \
           "$REALM_FILE" > "${REALM_FILE}.tmp"
        mv "${REALM_FILE}.tmp" "$REALM_FILE"
    else
        local escaped_secret=$(echo "$backend_client_secret" | sed 's/[&/\]/\\&/g')
        sed -i "s|CHANGE_THIS_SECRET_IN_PRODUCTION|$escaped_secret|g" "$REALM_FILE"
    fi
    
    print_success "Updated admin password in realm configuration"
    print_success "Generated backend API client secret: ${backend_client_secret:0:16}***"
    
    # Export for docker-compose
    export GENERATED_BACKEND_CLIENT_SECRET="$backend_client_secret"
}

#-------------------------------------------------------------------------------
# UPDATE .ENV FILE
#-------------------------------------------------------------------------------

update_env_file() {
    print_step "5" "Updating .env file..."
    
    # Backup existing .env
    if [[ -f "$ENV_FILE" ]]; then
        cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        print_info "Created backup of .env: ${ENV_FILE}.backup.*"
    fi
    
    # Create/update .env with generated secrets
    cat >> "${ENV_FILE}" << EOF

# =============================================================================
# AUTO-GENERATED SECRETS (Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ"))
# These were generated by scripts/setup-secrets.sh
# =============================================================================

# DO NOT commit these values to git!
# They are in .gitignore and should remain local only.

JWT_SECRET_KEY=${GENERATED_JWT_SECRET}
POSTGRES_PASSWORD=${GENERATED_POSTGRES_PASSWORD}
KEYCLOAK_ADMIN_PASSWORD=${GENERATED_KEYCLOAK_ADMIN_PASSWORD}
ENCRYPTION_KEY=${GENERATED_ENCRYPTION_KEY}

# Backend API Client Secret (from Keycloak realm)
BACKEND_CLIENT_SECRET=${GENERATED_BACKEND_CLIENT_SECRET}
EOF

    print_success "Updated .env file with generated secrets"
    print_warning "Review .env file to ensure values are correct"
}

#-------------------------------------------------------------------------------
# CREATE .GITIGNORE ENTRY IF MISSING
#-------------------------------------------------------------------------------

ensure_gitignore() {
    print_step "6" "Ensuring secrets are excluded from git..."
    
    local gitignore_file="${ATA_PROJECT_ROOT}/.gitignore"
    local secrets_entry="secrets/"
    
    if [[ -f "$gitignore_file" ]]; then
        if grep -q "^secrets/" "$gitignore_file"; then
            print_success "secrets/ already in .gitignore"
        else
            echo "" >> "$gitignore_file"
            echo "# Secrets - NEVER COMMIT" >> "$gitignore_file"
            echo "secrets/" >> "$gitignore_file"
            echo "*.secret" >> "$gitignore_file"
            echo "*.password" >> "$gitignore_file"
            print_success "Added secrets/ to .gitignore"
        fi
    else
        print_warning ".gitignore not found at $gitignore_file"
        print_info "Manually add 'secrets/' to your .gitignore file"
    fi
}

#-------------------------------------------------------------------------------
# GENERATE SUMMARY REPORT
#-------------------------------------------------------------------------------

generate_summary_report() {
    print_header "Secret Generation Complete!"
    
    cat << EOF
============================================================================
SUMMARY
============================================================================

Generated Secrets:
  ✅ PostgreSQL password:     ${GENERATED_POSTGRES_PASSWORD:0:12}...
  ✅ Keycloak admin password: ${GENERATED_KEYCLOAK_ADMIN_PASSWORD:0:12}...
  ✅ JWT secret:              ${GENERATED_JWT_SECRET:0:20}...
  ✅ Encryption key:          ${GENERATED_ENCRYPTION_KEY:0:12}...
  ✅ Backend API client secret: ${GENERATED_BACKEND_CLIENT_SECRET:0:12}...

Files Modified:
  ✅ ${SECRETS_DIR}/postgres_password.txt
  ✅ ${SECRETS_DIR}/keycloak_admin_password.txt
  ✅ ${SECRETS_DIR}/jwt_secret.txt
  ✅ ${SECRETS_DIR}/encryption_key.txt
  ✅ ${REALM_FILE}
  ✅ ${ENV_FILE}

Permission Settings:
  ✅ secrets/ directory: 700 (owner read/write/execute only)
  ✅ Secret files: 600 (owner read/write only)

============================================================================
SECURITY NOTES
============================================================================

1. These secrets are stored in ./secrets/ directory
2. The directory is excluded from git via .gitignore
3. File permissions are set to 600 (read/write owner only)
4. DO NOT share these secrets via email, chat, or other channels
5. Store backup copies securely (e.g., encrypted password manager)
6. Rotate secrets periodically (recommended: every 90 days)

============================================================================
NEXT STEPS
============================================================================

1. Review .env file for correctness
   nano .env

2. Verify .gitignore excludes secrets
   cat .gitignore | grep secrets

3. Test Docker deployment
   docker-compose up -d --build

4. Verify services are healthy
   docker-compose ps
   curl -kf https://localhost/health

5. Save backup of secrets (encrypted):
   tar -czf secrets-backup.tar.gz secrets/
   gpg --symmetric --cipher-algo AES256 secrets-backup.tar.gz

============================================================================
EOF

    print_success "Setup complete!"
}

#-------------------------------------------------------------------------------
# ROLLBACK FUNCTION (For Testing)
#-------------------------------------------------------------------------------

rollback() {
    print_info "Rolling back changes..."
    
    if [[ -f "${ENV_FILE}.backup"* ]]; then
        local latest_backup=$(ls -t "${ENV_FILE}.backup."* | head -1)
        cp "$latest_backup" "$ENV_FILE"
        print_success "Restored .env from backup: $latest_backup"
    fi
    
    print_warning "Secret files in secrets/ directory were NOT reverted"
    print_warning "Run 'rm -rf secrets/* && ./scripts/setup-secrets.sh' to regenerate"
}

#-------------------------------------------------------------------------------
# DISPLAY HELP
#-------------------------------------------------------------------------------

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Generate cryptographic secrets for ATA Speech Anonymizer Docker deployment.

OPTIONS:
    -h, --help      Show this help message
    -r, --rollback  Rollback .env to backup (if available)
    -f, --force     Overwrite existing secrets without confirmation

EXAMPLES:
    # Generate new secrets (default)
    ./scripts/setup-secrets.sh

    # Rollback to previous .env
    ./scripts/setup-secrets.sh --rollback

    # Force overwrite existing secrets
    ./scripts/setup-secrets.sh --force

NOTES:
    • All secrets are generated using OpenSSL's secure random generator
    • Secret files have 600 permissions (read/write owner only)
    • secrets/ directory has 700 permissions
    • The secrets/ directory is excluded from git
    • Backup of .env is created automatically

EOF
}

#-------------------------------------------------------------------------------
# MAIN EXECUTION
#-------------------------------------------------------------------------------

main() {
    # Parse arguments
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--rollback)
            rollback
            exit 0
            ;;
        -f|--force)
            FORCE_OVERWRITE=true
            ;;
        "")
            # No arguments - proceed normally
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    
    # Pre-flight checks
    check_prerequisites
    create_secrets_directory
    
    # Check if secrets already exist
    if [[ -f "${SECRETS_DIR}/postgres_password.txt" ]] && [[ "$FORCE_OVERWRITE" != "true" ]]; then
        print_warning "Secrets already exist in $SECRETS_DIR"
        echo ""
        echo -e "${YELLOW}Options:${NC}"
        echo "  [Y] Regenerate all secrets (OVERWRITE EXISTING)"
        echo "  [N] Exit without changes"
        echo "  [F] Continue with existing secrets (skip generation)"
        read -p "Choose option (Y/N/F): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            FORCE_OVERWRITE=true
            print_info "Proceeding with regeneration..."
        elif [[ $REPLY =~ ^[Ff]$ ]]; then
            print_info "Using existing secrets - skipping generation"
            exit 0
        else
            print_info "Exiting without changes"
            exit 0
        fi
    fi
    
    # Generate secrets and update configurations
    generate_secrets
    update_keycloak_realm
    update_env_file
    ensure_gitignore
    
    # Summary
    generate_summary_report
}

# Run main function
main "$@"
