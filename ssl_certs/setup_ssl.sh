#!/bin/bash
#===============================================================================
# SSL Certificate Management for Speech Anonymizer
# Features:
#   - Auto-generates self-signed certs if none exist
#   - Detects and respects user-provided certificates
#   - Backs up self-generated certs (not user-provided)
#   - Self-contained in ATA repo
#   - Works for both development and Docker deployments
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# CONFIGURATION
#-------------------------------------------------------------------------------

CERT_DIR="/mnt/Data_Mount/VERANDA_DataMount/ATA/ssl_certs"
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${CERT_DIR}/server.key"
MARKER_FILE="${CERT_DIR}/.generated_by_ata"
BACKUP_DIR="${CERT_DIR}/.backups"

VALID_DAYS=365
KEY_SIZE=2048
CN="transcriber.cloud.cci.charite.de"
ORG="VERANDA Project"
COUNTRY="DE"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

#-------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#-------------------------------------------------------------------------------

print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

#-------------------------------------------------------------------------------
# CERTIFICATE DETECTION FUNCTIONS
#-------------------------------------------------------------------------------

check_cert_exists() {
    if [[ -f "$CERT_FILE" && -f "$KEY_FILE" ]]; then
        return 0  # Certs exist
    fi
    return 1  # No certs
}

is_self_generated() {
    if [[ -f "$MARKER_FILE" ]]; then
        return 0  # Self-generated
    fi
    return 1  # User-provided or unknown
}

check_cert_expiry() {
    if [[ -f "$CERT_FILE" ]]; then
        local expiry_date=$(openssl x509 -enddate -noout -in "$CERT_FILE" 2>/dev/null | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry_date" +%s 2>/dev/null)
        local now_epoch=$(date +%s)
        local days_left=$(( (expiry_epoch - now_epoch) / 86400 ))
        
        echo "$days_left"
    else
        echo "-1"
    fi
}

#-------------------------------------------------------------------------------
# CERTIFICATE CREATION
#-------------------------------------------------------------------------------

create_self_signed_cert() {
    print_info "Creating self-signed SSL certificate..."
    
    # Ensure directory exists
    mkdir -p "$CERT_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Generate private key
    openssl genrsa -out "$KEY_FILE" $KEY_SIZE 2>/dev/null
    
    # Generate certificate
    openssl req -new -x509 -key "$KEY_FILE" -out "$CERT_FILE" \
        -days $VALID_DAYS \
        -subj "/C=${COUNTRY}/ST=Berlin/L=Berlin/O=${ORG}/CN=${CN}" \
        2>/dev/null
    
    # Set secure permissions
    chmod 600 "$KEY_FILE"
    chmod 644 "$CERT_FILE"
    
    # Mark as self-generated
    touch "$MARKER_FILE"
    echo "Self-generated on $(date)" > "$MARKER_FILE"
    
    print_success "Certificate created successfully!"
    echo "  • Certificate: $CERT_FILE"
    echo "  • Private Key: $KEY_FILE"
    echo "  • Valid for: $VALID_DAYS days"
    echo "  • CN: $CN"
}

backup_self_generated_cert() {
    if is_self_generated; then
        local backup_ts=$(date +%Y%m%d_%H%M%S)
        cp "$CERT_FILE" "${BACKUP_DIR}/server_${backup_ts}.crt"
        cp "$KEY_FILE" "${BACKUP_DIR}/server_${backup_ts}.key"
        print_info "Backed up self-generated certificate: ${backup_ts}"
    else
        print_warning "Skipping backup - certificate is user-provided"
    fi
}

#-------------------------------------------------------------------------------
# MAIN LOGIC
#-------------------------------------------------------------------------------


validate_certificate() {
    if [[ ! -f "$CERT_FILE" || ! -f "$KEY_FILE" ]]; then
        return 1
    fi
    
    # Check key matches cert using proper OpenSSL comparison
    local cert_modulus=$(openssl x509 -noout -modulus -in "$CERT_FILE" 2>/dev/null | openssl md5 | awk '{print $NF}')
    local key_modulus=$(openssl rsa -noout -modulus -in "$KEY_FILE" 2>/dev/null | openssl md5 | awk '{print $NF}')
    
    if [[ "$cert_modulus" != "$key_modulus" ]]; then
        print_error "Certificate and key do not match!"
        print_info "Cert MD5: $cert_modulus"
        print_info "Key MD5:  $key_modulus"
        return 1
    fi
    
    # Check expiry
    local days_left=$(check_cert_expiry)
    if [[ "$days_left" -lt 30 ]]; then
        print_warning "Certificate expires in $days_left days!"
    fi
    
    return 0
}

main() {
    echo ""
    echo "================================================"
    echo "  SSL Certificate Manager for Speech Anonymizer"
    echo "================================================"
    echo ""
    
    # Ensure cert directory exists
    mkdir -p "$CERT_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Check current state
    print_info "Certificate directory: $CERT_DIR"
    
    if check_cert_exists; then
        print_success "Certificates already exist"
        
        # Validate them
        if validate_certificate; then
            local days_left=$(check_cert_expiry)
            print_info "Certificate expires in $days_left days"
            
            if [[ "$days_left" -lt 30 ]]; then
                print_warning "Certificate expiring soon - consider regeneration"
                read -p "Regenerate now? [y/N] " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    backup_self_generated_cert
                    rm "$CERT_FILE" "$KEY_FILE" "$MARKER_FILE"
                    create_self_signed_cert
                fi
            else
                print_info "Certificate is valid"
            fi
        else
            print_error "Certificate validation failed!"
            exit 1
        fi
    else
        print_warning "No certificates found"
        print_info "Creating self-signed certificate..."
        create_self_signed_cert
    fi
    
    # Verify final state
    echo ""
    print_info "Certificate Status:"
    if check_cert_exists && validate_certificate; then
        print_success "✅ Certificates ready for use"
        
        echo ""
        echo "================================================"
        echo "  Usage Examples"
        echo "================================================"
        echo ""
        echo "  Nginx config:"
        echo "    ssl_certificate ${CERT_FILE};"
        echo "    ssl_certificate_key ${KEY_FILE};"
        echo ""
        echo "  Flask/Python:"
        echo "    ssl_context = ('${CERT_FILE}', '${KEY_FILE}')"
        echo ""
        echo "  Docker:"
        echo "    volumes:"
        echo "      - ./ssl_certs:/app/ssl_certs:ro"
        echo ""
        echo "================================================"
    else
        print_error "❌ Certificate setup failed!"
        exit 1
    fi
}

# Handle subcommands
case "${1:-setup}" in
    setup|check)
        main
        ;;
    regenerate)
        backup_self_generated_cert
        rm -f "$CERT_FILE" "$KEY_FILE" "$MARKER_FILE"
        create_self_signed_cert
        print_success "Certificate regenerated"
        ;;
    status)
        if check_cert_exists; then
            print_info "Certificate status:"
            openssl x509 -noout -dates -subject -in "$CERT_FILE" 2>/dev/null
            echo ""
            echo "Days until expiry: $(check_cert_expiry)"
        else
            print_warning "No certificate found"
        fi
        ;;
    *)
        echo "Usage: $0 {setup|regenerate|status}"
        exit 1
        ;;
esac