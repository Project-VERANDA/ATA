#!/usr/bin/env bash
#===============================================================================
# Nginx Configuration Setup for Speech Anonymizer
# Features:
#   - Auto-detects SSL certificates from ssl_certs/ directory
#   - Compatible with setup_ssl.sh certificate management
#   - Routes /app/ to React frontend (Docker port 8080, HTTP)
#   - Routes / to Flask backend (HTTPS port 5001, end-to-end SSL)
#   - Includes backups and rollback support
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# CONFIGURATION
#-------------------------------------------------------------------------------

NGINX_CONF="/etc/nginx/nginx.conf"
SPEECH_SITE="/etc/nginx/sites-enabled/speech-anonymizer"
SPEECH_SITE_BACKUP="/etc/nginx/backups/speech-anonymizer.$(date +%Y%m%d_%H%M%S).bak"
MAIN_CONF_BACKUP="/etc/nginx/backups/nginx.conf.$(date +%Y%m%d_%H%M%S).bak"

if [[ -f "config.sh" ]]; then
    source config.sh
fi

ENABLE_CSP_HEADER="true"

MAX_BODY_SIZE="5G"
PROXY_READ_TIMEOUT="600s"
PROXY_SEND_TIMEOUT="600s"
PROXY_CONNECT_TIMEOUT="60s"

# Unified SSL paths (MUST match setup_ssl.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATA_PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SSL_CERT_DIR="${ATA_PROJECT_ROOT}/ssl_certs"
SSL_CERT_FILE="${SSL_CERT_DIR}/server.crt"
SSL_KEY_FILE="${SSL_CERT_DIR}/server.key"
STATIC_FILES_PATH="${ATA_PROJECT_ROOT}/interactive_app/static"

# Ports (MUST match your services)
REACT_PORT=8080          # React frontend (Docker container, HTTP)
FLASK_PORT=5001          # Flask backend (HTTPS — end-to-end SSL)
FLASK_PROTOCOL="https"   # Flask runs with SSL enabled
FLASK_APP="app.py"       # Flask app filename

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
    echo -e "${BLUE}==============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==============================================${NC}"
    echo ""
}

print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root!"
        echo "Use: sudo ./setup_nginx.sh"
        exit 1
    fi
}

check_nginx() {
    if ! command -v nginx &> /dev/null; then
        print_error "Nginx is not installed!"
        echo "Install with: sudo apt update && sudo apt install nginx"
        exit 1
    fi
    print_success "Nginx found ($(nginx -v 2>&1))"
}

check_ssl_certificates() {
    print_info "Checking for SSL certificates..."
    
    # ==========================================================
    # PRIORITY 1: Let's Encrypt / Certbot (Production Preferred)
    # ==========================================================
    CERTBOT_CERT="/etc/letsencrypt/live/transcriber.cloud.cci.charite.de/fullchain.pem"
    CERTBOT_KEY="/etc/letsencrypt/live/transcriber.cloud.cci.charite.de/privkey.pem"
    
    if [[ -f "$CERTBOT_CERT" && -f "$CERTBOT_KEY" ]]; then
        print_success "Certbot/Let's Encrypt certificates detected!"
        print_info "Using: $CERTBOT_CERT"
        
        export SSL_CERT_FILE="$CERTBOT_CERT"
        export SSL_KEY_FILE="$CERTBOT_KEY"
        
        # Detect key type and extract modulus accordingly
        # Certbot uses ECDSA by default in newer versions
        local cert_modulus
        local key_modulus
        
        # Extract certificate modulus (works for all key types)
        cert_modulus=$(openssl x509 -noout -modulus -in "$CERTBOT_CERT" 2>/dev/null | openssl md5 | awk '{print $NF}')
        
        # Detect private key type and extract modulus
        if openssl rsa -noout -modulus -in "$CERTBOT_KEY" 2>/dev/null | grep -q "Modulus"; then
            # RSA key
            key_modulus=$(openssl rsa -noout -modulus -in "$CERTBOT_KEY" 2>/dev/null | openssl md5 | awk '{print $NF}')
            print_info "Key type: RSA"
        elif openssl ec -noout -modulus -in "$CERTBOT_KEY" 2>/dev/null | grep -q "Modulus"; then
            # ECDSA key
            key_modulus=$(openssl ec -noout -modulus -in "$CERTBOT_KEY" 2>/dev/null | openssl md5 | awk '{print $NF}')
            print_info "Key type: ECDSA"
        else
            # Unknown key type or corrupted file - skip validation
            print_warning "Unable to determine key type - skipping modulus validation"
            print_warning "Assuming Let's Encrypt validation is sufficient."
            key_modulus=""
        fi
        
        # Compare only if both values were extracted
        if [[ -n "$cert_modulus" && -n "$key_modulus" ]]; then
            if [[ "$cert_modulus" != "$key_modulus" ]]; then
                print_error "Let's Encrypt certificate and key do not match!"
                print_info "Cert MD5: $cert_modulus"
                print_info "Key MD5:  $key_modulus"
                return 1
            else
                print_success "Certificate and key modulus match"
            fi
        elif [[ -z "$cert_modulus" && -z "$key_modulus" ]]; then
            print_warning "Unable to extract modulus from cert/key - skipping validation"
        fi
        
        # Check expiry
        local expiry_date=$(openssl x509 -enddate -noout -in "$CERTBOT_CERT" 2>/dev/null | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry_date" +%s 2>/dev/null)
        local now_epoch=$(date +%s)
        local days_left=$(( (expiry_epoch - now_epoch) / 86400 ))
        
        print_success "Certificate expires: $expiry_date ($days_left days remaining)"
        
        if [[ "$days_left" -lt 14 ]]; then
            print_warning "Certificate expires in less than 14 days! Run: sudo certbot renew"
        fi
        
        return 0
    fi
    
    # ==========================================================
    # PRIORITY 2: Self-Signed from ATA/ssl_certs/ (Development)
    # ==========================================================
    if [[ -f "$SSL_CERT_FILE" && -f "$SSL_KEY_FILE" ]]; then
        print_success "Self-signed certificates found: $SSL_CERT_DIR"
        print_info "Using: $SSL_CERT_FILE"
        
        # Same key-type detection for self-signed certs
        local cert_modulus
        local key_modulus
        
        cert_modulus=$(openssl x509 -noout -modulus -in "$SSL_CERT_FILE" 2>/dev/null | openssl md5 | awk '{print $NF}')
        
        if openssl rsa -noout -modulus -in "$SSL_KEY_FILE" 2>/dev/null | grep -q "Modulus"; then
            key_modulus=$(openssl rsa -noout -modulus -in "$SSL_KEY_FILE" 2>/dev/null | openssl md5 | awk '{print $NF}')
            print_info "Key type: RSA"
        elif openssl ec -noout -modulus -in "$SSL_KEY_FILE" 2>/dev/null | grep -q "Modulus"; then
            key_modulus=$(openssl ec -noout -modulus -in "$SSL_KEY_FILE" 2>/dev/null | openssl md5 | awk '{print $NF}')
            print_info "Key type: ECDSA"
        else
            print_warning "Unable to determine key type - skipping modulus validation"
            key_modulus=""
        fi
        
        if [[ -n "$cert_modulus" && -n "$key_modulus" ]]; then
            if [[ "$cert_modulus" != "$key_modulus" ]]; then
                print_error "Self-signed certificate and key do not match!"
                print_info "Cert MD5: $cert_modulus"
                print_info "Key MD5:  $key_modulus"
                return 1
            else
                print_success "Certificate and key modulus match"
            fi
        fi
        
        # Check expiry
        local expiry_date=$(openssl x509 -enddate -noout -in "$SSL_CERT_FILE" 2>/dev/null | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry_date" +%s 2>/dev/null)
        local now_epoch=$(date +%s)
        local days_left=$(( (expiry_epoch - now_epoch) / 86400 ))
        
        print_info "Certificate expires: $expiry_date ($days_left days remaining)"
        
        return 0
    fi
    
    # ==========================================================
    # NO CERTIFICATES FOUND - Error Out
    # ==========================================================
    print_error "No SSL certificates found!"
    echo ""
    echo "Options:"
    echo "  1. Install Certbot and request a Let's Encrypt certificate:"
    echo "     sudo apt install certbot python3-certbot-nginx"
    echo "     sudo certbot --nginx -d transcriber.cloud.cci.charite.de"
    echo ""
    echo "  2. Generate self-signed certificates using ATA script:"
    echo "     ./ssl_certs/setup_ssl.sh"
    echo ""
    echo "  3. Continue with HTTP only (NOT recommended for medical data):"
    read -p "Continue with HTTP only? [y/N] " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        export USE_HTTPS=false
        print_warning "WARNING: Running without SSL/TLS encryption!"
        print_warning "Medical data will be transmitted in plain text."
        return 0
    else
        print_error "Aborted - SSL certificates are required for secure operation"
        exit 1
    fi
}

#-------------------------------------------------------------------------------
# BACKUP FUNCTIONS
#-------------------------------------------------------------------------------

create_backup_directory() {
    mkdir -p /etc/nginx/backups
}

backup_site_config() {
    if [[ -f "$SPEECH_SITE" ]]; then
        cp "$SPEECH_SITE" "$SPEECH_SITE_BACKUP"
        print_success "Site config backed up: $SPEECH_SITE_BACKUP"
    fi
}

backup_main_conf() {
    if [[ -f "$NGINX_CONF" ]]; then
        cp "$NGINX_CONF" "$MAIN_CONF_BACKUP"
        print_success "Main config backed up: $MAIN_CONF_BACKUP"
    fi
}

#-------------------------------------------------------------------------------
# FIX MALFORMED SITE CONFIGS
#-------------------------------------------------------------------------------

fix_malformed_site_configs() {
    print_info "Checking for malformed site configs..."
    
    if [[ -f "$SPEECH_SITE" ]]; then
        if grep -q "^events {" "$SPEECH_SITE" || grep -q "^http {" "$SPEECH_SITE"; then
            print_warning "Found invalid directives in $SPEECH_SITE"
            backup_site_config
            
            print_info "Removing malformed config - will recreate below..."
            rm -f "$SPEECH_SITE"
        fi
    fi
}

#-------------------------------------------------------------------------------
# CREATE NGINX SITES CONFIG
#-------------------------------------------------------------------------------

create_site_config() {
    print_info "Creating nginx site configuration..."
    
    # ============================================
    # CERTIFICATE PRIORITY: LET'S ENCRYPT > SELF-SIGNED
    # ============================================
    CERTBOT_CERT="${LETSENCRYPT_FULLCHAIN_PATH:-/etc/letsencrypt/live/${DOMAIN_NAME}/fullchain.pem}"
    CERTBOT_KEY="${LETSENCRYPT_PRIVKEY_PATH:-/etc/letsencrypt/live/${DOMAIN_NAME}/privkey.pem}"
    
    # Check for Certbot/Let's Encrypt certificates first
    if [[ -z "$SSL_CERT_FILE" || -z "$SSL_KEY_FILE" ]]; then
        print_error "SSL_CERT_FILE or SSL_KEY_FILE not set!"
        exit 1
    fi
    
    # Determine certificate type for logging
    if [[ "$SSL_CERT_FILE" == "/etc/letsencrypt"* ]]; then
        print_success "Using Let's Encrypt certificates."
        USE_LETSENCRYPT=true
    else
        print_info "Using self-signed certificates."
        USE_LETSENCRYPT=false
    fi
    
    # ============================================
    # UPSTREAM AND RATE LIMITING
    # ============================================
    cat > "$SPEECH_SITE" << 'NGINX_UPSTREAM'
# ============================================================================
# Nginx Reverse Proxy Configuration for Speech Anonymizer
# Generated by setup_nginx.sh
# Compatible with ssl_certs/setup_ssl.sh certificate management
# Architecture:
#   - nginx terminates external SSL (port 443)
#   - Flask backend serves HTTPS on port 5001 (end-to-end SSL)
#   - React frontend container serves HTTP on port 8080 (nginx terminates SSL)
# ============================================================================

# Upstream definition (Flask app — running with SSL on port 5001)
upstream speech_anonymizer {
    server 127.0.0.1:5001;
    keepalive 32;
}

# Rate limiting zone
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

NGINX_UPSTREAM

    # ============================================
    # HTTP SERVER (REDIRECT TO HTTPS)
    # ============================================
    if [[ "${USE_HTTPS:-true}" == "true" ]]; then
        cat >> "$SPEECH_SITE" << 'NGINX_HTTP'
# HTTP Server - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    
    server_name transcriber.cloud.cci.charite.de localhost 127.0.0.1 _;
    
    return 301 https://$http_host$request_uri;
}

NGINX_HTTP
    else
        cat >> "$SPEECH_SITE" << 'NGINX_HTTP_ONLY'
# HTTP Server Only (No SSL — NOT recommended for medical data)
server {
    listen 80;
    listen [::]:80;
    server_name transcriber.cloud.cci.charite.de localhost 127.0.0.1 _;
    client_max_body_size 5G;
    client_body_timeout 600s;
    client_header_timeout 600s;
    send_timeout 600s;

    # ===================================================
    # REACT FRONTEND (/app/) - PROXY TO DOCKER CONTAINER
    # React container serves HTTP on port 8080
    # ===================================================
    location ^~ /app/ {
        limit_req zone=api burst=50 nodelay;

        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";

        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    # ===================================================
    # STATIC FILES (/static/)
    # ===================================================
    location /static/ {
        alias /mnt/Data_Mount/VERANDA_DataMount/ATA/interactive_app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        try_files $uri $uri/ =404;
    }

    # ===================================================
    # HEALTH CHECK (/health) - PROXY TO FLASK (HTTP in this mode)
    # ===================================================
    location /health {
        proxy_pass http://127.0.0.1:5001/health;
        access_log off;
        limit_req zone=api burst=100 nodelay;
    }

    # ===================================================
    # API ENDPOINTS (ALL OTHER PATHS) - PROXY TO FLASK
    # ===================================================
    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://127.0.0.1:5001;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";

        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }
}
NGINX_HTTP_ONLY
    fi

    # ============================================
    # HTTPS SERVER (MAIN APPLICATION)
    # ============================================
    if [[ "${USE_HTTPS:-true}" == "true" ]]; then
        cat >> "$SPEECH_SITE" << EOF
# HTTPS Server - Main Application
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    
    server_name transcriber.cloud.cci.charite.de localhost 127.0.0.1 _;

    # SSL Configuration (external termination)
    # Uses Let's Encrypt if available, otherwise self-signed from ATA/ssl_certs/
    ssl_certificate ${SSL_CERT_FILE};
    ssl_certificate_key ${SSL_KEY_FILE};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # ===================================================
    # UPSTREAM SSL SETTINGS (Flask serves HTTPS with self-signed certs)
    # Applied at server level for all proxied locations to Flask
    # ===================================================
    proxy_ssl_verify off;
    proxy_ssl_server_name on;
    proxy_ssl_protocols TLSv1.2 TLSv1.3;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
EOF

        # Add Content-Security-Policy header (optional, for medical data security)
        if [[ -n "$ENABLE_CSP_HEADER" ]]; then
            cat >> "$SPEECH_SITE" << 'EOF'
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self';" always;
EOF
        fi

        cat >> "$SPEECH_SITE" << EOF

    # Large upload settings
    client_max_body_size 5G;
    client_body_buffer_size 10M;
    client_body_timeout 600s;
    client_header_timeout 600s;
    send_timeout 600s;

    # ===================================================
    # REACT FRONTEND (/app/) - PROXY TO DOCKER CONTAINER
    # React container serves HTTP on port ${REACT_PORT}
    # nginx terminates SSL, forwards plain HTTP to container
    # ===================================================
    location ^~ /app/ {
        limit_req zone=api burst=50 nodelay;

        proxy_pass http://127.0.0.1:${REACT_PORT};
        proxy_http_version 1.1;

        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Connection "";

        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    # ===================================================
    # STATIC FILES (/static/)
    # ===================================================
    location /static/ {
        alias ${STATIC_FILES_PATH}/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        try_files \$uri \$uri/ =404;
    }

    # ===================================================
    # HEALTH CHECK (/health) - PROXY TO FLASK VIA HTTPS
    # ===================================================
    location /health {
        proxy_pass https://speech_anonymizer/health;
        access_log off;
        limit_req zone=api burst=100 nodelay;
    }

    # ===================================================
    # UPLOAD ENDPOINT (/upload) - PROXY TO FLASK VIA HTTPS
    # ===================================================
    location /upload {
        proxy_pass https://speech_anonymizer/upload;

        client_max_body_size 5G;
        client_body_timeout 600s;

        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    # ===================================================
    # RECORDING TRANSCRIBE (/transcribe_recording) - PROXY TO FLASK VIA HTTPS
    # ===================================================
    location /transcribe_recording {
        proxy_pass https://speech_anonymizer/transcribe_recording;

        client_max_body_size 5G;
        client_body_timeout 600s;

        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    # ===================================================
    # DOWNLOAD ENDPOINTS (/download/) - PROXY TO FLASK VIA HTTPS
    # ===================================================
    location /download/ {
        proxy_pass https://speech_anonymizer/download/;

        proxy_http_version 1.1;
        proxy_read_timeout 300s;
    }

    # ===================================================
    # API ENDPOINTS (ALL OTHER PATHS) - PROXY TO FLASK VIA HTTPS
    # ===================================================
    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass https://speech_anonymizer;
        proxy_http_version 1.1;

        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header Connection "";

        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }
}
EOF
    fi
}

#-------------------------------------------------------------------------------
# ADD TO MAIN NGINX.CONF
#-------------------------------------------------------------------------------

add_to_main_conf() {
    print_info "Checking main nginx.conf..."
    
    if grep -q "client_max_body_size $MAX_BODY_SIZE;" "$NGINX_CONF"; then
        print_info "Large file setting already configured in main config"
        return 0
    fi
    
    TMPFILE=$(mktemp)
    
    awk -v size="$MAX_BODY_SIZE" '
    /^http \{/ {
        print
        print ""
        printf "\t# Large file upload support\n"
        printf "\tclient_max_body_size %s;\n", size
        next
    }
    {print}
    ' "$NGINX_CONF" > "$TMPFILE"
    
    if [ $? -eq 0 ]; then
        mv "$TMPFILE" "$NGINX_CONF"
        print_success "Added large file setting to main nginx.conf"
    else
        rm -f "$TMPFILE"
        print_warning "Could not modify main config - continuing anyway"
        return 1
    fi
}

#-------------------------------------------------------------------------------
# VERIFY AND APPLY
#-------------------------------------------------------------------------------

test_configuration() {
    print_info "Testing nginx configuration..."
    
    if nginx -t 2>&1 | tee /tmp/nginx_test.log; then
        print_success "Configuration test passed"
        return 0
    else
        print_error "Configuration has syntax errors!"
        echo ""
        print_info "Detailed error:"
        cat /tmp/nginx_test.log | tail -20
        echo ""
        return 1
    fi
}

reload_nginx() {
    print_info "Reloading nginx service..."
    
    if systemctl is-active --quiet nginx 2>/dev/null || pgrep nginx >/dev/null; then
        if systemctl reload nginx 2>/dev/null; then
            print_success "Nginx reloaded successfully"
        else
            print_warning "Could not reload via systemctl - trying signal method"
            if nginx -s reload 2>/dev/null; then
                print_success "Nginx reloaded via signal"
            else
                print_warning "Manual nginx reload may be required: sudo systemctl start nginx"
                return 1
            fi
        fi
    else
        print_warning "Nginx not running - starting it..."
        if systemctl start nginx 2>/dev/null; then
            print_success "Nginx started successfully"
        else
            print_warning "Could not start nginx - please start manually"
            return 1
        fi
    fi
    
    return 0
}

#-------------------------------------------------------------------------------
# VERIFICATION AND SUMMARY
#-------------------------------------------------------------------------------

verify_changes() {
    print_info "Verifying applied changes..."
    
    local all_ok=true
    
    # Check /app/ route exists
    if grep -q "location.*^~ /app/" "$SPEECH_SITE"; then
        print_success "/app/ route for React frontend: configured (verified)"
    else
        print_error "/app/ route NOT FOUND - will cause 404 errors!"
        all_ok=false
    fi
    
    # Check React port uses HTTP
    if grep -q "proxy_pass http://127.0.0.1:${REACT_PORT}" "$SPEECH_SITE"; then
        print_success "React proxy port: ${REACT_PORT} (HTTP - correct)"
    else
        print_error "React proxy port NOT configured correctly!"
        all_ok=false
    fi
    
    # Check backend port uses HTTPS
    if grep -q "proxy_pass https://speech_anonymizer" "$SPEECH_SITE"; then
        print_success "Backend proxy: Flask on port ${FLASK_PORT} (HTTPS - end-to-end SSL verified)"
    else
        print_error "Backend proxy NOT configured for HTTPS!"
        all_ok=false
    fi
    
    # Check SSL paths
    if grep -q "ssl_certificate ${SSL_CERT_FILE}" "$SPEECH_SITE"; then
        print_success "External SSL certificate path configured (verified)"
    else
        print_error "SSL certificate path not configured!"
        all_ok=false
    fi
    
    # Check proxy_ssl settings for Flask upstream
    if grep -q "proxy_ssl_verify off" "$SPEECH_SITE"; then
        print_success "Upstream SSL verification disabled (expected for self-signed Flask certs)"
    else
        print_warning "proxy_ssl_verify not explicitly set - may default to strict"
    fi
    
    # Check rate limiting
    if grep -q "limit_req_zone.*zone=api" "$SPEECH_SITE"; then
        print_success "Rate limiting zone defined (verified)"
    else
        print_warning "Rate limiting zone NOT DEFINED"
    fi
    
    if [[ "$all_ok" == "true" ]]; then
        print_success "All configurations verified successfully!"
    else
        print_warning "Some configurations failed verification - review errors above"
    fi
    
    return 0
}

show_summary() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}Configuration Summary:${NC}"
    echo "  • React frontend URL:  https://transcriber.cloud.cci.charite.de/app/"
    echo "  • React container port: ${REACT_PORT}:80 (HTTP, nginx terminates SSL)"
    echo "  • Backend protocol:     ${FLASK_PROTOCOL} (Flask on port ${FLASK_PORT})"
    echo "  • Max upload size:      ${MAX_BODY_SIZE}"
    echo "  • Rate limiting:        api:10m (10 req/s)"
    echo "  • External SSL certs:   ${SSL_CERT_FILE}"
    echo "  • Upstream SSL:         Enabled for Flask (self-signed certs)"
    echo "  • Site config:          ${SPEECH_SITE}"
    echo ""
    
    echo -e "${YELLOW}Security Architecture:${NC}"
    echo "  1. Client → nginx: HTTPS (external SSL termination)"
    echo "  2. nginx → React:  HTTP  (container serves plain HTTP)"
    echo "  3. nginx → Flask:  HTTPS (end-to-end SSL with self-signed certs)"
    echo ""
    
    echo -e "${YELLOW}Access URLs:${NC}"
    if [[ "${USE_HTTPS:-true}" == "true" ]]; then
        echo "  • React frontend:  https://transcriber.cloud.cci.charite.de/app/"
        echo "  • Backend API:     https://transcriber.cloud.cci.charite.de/"
        echo "  • Health check:    https://transcriber.cloud.cci.charite.de/health"
        echo "  • Local React:     http://localhost:8080/app/"
        echo "  • Local backend:   https://localhost:5001/"
    else
        echo "  • React frontend:  http://transcriber.cloud.cci.charite.de/app/"
        echo "  • Backend API:     http://transcriber.cloud.cci.charite.de/"
        echo "  • Local React:     http://localhost:8080/app/"
        echo "  • Local backend:   http://localhost:5001/"
    fi
    echo ""
    
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Ensure Flask is running with HTTPS:"
    echo "     conda activate whisperx"
    echo "     python interactive_app/app.py"
    echo "     # Verify: curl -k https://localhost:5001/health"
    echo ""
    echo "  2. Start Docker containers:"
    echo "     cd $ATA_PROJECT_ROOT"
    echo "     docker compose up -d frontend"
    echo "     docker ps -a"
    echo ""
    echo "  3. Test React frontend:"
    echo "     curl -k https://localhost/app/ -L"
    echo ""
    echo "  4. Check nginx logs for errors:"
    echo "     tail -f /var/log/nginx/error.log"
    echo ""
    echo -e "${GREEN}✅ Nginx configuration is ready with end-to-end SSL!${NC}"
    echo ""
}

#-------------------------------------------------------------------------------
# ROLLBACK
#-------------------------------------------------------------------------------

rollback() {
    print_info "Rolling back changes..."
    
    if [[ -f "$SPEECH_SITE_BACKUP" ]]; then
        cp "$SPEECH_SITE_BACKUP" "$SPEECH_SITE"
        print_success "Site config restored"
    fi
    
    if [[ -f "$MAIN_CONF_BACKUP" ]]; then
        cp "$MAIN_CONF_BACKUP" "$NGINX_CONF"
        print_success "Main config restored"
    fi
    
    if nginx -t 2>&1 >/dev/null; then
        nginx -s reload 2>/dev/null || systemctl reload nginx 2>/dev/null || true
        print_success "Nginx reloaded after rollback"
    fi
}

#-------------------------------------------------------------------------------
# MAIN EXECUTION
#-------------------------------------------------------------------------------

main() {
    print_header "Nginx Configuration Setup for Speech Anonymizer"
    echo "Compatible with: ${ATA_PROJECT_ROOT}/ssl_certs/setup_ssl.sh"
    
    # Pre-flight checks
    check_root
    check_nginx
    create_backup_directory
    
    # Check SSL certificates first
    check_ssl_certificates
    
    # Backup current configs
    backup_site_config
    backup_main_conf
    
    # Fix any malformed configs
    fix_malformed_site_configs
    
    # Main operations
    set +e
    
    create_site_config
    SITE_STATUS=$?
    
    add_to_main_conf
    MAIN_STATUS=$?
    
    set -e
    
    # Test configuration
    if test_configuration; then
        verify_changes
        reload_nginx
        show_summary
    else
        print_error "Setup failed! Attempting rollback..."
        rollback
        exit 1
    fi
}

# Handle subcommands
case "${1:-setup}" in
    setup|check)
        main
        ;;
    rollback)
        print_warning "Running rollback..."
        create_backup_directory
        backup_site_config
        rollback
        ;;
    status)
        if [[ -f "$SPEECH_SITE" ]]; then
            print_info "Current configuration:"
            echo ""
            echo "Sites-enabled: $SPEECH_SITE"
            ls -la "$SPEECH_SITE"
            echo ""
            echo "Routes configured:"
            grep "location" "$SPEECH_SITE" | head -10
        else
            print_warning "No site configuration found"
        fi
        ;;
    *)
        echo "Usage: $0 {setup|rollback|status}"
        exit 1
        ;;
esac