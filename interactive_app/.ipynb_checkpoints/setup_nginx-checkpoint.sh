#!/usr/bin/env bash
#===============================================================================
# Nginx Configuration Setup for Speech Anonymizer
# Features:
#   - Auto-detects SSL certificates from ssl_certs/ directory
#   - Compatible with setup_ssl.sh certificate management
#   - Routes /app/ to React frontend (Docker port 8080)
#   - Routes / to Flask backend (HTTPS port 5001)
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
REACT_PORT=8080          # React frontend (Docker container)
FLASK_PORT=5002          # Flask backend (HTTPS)
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
    
    if [[ -f "$SSL_CERT_FILE" && -f "$SSL_KEY_FILE" ]]; then
        print_success "Certificates found: $SSL_CERT_DIR"
        return 0
    else
        print_warning "No certificates found in $SSL_CERT_DIR!"
        echo ""
        echo -e "${YELLOW}Please run the SSL certificate setup script first:${NC}"
        echo "  cd $ATA_PROJECT_ROOT"
        echo "  ./ssl_certs/setup_ssl.sh"
        echo ""
        echo "Or continue without SSL (HTTP only):"
        read -p "Continue with HTTP only? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            export USE_HTTPS=false
            return 0
        else
            print_error "Aborted - SSL certificates are required for HTTPS"
            exit 1
        fi
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
# CREATE NGINX SITES CONFIG (UPDATED WITH /app/ ROUTE)
#-------------------------------------------------------------------------------

create_site_config() {
    print_info "Creating nginx site configuration..."
    
    # ============================================
    # UPSTREAM AND RATE LIMITING
    # ============================================
    cat > "$SPEECH_SITE" << 'NGINX_UPSTREAM'
# ============================================================================
# Nginx Reverse Proxy Configuration for Speech Anonymizer
# Generated by setup_nginx.sh
# Compatible with ssl_certs/setup_ssl.sh certificate management
# UPDATED: Added /app/ route for React frontend (port 8080)
# FIXED: Connects to Flask backend via HTTPS (port 5001)
# ============================================================================

# Upstream definition (Flask app - running with SSL)
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
# HTTP Server Only (No SSL)
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
    # ===================================================
    location ^~ /app/ {
        limit_req zone=api burst=50 nodelay;

        proxy_pass https://127.0.0.1:8080;
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
    # HEALTH CHECK (/health)
    # ===================================================
    location /health {
        proxy_pass https://127.0.0.1:5001/health;
        access_log off;
        limit_req zone=api burst=100 nodelay;
    }

    # ===================================================
    # API ENDPOINTS (ALL OTHER PATHS) - PROXY TO FLASK
    # ===================================================
    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass https://127.0.0.1:5001;
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

    # SSL Configuration (from ssl_certs/ directory)
    ssl_certificate ${SSL_CERT_FILE};
    ssl_certificate_key ${SSL_KEY_FILE};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Large upload settings
    client_max_body_size 5G;
    client_body_buffer_size 10M;
    client_body_timeout 600s;
    client_header_timeout 600s;
    send_timeout 600s;

    # ===================================================
    # REACT FRONTEND (/app/) - PROXY TO DOCKER CONTAINER
    # ===================================================
    location ^~ /app/ {
        limit_req zone=api burst=50 nodelay;

        proxy_pass https://127.0.0.1:${REACT_PORT};
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
    # HEALTH CHECK (/health)
    # ===================================================
    location /health {
        proxy_pass http://speech_anonymizer/health;
        proxy_ssl_verify off;
        proxy_ssl_server_name on;
        access_log off;
        limit_req zone=api burst=100 nodelay;
    }

    # ===================================================
    # UPLOAD ENDPOINT (/upload)
    # ===================================================
    location /upload {
        proxy_pass http://speech_anonymizer/upload;
        proxy_ssl_verify off;
        proxy_ssl_server_name on;

        client_max_body_size 5G;
        client_body_timeout 600s;

        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    # ===================================================
    # RECORDING TRANSCRIBE (/transcribe_recording)
    # ===================================================
    location /transcribe_recording {
        proxy_pass http://speech_anonymizer/transcribe_recording;
        proxy_ssl_verify off;
        proxy_ssl_server_name on;

        client_max_body_size 5G;
        client_body_timeout 600s;

        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    # ===================================================
    # DOWNLOAD ENDPOINTS (/download/)
    # ===================================================
    location /download/ {
        proxy_pass http://speech_anonymizer/download/;
        proxy_ssl_verify off;
        proxy_ssl_server_name on;

        proxy_http_version 1.1;
        proxy_read_timeout 300s;
    }

    # ===================================================
    # API ENDPOINTS (ALL OTHER PATHS) - PROXY TO FLASK
    # ===================================================
    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://speech_anonymizer;
        proxy_http_version 1.1;

        proxy_ssl_verify off;
        proxy_ssl_server_name on;
        proxy_ssl_session_reuse on;

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
    
    # Check React port
    if grep -q "proxy_pass https://127.0.0.1:${REACT_PORT}" "$SPEECH_SITE"; then
        print_success "React proxy port: ${REACT_PORT} (verified)"
    else
        print_error "React proxy port NOT configured!"
        all_ok=false
    fi
    
    # Check backend port
    if grep -q "proxy_pass http://127.0.0.1:5001;" "$SPEECH_SITE" || grep -q "proxy_pass http://127.0.0.1:${FLASK_PORT}" "$SPEECH_SITE"; then
        print_success "Backend proxy: Flask on port ${FLASK_PORT} (verified)"
    else
        print_error "Backend proxy NOT configured!"
        all_ok=false
    fi
    
    # Check SSL paths
    if grep -q "ssl_certificate ${SSL_CERT_FILE}" "$SPEECH_SITE"; then
        print_success "SSL certificate path configured (verified)"
    else
        print_error "SSL certificate path not configured!"
        all_ok=false
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
    echo "  • React container port: ${REACT_PORT}:80"
    echo "  • Backend protocol:     ${FLASK_PROTOCOL} (to Flask on port ${FLASK_PORT})"
    echo "  • Max upload size:      ${MAX_BODY_SIZE}"
    echo "  • Rate limiting:        api:10m (10 req/s)"
    echo "  • SSL certificates:     ${SSL_CERT_DIR}"
    echo "  • Site config:          ${SPEECH_SITE}"
    echo ""
    
    echo -e "${YELLOW}Access URLs:${NC}"
    if [[ "${USE_HTTPS:-true}" == "true" ]]; then
        echo "  • React frontend:  https://transcriber.cloud.cci.charite.de/app/"
        echo "  • Backend API:     https://transcriber.cloud.cci.charite.de/"
        echo "  • Health check:    https://transcriber.cloud.cci.charite.de/health"
        echo "  • Local React:     https://localhost:8080/app/"
        echo "  • Local backend:   https://localhost:5001/"
    else
        echo "  • React frontend:  http://transcriber.cloud.cci.charite.de/app/"
        echo "  • Backend API:     http://transcriber.cloud.cci.charite.de/"
        echo "  • Local React:     http://localhost:8080/app/"
        echo "  • Local backend:   http://localhost:5001/"
    fi
    echo ""
    
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Start Docker containers:"
    echo "     cd $ATA_PROJECT_ROOT"
    echo "     docker compose up -d frontend"
    echo ""
    echo "  2. Verify containers are running:"
    echo "     docker ps -a"
    echo ""
    echo "  3. Test React frontend:"
    echo "     curl -k https://localhost/app/"
    echo ""
    echo "  4. Check nginx logs for errors:"
    echo "     tail -f /var/log/nginx/error.log"
    echo ""
    echo -e "${GREEN}✅ Nginx configuration is ready!${NC}"
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
    
    if nginx -t 2>/dev/null; then
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