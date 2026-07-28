#!/bin/bash
#===============================================================================
# Nginx Configuration Setup for Speech Anonymizer
# Uses unified SSL certificates from ATA/ssl_certs/
# Compatible with setup_ssl.sh certificate management
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
RATE_LIMIT_ZONE="api:10m"
RATE_LIMIT_RATE="10r/s"

# Unified SSL paths (managed by setup_ssl.sh)
SSL_CERT_DIR="/mnt/Data_Mount/VERANDA_DataMount/ATA/ssl_certs"
SSL_CERT_FILE="${SSL_CERT_DIR}/server.crt"
SSL_KEY_FILE="${SSL_CERT_DIR}/server.key"
ATA_PROJECT_ROOT="/mnt/Data_Mount/VERANDA_DataMount/ATA"
STATIC_FILES_PATH="${ATA_PROJECT_ROOT}/interactive_app/static"

# Flask app port
FLASK_PORT=5001

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
        print_warning "No certificates found!"
        echo ""
        echo -e "${YELLOW}Please run the certificate setup script first:${NC}"
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
# CREATE NGINX SITES CONFIG (CORRECTED VERSION)
#-------------------------------------------------------------------------------

create_site_config() {
    print_info "Creating nginx site configuration..."
    
    # ============================================
    # UPSTREAM AND RATE LIMITING (MUST BE FIRST)
    # ============================================
    cat > "$SPEECH_SITE" << 'NGINX_UPSTREAM'
# ============================================================================
# Nginx Reverse Proxy Configuration for Speech Anonymizer
# Generated by setup_nginx.sh
# Uses unified SSL certificates from ATA/ssl_certs/
# ============================================================================

# Upstream definition (Flask app)
upstream speech_anonymizer {
    server 127.0.0.1:5001;
    keepalive 32;
}

# Rate limiting zone (REQUIRED - defines memory zone for rate limiting)
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
    
    # Handle domain and localhost (FIXED - was causing underscore redirect)
    server_name transcriber.cloud.cci.charite.de localhost 127.0.0.1 _;
    
    # Redirect HTTP to HTTPS (preserves actual host header)
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
    
    # Handle both domain and localhost
    server_name transcriber.cloud.cci.charite.de localhost 127.0.0.1 _;

    # SSL Configuration (from unified location)
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
EOF
    else
        # Close HTTP-only server block
        cat >> "$SPEECH_SITE" << 'NGINX_CLOSE_HTTP'
}
NGINX_CLOSE_HTTP
    fi
    
    # ============================================
    # COMMON SETTINGS (LOCATION BLOCKS)
    # ============================================
    cat >> "$SPEECH_SITE" << 'NGINX_COMMON'

    # ============================================
    # LARGE UPLOAD SETTINGS (5 GB)
    # ============================================
    client_max_body_size 5G;
    client_body_buffer_size 10M;

    # Timeouts for large uploads
    client_body_timeout 600s;
    client_header_timeout 600s;
    send_timeout 600s;

    location / {
        # Rate limiting (exempt health check)
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://speech_anonymizer;
        proxy_http_version 1.1;

        # Essential headers for proper proxying
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header Connection "";

        # Buffering settings for large uploads
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    location /health {
        proxy_pass http://speech_anonymizer/health;
        access_log off;
        limit_req zone=api burst=100 nodelay;
    }

    location /static/ {
        alias /mnt/Data_Mount/VERANDA_DataMount/ATA/interactive_app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        try_files $uri $uri/ =404;
    }

    # API endpoints - large upload support
    location /upload {
        client_max_body_size 5G;
        client_body_timeout 600s;

        proxy_pass http://speech_anonymizer/upload;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    location /transcribe_recording {
        client_max_body_size 5G;
        client_body_timeout 600s;

        proxy_pass http://speech_anonymizer/transcribe_recording;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }

    # Download endpoints
    location /download/ {
        proxy_pass http://speech_anonymizer/download/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
NGINX_COMMON
}

#-------------------------------------------------------------------------------
# ADD CLIENT_MAX_BODY_SIZE TO MAIN NGINX.CONF
#-------------------------------------------------------------------------------

add_to_main_conf() {
    print_info "Checking main nginx.conf..."
    
    if grep -q "client_max_body_size $MAX_BODY_SIZE;" "$NGINX_CONF"; then
        print_info "Already configured in main config"
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
        print_success "Added to main nginx.conf"
    else
        rm -f "$TMPFILE"
        print_error "Failed to modify main config"
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
            print_warning "Could not reload via systemctl (trying signal method)..."
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
    
    # Check client_max_body_size
    if grep -q "client_max_body_size 5G" "$SPEECH_SITE"; then
        print_success "Large upload support: 5G (verified)"
    else
        print_error "Large upload setting not found!"
        all_ok=false
    fi
    
    # Check rate limiting zone (CRITICAL)
    if grep -q "limit_req_zone.*zone=api" "$SPEECH_SITE"; then
        print_success "Rate limiting zone defined: api (verified)"
    else
        print_error "Rate limiting zone NOT DEFINED!"
        all_ok=false
    fi
    
    # Check server_name (FIXED)
    if grep -q "server_name transcriber.cloud.cci.charite.de" "$SPEECH_SITE"; then
        print_success "Server name configured: transcriber.cloud.cci.charite.de (verified)"
    else
        print_error "Server name not configured correctly!"
        all_ok=false
    fi
    
    # Check SSL paths (if HTTPS enabled)
    if [[ "${USE_HTTPS:-true}" == "true" ]]; then
        if grep -q "ssl_certificate ${SSL_CERT_FILE}" "$SPEECH_SITE"; then
            print_success "SSL certificate path configured (verified)"
        else
            print_error "SSL certificate path not configured!"
            all_ok=false
        fi
        
        if grep -q "ssl_certificate_key ${SSL_KEY_FILE}" "$SPEECH_SITE"; then
            print_success "SSL key path configured (verified)"
        else
            print_error "SSL key path not configured!"
            all_ok=false
        fi
    fi
    
    # Check proxy timeouts
    if grep -q "proxy_read_timeout 600s" "$SPEECH_SITE"; then
        print_success "Proxy timeouts configured (verified)"
    else
        print_warning "Proxy timeouts not configured"
    fi
    
    return 0
}

show_summary() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}Configuration Summary:${NC}"
    echo "  • Max upload size:        5 GB"
    echo "  • Rate limiting zone:     api:10m (10 req/s)"
    echo "  • Proxy read timeout:     600s"
    echo "  • Proxy send timeout:     600s"
    echo "  • SSL certificates:       ${SSL_CERT_DIR}"
    echo "  • Site config:            ${SPEECH_SITE}"
    echo "  • Flask app port:         ${FLASK_PORT}"
    echo ""
    
    if [[ "${USE_HTTPS:-true}" == "true" ]]; then
        echo -e "${GREEN}Access URLs:${NC}"
        echo "  • HTTPS:  https://transcriber.cloud.cci.charite.de/"
        echo "  • HTTP:   http://localhost:5001/ (direct to Flask)"
    else
        echo -e "${YELLOW}HTTP-ONLY MODE (no encryption):${NC}"
        echo "  • Access: http://localhost:5001/"
    fi
    
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Start your Flask app:"
    echo "     cd ${ATA_PROJECT_ROOT}/interactive_app"
    echo "     python \"app (2).py\""
    echo ""
    echo "  2. Test locally:"
    echo "     curl -k https://localhost/health"
    echo ""
    echo "  3. Ask IT to configure DNS (if using domain):"
    echo "     transcriber.cloud.cci.charite.de → $(hostname -I 2>/dev/null | head -1)"
    echo ""
    echo -e "${GREEN}✅ Ready to test audio uploads!${NC}"
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
    
    # Pre-flight checks
    check_root
    check_nginx
    create_backup_directory
    
    # Check SSL certificates first (fail early if missing)
    check_ssl_certificates
    
    # Backup current configs
    backup_site_config
    backup_main_conf
    
    # Fix any malformed configs
    fix_malformed_site_configs
    
    # Main operations
    set +e  # Temporarily disable exit-on-error for controlled error handling
    
    create_site_config
    ADD_STATUS=$?
    
    add_to_main_conf
    MAIN_STATUS=$?
    
    set -e  # Re-enable exit-on-error
    
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

# Run main function
main "$@"