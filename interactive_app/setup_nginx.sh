#!/bin/bash
#===============================================================================
# NGINX Complete Setup Script - Fixed Version
# Handles malformed site configs + adds 5GB upload support
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# CONFIGURATION
#-------------------------------------------------------------------------------

NGINX_CONF="/etc/nginx/nginx.conf"
SPEECH_SITE="/etc/nginx/sites-enabled/speech-anonymizer"
SPEECH_SITE_BACKUP="/etc/nginx/backups/speech-anonymizer.fixed.$(date +%Y%m%d_%H%M%S).old"
MAIN_CONF_BACKUP="/etc/nginx/backups/nginx.conf.main.$(date +%Y%m%d_%H%M%S).old"
MAX_BODY_SIZE="5G"
PROXY_TIMEOUT="600s"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

#-------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#-------------------------------------------------------------------------------

print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "Must run as root: sudo $0"
        exit 1
    fi
}

check_nginx() {
    if ! command -v nginx &> /dev/null; then
        print_error "Nginx not installed"
        exit 1
    fi
}

ensure_backup_dir() {
    mkdir -p /etc/nginx/backups
}

#-------------------------------------------------------------------------------
# FIX SITES ENABLED CONFIG (The Critical Bug Fix)
#-------------------------------------------------------------------------------

fix_speech_anonymizer_config() {
    print_info "Checking speech-anonymizer site config..."
    
    if [[ ! -f "$SPEECH_SITE" ]]; then
        print_info "Site config not found - nothing to fix"
        return 0
    fi
    
    # Check for invalid directives
    if grep -q "^events {" "$SPEECH_SITE" || grep -q "^http {" "$SPEECH_SITE"; then
        print_warning "Found invalid 'events' or 'http' blocks - this must be fixed"
        
        # Backup current broken config
        cp "$SPEECH_SITE" "$SPEECH_SITE_BACKUP"
        print_success "Backed up broken config to: $SPEECH_SITE_BACKUP"
        
        print_info "Replacing with corrected configuration..."
        
        cat > "$SPEECH_SITE" << 'EOF'
# ============================================================================
# Nginx Reverse Proxy Configuration for Speech Anonymizer
# CORRECTED: Removed events {} and http {} blocks
# ============================================================================

server_tokens off;

limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

upstream speech_anonymizer {
    server 127.0.0.1:5001;
    keepalive 32;
}

# HTTP Server - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name _;
    return 301 https://$server_name$request_uri;
}

# HTTPS Server - Main Application
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name transcriber.cloud.cci.charite.de;

    # SSL Configuration
    ssl_certificate /app/certs/server.crt;
    ssl_certificate_key /etc/nginx/backups/speech-anonymizer.fixed.$(date +%Y%m%d_%H%M%S).old;
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
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://speech_anonymizer;
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

    location /health {
        proxy_pass http://speech_anonymizer/health;
        access_log off;
    }

    location /static/ {
        alias /mnt/Data_Mount/VERANDA_DataMount/ATA/interactive_app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        try_files $uri $uri/ =404;
    }
}
EOF
        
        # Note: You'll need to update the SSL certificate paths below if different
        sed -i "s|ssl_certificate_key /etc/nginx/backups/speech-anonymizer.fixed.*|ssl_certificate_key /app/certs/server.key;|" "$SPEECH_SITE"
        
        print_success "Site config replaced successfully"
    else
        print_info "Site config already valid - no changes needed"
    fi
}

#-------------------------------------------------------------------------------
# ADD TO MAIN NGINX.CONF (Optional - for global setting)
#-------------------------------------------------------------------------------

add_client_max_to_main_conf() {
    print_info "Checking main nginx.conf..."
    
    if grep -q "client_max_body_size $MAX_BODY_SIZE;" "$NGINX_CONF"; then
        print_info "Already configured in main conf"
        return 0
    fi
    
    print_info "Adding client_max_body_size to main config..."
    
    # Create temp file
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
        cp "$NGINX_CONF" "$MAIN_CONF_BACKUP"
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

test_and_reload() {
    print_info "Testing nginx configuration..."
    
    if nginx -t 2>&1 | tee /tmp/nginx_test.log; then
        print_success "Configuration test passed"
        
        if systemctl is-active --quiet nginx 2>/dev/null; then
            print_info "Reloading nginx..."
            systemctl reload nginx
            print_success "Nginx reloaded"
        else
            print_warning "Nginx not running - start manually with: sudo systemctl start nginx"
        fi
    else
        print_error "Configuration test FAILED!"
        echo ""
        echo "Error details:"
        cat /tmp/nginx_test.log | tail -20
        echo ""
        print_info "Restoring from backup..."
        
        if [[ -f "$SPEECH_SITE_BACKUP" ]]; then
            cp "$SPEECH_SITE_BACKUP" "$SPEECH_SITE"
            print_info "Site config restored"
        fi
        
        if [[ -f "$MAIN_CONF_BACKUP" ]]; then
            cp "$MAIN_CONF_BACKUP" "$NGINX_CONF"
            print_info "Main config restored"
        fi
        
        exit 1
    fi
}

show_summary() {
    echo ""
    print_success "=========================================="
    print_success "  NGINX Configuration Complete!"
    print_success "=========================================="
    echo ""
    echo -e "${GREEN}Applied Changes:${NC}"
    echo "  • Max upload size:      $MAX_BODY_SIZE"
    echo "  • Proxy timeout:        $PROXY_TIMEOUT"
    echo "  • Site config fixed:    Yes"
    echo ""
    echo -e "${YELLOW}Verification:${NC}"
    echo "  # Check config:"
    echo "  grep 'client_max_body_size' /etc/nginx/sites-enabled/*"
    echo ""
    echo "  # Test syntax:"
    echo "  sudo nginx -t"
    echo ""
    echo "  # Check nginx status:"
    echo "  sudo systemctl status nginx"
    echo ""
    echo -e "${GREEN}Ready to test large audio uploads!${NC}"
    echo ""
}

#-------------------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------------------

main() {
    print_header "Nginx Complete Setup (Fixed)"
    
    check_root
    check_nginx
    ensure_backup_dir
    
    fix_speech_anonymizer_config
    add_client_max_to_main_conf
    test_and_reload
    show_summary
}

print_header() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

main "$@"