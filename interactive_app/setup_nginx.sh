#!/bin/bash
#===============================================================================
# NGINX Configuration Setup for Speech Anonymizer App
# Purpose: Enable large audio file uploads (up to 5GB) with proper timeout settings
# Author: Setup Script v2.0 (Fixed & Production Ready)
#===============================================================================

set -e  # Exit immediately on any error

#-------------------------------------------------------------------------------
# CONFIGURATION
#-------------------------------------------------------------------------------

NGINX_CONF="/etc/nginx/nginx.conf"
SITES_DIR="/etc/nginx/sites-enabled"
SITE_CONFIG="$SITES_DIR/default"
BACKUP_DIR="/etc/nginx/backups"
MAX_BODY_SIZE="5G"
PROXY_READ_TIMEOUT="600s"
PROXY_SEND_TIMEOUT="600s"
PROXY_CONNECT_TIMEOUT="60s"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root!"
        echo "Use: sudo ./setup_nginx_complete.sh"
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

#-------------------------------------------------------------------------------
# BACKUP FUNCTIONS
#-------------------------------------------------------------------------------

create_backup_directory() {
    if [[ ! -d "$BACKUP_DIR" ]]; then
        mkdir -p "$BACKUP_DIR"
        print_info "Created backup directory: $BACKUP_DIR"
    fi
}

create_backup() {
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="${BACKUP_DIR}/nginx.conf.${TIMESTAMP}"
    
    if [[ -f "$NGINX_CONF" ]]; then
        cp "$NGINX_CONF" "$BACKUP_FILE"
        print_success "Backup saved to: $BACKUP_FILE"
    else
        print_error "$NGINX_CONF not found!"
        exit 1
    fi
}

restore_backup() {
    if [[ -n "$BACKUP_FILE" && -f "$BACKUP_FILE" ]]; then
        print_info "Restoring backup..."
        cp "$BACKUP_FILE" "$NGINX_CONF"
        print_success "Backup restored successfully"
    fi
}

#-------------------------------------------------------------------------------
# MAIN SETUP FUNCTIONS
#-------------------------------------------------------------------------------

add_client_max_body_size() {
    print_info "Checking current client_max_body_size setting..."
    
    if grep -q "client_max_body_size $MAX_BODY_SIZE;" "$NGINX_CONF"; then
        print_info "Configuration already set to $MAX_BODY_SIZE - no changes needed"
        return 0
    fi
    
    if grep -q "client_max_body_size" "$NGINX_CONF"; then
        print_warning "Different client_max_body_size value already exists"
        grep "client_max_body_size" "$NGINX_CONF" | head -1
        read -p "Continue anyway and update to $MAX_BODY_SIZE? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Aborted by user"
            return 1
        fi
    fi
    
    print_info "Adding client_max_body_size $MAX_BODY_SIZE to http block..."
    
    # Create temp file
    TMPFILE=$(mktemp)
    
    # Use awk to insert with proper TAB indentation (critical for nginx)
    awk -v size="$MAX_BODY_SIZE" '
    /^http \{/ {
        print
        print ""
        printf "\t# Large file upload support for Speech Anonymizer\n"
        printf "\tclient_max_body_size %s;\n", size
        next
    }
    {print}
    ' "$NGINX_CONF" > "$TMPFILE"
    
    # Check if awk succeeded
    if [ $? -ne 0 ]; then
        print_error "Failed to modify configuration!"
        rm -f "$TMPFILE"
        return 1
    fi
    
    # Backup the modified config before replacing
    mv "$TMPFILE" "$NGINX_CONF"
    print_success "Added client_max_body_size directive"
    
    # Verify it was added correctly
    if grep -q "client_max_body_size $MAX_BODY_SIZE;" "$NGINX_CONF"; then
        print_success "Directive verified in config"
    else
        print_error "Verification failed - directive not found!"
        return 1
    fi
    
    return 0
}

add_proxy_settings() {
    print_info "Checking for proxy timeout settings..."
    
    # Check if site config exists
    if [[ ! -f "$SITE_CONFIG" ]]; then
        print_warning "Site config not found at $SITE_CONFIG"
        print_info "You may need to manually configure proxy settings"
        return 0
    fi
    
    # Check if already configured
    if grep -q "proxy_read_timeout" "$SITE_CONFIG"; then
        print_success "Proxy settings already configured"
        return 0
    fi
    
    print_info "Adding proxy timeout settings to site config..."
    
    TMPFILE=$(mktemp)
    
    # Use awk to add proxy settings inside location / block
    awk -v read_timeout="$PROXY_READ_TIMEOUT" \
        -v send_timeout="$PROXY_SEND_TIMEOUT" \
        -v connect_timeout="$PROXY_CONNECT_TIMEOUT" '
    /^location \/ \{/ {
        print
        getline  # Read opening brace line (or first content line)
        print
        print "\t# Large upload timeout settings for Speech Anonymizer"
        printf "\tproxy_read_timeout %s;\n", read_timeout
        printf "\tproxy_send_timeout %s;\n", send_timeout
        printf "\tproxy_connect_timeout %s;\n", connect_timeout
        next
    }
    {print}
    ' "$SITE_CONFIG" > "$TMPFILE"
    
    if [ $? -ne 0 ]; then
        print_error "Failed to add proxy settings!"
        rm -f "$TMPFILE"
        return 1
    fi
    
    mv "$TMPFILE" "$SITE_CONFIG"
    print_success "Added proxy timeout settings"
    
    return 0
}

test_configuration() {
    print_info "Testing nginx configuration syntax..."
    
    if nginx -t 2>/dev/null; then
        print_success "Configuration syntax is valid"
        return 0
    else
        print_error "Configuration has syntax errors!"
        echo ""
        print_info "Detailed error:"
        nginx -t 2>&1 | tail -20
        echo ""
        return 1
    fi
}

reload_nginx() {
    print_info "Reloading nginx service..."
    
    # Check if nginx service is active
    if systemctl is-active --quiet nginx 2>/dev/null || pgrep nginx >/dev/null; then
        if systemctl reload nginx 2>/dev/null; then
            print_success "Nginx reloaded successfully"
        else
            print_warning "Could not reload via systemctl (trying signal method)..."
            if nginx -s reload 2>/dev/null; then
                print_success "Nginx reloaded via signal"
            else
                print_warning "Manual nginx reload may be required"
                return 1
            fi
        fi
    else
        print_warning "Nginx service not running - starting it..."
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
# VERIFICATION FUNCTIONS
#-------------------------------------------------------------------------------

verify_changes() {
    print_info "Verifying applied changes..."
    
    # Check client_max_body_size
    if grep -q "client_max_body_size $MAX_BODY_SIZE;" "$NGINX_CONF"; then
        print_success "client_max_body_size = $MAX_BODY_SIZE (verified)"
    else
        print_error "client_max_body_size not found!"
        return 1
    fi
    
    # Check proxy settings if site config exists
    if [[ -f "$SITE_CONFIG" ]]; then
        if grep -q "proxy_read_timeout" "$SITE_CONFIG"; then
            print_success "Proxy timeouts configured (verified)"
        else
            print_warning "Proxy timeouts not configured"
        fi
    fi
    
    return 0
}

show_summary() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}Configuration Summary:${NC}"
    echo "  • Max upload size:        $MAX_BODY_SIZE"
    echo "  • Proxy read timeout:     $PROXY_READ_TIMEOUT"
    echo "  • Proxy send timeout:     $PROXY_SEND_TIMEOUT"
    echo "  • Proxy connect timeout:  $PROXY_CONNECT_TIMEOUT"
    echo "  • Config file:            $NGINX_CONF"
    echo "  • Site config:            $SITE_CONFIG"
    echo "  • Backup directory:       $BACKUP_DIR"
    echo ""
    
    echo -e "${YELLOW}Rollback Instructions:${NC}"
    echo "  1. Find latest backup: ls $BACKUP_DIR"
    echo "  2. Restore: sudo cp \$BACKUP_FILE $NGINX_CONF"
    echo "  3. Reload: sudo nginx -s reload"
    echo ""
    
    echo -e "${YELLOW}Verification Commands:${NC}"
    echo "  # Check max body size:"
    echo "  grep 'client_max_body_size' $NGINX_CONF"
    echo ""
    echo "  # Test syntax:"
    echo "  sudo nginx -t"
    echo ""
    echo "  # Reload nginx:"
    echo "  sudo nginx -s reload"
    echo ""
    echo -e "${GREEN}✅ All checks passed! Ready to test audio uploads.${NC}"
    echo ""
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
    create_backup
    
    # Main operations (with error recovery)
    set +e  # Temporarily disable exit-on-error for controlled error handling
    
    add_client_max_body_size
    ADD_STATUS=$?
    
    add_proxy_settings
    PROXY_STATUS=$?
    
    test_configuration
    TEST_STATUS=$?
    
    set -e  # Re-enable exit-on-error
    
    # Evaluate results
    if [[ $ADD_STATUS -eq 0 && $TEST_STATUS -eq 0 ]]; then
        reload_nginx
        verify_changes
        show_summary
    else
        print_error "Setup failed! Attempting rollback..."
        restore_backup
        
        if test_configuration; then
            print_success "Rollback successful - original configuration restored"
        else
            print_error "Rollback failed! Manual intervention required!"
            print_info "Last known good backup: $BACKUP_FILE"
            exit 1
        fi
        exit 1
    fi
}

# Run main function
main "$@"