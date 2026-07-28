#!/bin/bash
# setup_nginx.sh - Configure nginx for large audio file uploads (5GB)

set -e  # Exit on error

echo "=============================================="
echo "  Nginx Configuration Setup for Speech App"
echo "=============================================="
echo ""

# Configuration
NGINX_CONF="/etc/nginx/nginx.conf"
BACKUP_FILE="${NGINX_CONF}.backup.$(date +%Y%m%d_%H%M%S)"
MAX_BODY_SIZE="5G"
PROXY_TIMEOUT="600s"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}[ERROR]${NC} This script must be run as root!"
   echo "Use: sudo ./setup_nginx.sh"
   exit 1
fi

echo -e "${YELLOW}[INFO]${NC} Checking nginx installation..."
if ! command -v nginx &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Nginx is not installed!"
    echo "Install with: sudo apt install nginx"
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Nginx found"

# Backup current config
echo ""
echo -e "${YELLOW}[INFO]${NC} Creating backup of current nginx config..."
if [[ -f "$NGINX_CONF" ]]; then
    cp "$NGINX_CONF" "$BACKUP_FILE"
    echo -e "${GREEN}[OK]${NC} Backup saved to: $BACKUP_FILE"
else
    echo -e "${RED}[ERROR]${NC} $NGINX_CONF not found!"
    exit 1
fi

# Check if already configured
if grep -q "client_max_body_size $MAX_BODY_SIZE;" "$NGINX_CONF"; then
    echo ""
    echo -e "${GREEN}[INFO]${NC} Configuration already set to $MAX_BODY_SIZE!"
    echo -e "${YELLOW}[SKIP]${NC} No changes needed."
else
    echo ""
    echo -e "${YELLOW}[INFO]${NC} Adding client_max_body_size $MAX_BODY_SIZE to http block..."
    
    # Create temp file for sed operation
    TMPFILE=$(mktemp)
    
    # Insert after the line containing "http {"
    sed "/^http {$/a\\
\\
\t# Large file upload support for Speech Anonymizer\\
\tclient_max_body_size $MAX_BODY_SIZE;" "$NGINX_CONF" > "$TMPFILE"
    
    if [ $? -eq 0 ]; then
        mv "$TMPFILE" "$NGINX_CONF"
        echo -e "${GREEN}[OK]${NC} Configuration updated successfully"
    else
        rm "$TMPFILE"
        echo -e "${RED}[ERROR]${NC} Failed to update configuration!"
        echo -e "${YELLOW}[INFO]${NC} Restoring backup..."
        cp "$BACKUP_FILE" "$NGINX_CONF"
        exit 1
    fi
fi

# Also update site-specific configs for proxy timeouts
echo ""
echo -e "${YELLOW}[INFO]${NC} Checking for site-specific proxy configurations..."
SITES_DIR="/etc/nginx/sites-enabled"
SITE_CONFIG="$SITES_DIR/default"

if [[ -f "$SITE_CONFIG" ]]; then
    # Add proxy timeouts if not present
    if ! grep -q "proxy_read_timeout" "$SITE_CONFIG"; then
        echo -e "${YELLOW}[INFO]${NC} Adding proxy timeout settings to site config..."
        
        # Find the location / block and add proxy settings
        TMPFILE=$(mktemp)
        awk -v timeout="$PROXY_TIMEOUT" '
        /^location \/ {/ {
            print $0
            getline
            print $0
            print "\t# Large upload timeout settings"
            print "\tproxy_read_timeout " timeout ";"
            print "\tproxy_send_timeout " timeout ";"
            print "\tproxy_connect_timeout 60s;"
            next
        }
        {print}
        ' "$SITE_CONFIG" > "$TMPFILE"
        
        mv "$TMPFILE" "$SITE_CONFIG"
        echo -e "${GREEN}[OK]${NC} Proxy timeouts added"
    else
        echo -e "${GREEN}[OK]${NC} Proxy settings already configured"
    fi
else
    echo -e "${YELLOW}[WARN]${NC} Site config not found at $SITE_CONFIG"
    echo -e "${YELLOW}[INFO]${NC} You may need to manually configure proxy settings"
fi

# Test nginx configuration
echo ""
echo -e "${YELLOW}[INFO]${NC} Testing nginx configuration..."
if nginx -t 2>&1 | grep -q "syntax is ok"; then
    echo -e "${GREEN}[OK]${NC} Configuration syntax valid"
else
    echo -e "${RED}[ERROR]${NC} Configuration has errors!"
    echo -e "${YELLOW}[INFO]${NC} Restoring backup..."
    cp "$BACKUP_FILE" "$NGINX_CONF"
    exit 1
fi

# Reload nginx
echo ""
echo -e "${YELLOW}[INFO]${NC} Reloading nginx service..."
if systemctl reload nginx; then
    echo -e "${GREEN}[OK]${NC} Nginx reloaded successfully"
else
    echo -e "${YELLOW}[WARN]${NC} Could not reload nginx (service may not be running)"
    echo -e "${YELLOW}[INFO]${NC} Start/restart with: sudo systemctl start nginx"
fi

# Summary
echo ""
echo "=============================================="
echo "  ✅ Setup Complete!"
echo "=============================================="
echo ""
echo -e "${GREEN}Configuration Summary:${NC}"
echo "  • Max upload size: $MAX_BODY_SIZE"
echo "  • Proxy timeout: $PROXY_TIMEOUT"
echo "  • Backup location: $BACKUP_FILE"
echo ""
echo -e "${YELLOW}To revert changes:${NC}"
echo "  sudo cp '$BACKUP_FILE' '$NGINX_CONF'"
echo "  sudo systemctl reload nginx"
echo ""
echo -e "${YELLOW}To verify:${NC}"
echo "  grep 'client_max_body_size' '$NGINX_CONF'"
echo ""