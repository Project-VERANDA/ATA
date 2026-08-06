#!/bin/bash
# scripts/create-docker-nginx-config.sh

set -e

echo "Creating Docker nginx config from production..."

# Create directory structure
mkdir -p docker/nginx/sites-available

# Copy your production config and adapt it
sudo cp /etc/nginx/sites-enabled/speech-anonymizer docker/nginx/sites-available/speech-anonymizer

# Apply Docker adaptations
echo "Applying Docker adaptations..."

# 1. Change backend from 127.0.0.1:5001 to backend:5001
sudo sed -i 's/server 127\.0\.0\.1:5001/server backend:5001/g' docker/nginx/sites-available/speech-anonymizer

# 2. Change SSL certs from Let's Encrypt to org certs
sudo sed -i 's|ssl_certificate /etc/letsencrypt/.*|ssl_certificate /etc/nginx/ssl/server.crt;|g' \
     docker/nginx/sites-available/speech-anonymizer
sudo sed -i 's|ssl_certificate_key /etc/letsencrypt/.*|ssl_certificate_key /etc/nginx/ssl/server.key;|g' \
     docker/nginx/sites-available/speech-anonymizer

# 3. Remove Certbot comments (no longer relevant in Docker)
sudo sed -i '/# managed by Certbot/d' docker/nginx/sites-available/speech-anonymizer

# 4. Remove Certbot redirect block (we manage redirects ourselves)
sudo sed -i '/if (\$host = transcriber.cloud.cci.charite.de)/,/return 301/d' docker/nginx/sites-available/speech-anonymizer

# 5. Change static files path from host mount to Docker mount
sudo sed -i 's|alias /mnt/Data_Mount/VERANDA_DataMount/ATA/interactive_app/static/|alias /var/www/static/|g' \
     docker/nginx/sites-available/speech-anonymizer

# 6. Add sensitive path blocking
cat >> docker/nginx/sites-available/speech-anonymizer << 'EOF'

    # ===================================================
    # BLOCK SENSITIVE PATHS (Added for Docker security)
    # ===================================================
    location ~ /pipeline/ {
        deny all;
        access_log off;
    }

    location ~ /model/ {
        deny all;
        access_log off;
    }
}
EOF

echo "✅ Docker nginx config created!"
echo ""
echo "Location: $(pwd)/docker/nginx/sites-available/speech-anonymizer"
echo ""
echo "Verify the config:"
echo "  cat docker/nginx/sites-available/speech-anonymizer | grep 'server backend'"
echo "  cat docker/nginx/sites-available/speech-anonymizer | grep 'ssl_certificate'"
echo ""
