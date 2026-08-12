# SSL Certificate Configuration
export DOMAIN_NAME="transcriber.cloud.cci.charite.de"
export LETSENCRYPT_FULLCHAIN_PATH="/etc/letsencrypt/live/transcriber.cloud.cci.charite.de/fullchain.pem"
export LETSENCRYPT_PRIVKEY_PATH="/etc/letsencrypt/live/transcriber.cloud.cci.charite.de/privkey.pem"

# Flask binding settings
export FLASK_BIND_HOST="0.0.0.0"
export FLASK_BIND_PORT="5001"