#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  python3 \
  python3-venv \
  python3-pip \
  nginx \
  ca-certificates \
  ffmpeg \
  libglib2.0-0 \
  libgomp1

rm -rf /var/lib/apt/lists/*

# Python venv
cd "$APP_DIR"
python3 - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) >= (3, 12):
    print("WARNING: Detected Python %d.%d. TensorFlow wheels may not be available for this setup." % (major, minor))
    print("WARNING: For easiest installs on OCI Always Free (ARM64), prefer Ubuntu 22.04 (Python 3.10).")
PY
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Nginx reverse proxy
install -m 0644 "$APP_DIR/deploy/oci/nginx.conf" /etc/nginx/sites-available/deepshield
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/deepshield /etc/nginx/sites-enabled/deepshield
nginx -t
systemctl restart nginx

# systemd service
install -m 0644 "$APP_DIR/deploy/oci/deepshield.service" /etc/systemd/system/deepshield.service
systemctl daemon-reload

echo "Install complete. Start with: sudo systemctl enable --now deepshield"
