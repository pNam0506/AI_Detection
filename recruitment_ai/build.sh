#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Create a directory for custom binaries if it doesn't exist
# and install system dependencies like poppler-utils
echo "Installing Poppler utils..."
apt-get update && apt-get install -y poppler-utils

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Collect Static Files
python manage.py collectstatic --noinput