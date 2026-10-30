#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/backend"
[ -d ../.venv ] || python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -q -r requirements.txt
echo "Open http://localhost:8000"
uvicorn app.main:app --host 127.0.0.1 --port 8000
