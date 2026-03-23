#!/bin/bash
# install.sh - Setup script for pi0-gamma-pairing

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-tk python3-dev

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing Python dependencies..."
# Install with uv (if you have it) or pip
uv pip install -e .

echo "Verifying tkinter..."
python -c "import tkinter; print('✅ tkinter installed successfully')"

echo "Setup complete!"
