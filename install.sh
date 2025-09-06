#!/bin/bash
# Installation script for algoscope

echo "🚀 Installing algoscope..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
python3 -m pip install -r requirements.txt

# Install algoscope in editable mode
echo "📦 Installing algoscope in editable mode..."
python3 -m pip install -e .

echo "✅ Installation complete!"
echo ""
echo "To test the installation, run:"
echo "  python examples/compare_linear_vs_binary.py"
echo ""
echo "For development dependencies, run:"
echo "  pip install -r requirements-dev.txt"
