#!/bin/bash
# Activation script for the scene-change-detection virtual environment

echo "🚀 Activating scene-change-detection virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Installed packages:"
pip list | grep -E "(opencv|numpy|pytest)"

echo ""
echo "🔧 Quick commands:"
echo "  Run tests:    python -m pytest tests/ -v"
echo "  Run examples: python examples/video_reader_usage.py"
echo "  Import test:  python -c 'from utils.video_reader import VideoReader; print(\"✅ Import successful\")'"
echo ""
echo "📚 To deactivate: deactivate" 