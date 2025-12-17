#!/bin/bash
# Setup script for the jigsaw puzzle solver

set -e

echo "🧩 Setting up Jigsaw Puzzle Solver..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Install Playwright browsers (chromium only for E2E tests)
echo "🎭 Installing Playwright browsers..."
playwright install chromium

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the Gradio interface, run:"
echo "  python app.py"
echo ""
echo "To run tests:"
echo "  pytest -v"
echo ""
