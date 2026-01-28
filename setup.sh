#!/bin/bash
# Setup script for the jigsaw puzzle solver

set -e

echo "🧩 Setting up Jigsaw Puzzle Solver..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "  brew install uv"
    exit 1
fi

# Install dependencies with uv (include dev extras for tests)
echo "📦 Installing dependencies with uv..."
uv sync --extra dev

# Install Playwright browsers (chromium only for E2E tests)
echo "🎭 Installing Playwright browsers..."
uv run playwright install chromium

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the Gradio interface, run:"
echo "  uv run python app.py"
echo ""
echo "To export template grids to PNG files, run:"
echo "  uv run python export_template_grid.py --help"
echo ""
echo "Examples:"
echo "  uv run python export_template_grid.py sample_puzzle"
echo "  uv run python export_template_grid.py grass_puzzle --dpi 150 --rotation 90"
echo ""
echo "To run tests:"
echo "  uv run python -m pytest -v"
echo ""
