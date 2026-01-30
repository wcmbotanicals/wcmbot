#!/bin/bash
# Setup script for WCMBot

set -e

echo "🧩 Setting up WCMBot..."

# Install dependencies (prefer uv, fall back to pip)
if command -v uv &> /dev/null; then
    echo "📦 Installing dependencies with uv..."
    uv sync --extra dev
    RUN_PREFIX="uv run"
else
    echo "⚠️  uv not found; falling back to pip. (Install uv with: brew install uv)"
    echo "📦 Installing dependencies with pip..."
    python -m pip install -e ".[dev]"
    RUN_PREFIX=""
fi

# Install Playwright browsers (chromium only for E2E tests)
echo "🎭 Installing Playwright browsers..."
if [ -n "$RUN_PREFIX" ]; then
    $RUN_PREFIX playwright install chromium
else
    playwright install chromium
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the Gradio interface, run:"
if [ -n "$RUN_PREFIX" ]; then
    echo "  uv run python app.py"
else
    echo "  python app.py"
fi
echo ""
echo "To export template grids to PNG files, run:"
if [ -n "$RUN_PREFIX" ]; then
    echo "  uv run python export_template_grid.py --help"
else
    echo "  python export_template_grid.py --help"
fi
echo ""
echo "Examples:"
if [ -n "$RUN_PREFIX" ]; then
    echo "  uv run python export_template_grid.py sample_puzzle"
    echo "  uv run python export_template_grid.py grass_puzzle --dpi 150 --rotation 90"
else
    echo "  python export_template_grid.py sample_puzzle"
    echo "  python export_template_grid.py grass_puzzle --dpi 150 --rotation 90"
fi
echo ""
echo "To run tests:"
if [ -n "$RUN_PREFIX" ]; then
    echo "  uv run python -m pytest -v"
else
    echo "  python -m pytest -v"
fi
echo ""
