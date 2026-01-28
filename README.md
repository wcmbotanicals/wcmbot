---
title: Wcmbot
emoji: 🌖
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
license: gpl-3.0
short_description: It's the WCMBot
---

# 🧩 WCMBot

A Gradio web application that helps solve jigsaw puzzles by identifying where individual pieces fit in a template image using computer vision techniques.

## Features

- **Upload puzzle piece images** and automatically find their position in the template
- **Automatic tab/knob inference** (no manual input required)
- **Multi-match visualization** with the same diagnostic subplots as the CLI workflow
- **Zoomable/pannable plots** using Plotly for interactive exploration of match results
- **Confidence scoring** for match quality
- **Grid overlay on template** showing row and column numbers (enabled by default) with customizable display
- **Multipiece mode (batch)**: upload a photo containing multiple pieces and solve them in sequence
- **Export template grids** to high-quality PNG images with configurable DPI and rotation support
- **Template rotation** in 90° increments for different puzzle orientations
- **Interactive Gradio interface** with modern UI/UX
- **HuggingFace Spaces ready**
- **Fully tested** with Playwright E2E tests

## How It Works

The application uses an improved pipeline inspired by `1.py`:

1. **Blue mask segmentation**: Uses HSV thresholds and morphological cleanup to extract the piece
2. **Knob-aware scaling**: Estimates scale using puzzle grid cell sizes and inferred knob/tab counts
3. **Binary template matching**: Runs multi-scale, multi-rotation correlation on binary patterns with top-N ranking
4. **Interactive diagnostics**: Displays multi-panel plots directly in the web UI with next/previous navigation

## Installation

### Prerequisites

- Python 3.10 or higher
- Either:
	- `uv` (recommended), or
	- `pip`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/wcmbotanicals/wcmbot.git
cd wcmbot
```

2. Install dependencies:
```bash
# Recommended
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

3. Install Playwright browsers (for testing):
```bash
playwright install
```

## Usage

### Running the Gradio App

Simply run:
```bash
uv run python app.py
```

Or, if you installed with pip:
```bash
python app.py
```

The app will launch the Gradio interface in your browser.

Optional flags:
```bash
# Make accessible on your LAN (binds 0.0.0.0)
uv run python app.py --accessible

# Enable Torch acceleration (uses MPS/CUDA if available)
uv run python app.py --gpu
```

### Using the Interface

1. **View the template** - The puzzle template is displayed on the right side with a grid overlay showing row and column numbers
2. **Toggle grid** - Enable/disable the grid overlay using the "Show Grid" checkbox in the Settings accordion
3. **Upload a piece** - Click the upload area or drag and drop a puzzle piece image
4. **Find the match** - Matching runs automatically after upload (or click "Find Piece Location")
5. **View results** - See the highlighted position on the template with confidence score
6. **Explore matches** - Use zoomable/pannable Plotly views to inspect match details and navigate between top matches
7. **Rotate template** - Use the template rotation control to view the puzzle in different orientations (90° increments)

Multipiece mode:
- Enable "Multipiece mode (batch)" to detect multiple pieces in one upload and solve them in sequence.
- Use the per-piece "Next candidate" controls to cycle candidate placements per detected piece.

### Exporting Template Grids

Use the `export_template_grid.py` script to export high-quality PNG images of puzzle templates with grids:

```bash
# Export a template at default DPI (150) with default rotation
python export_template_grid.py sample_puzzle

# Export with custom DPI
python export_template_grid.py sample_puzzle --dpi 100

# Export with rotation (0, 90, 180, or 270 degrees)
python export_template_grid.py sample_puzzle --rotation 90

# Export to a custom location
python export_template_grid.py sample_puzzle --output ~/my_templates/

# List all available templates
python export_template_grid.py --list
```

The script automatically calculates the physical dimensions based on the `export_width_cm` parameter in your template configuration.

### HuggingFace Spaces

To deploy to HuggingFace Spaces:

1. Create a new Space on HuggingFace
2. Push this repository to the Space
3. The `app.py` file will automatically be detected and run

## Testing

The project includes comprehensive E2E test coverage using Playwright:

### Run E2E Tests

```bash
pytest test_gradio.py -v
```

### Run All Tests

```bash
pytest -v
```

## Project Structure

```
wcmbot/
├── app.py                         # Gradio interface
├── export_template_grid.py        # Export templates with grids as PNG
├── media/
│   ├── templates/                 # Template images + registry
│   │   ├── templates.json
│   │   ├── sample_puzzle.png
│   │   └── grass_puzzle.png
│   └── pieces/                    # Sample piece images
│       ├── sample_puzzle/
│       └── grass_puzzle/
├── wcmbot/
│   ├── matcher.py                 # Matching pipeline
│   ├── solving.py                 # Reusable solve helpers
│   ├── multipiece.py              # Multipiece detection
│   ├── template_settings.py       # Template registry loader
│   └── viz.py                     # Rendering helpers
└── tests/                         # Pytest + Playwright tests
```

## Technology Stack

- **Gradio**: Web interface framework
- **OpenCV**: Computer vision and image matching
- **Pillow**: Image processing
- **NumPy**: Numerical operations
- **Plotly**: Interactive zoomable/pannable visualizations
- **pytest**: Testing framework
- **Playwright**: Browser automation for E2E tests

## How the Matching Works

The matcher mirrors the performant notebook script:

1. **Color segmentation**: Two HSV ranges isolate blue plastic, followed by morphological cleanup and largest-component filtering
2. **Knob-aware scaling**: Estimates the correct template scale from grid cell dimensions and knob counts
3. **Binary correlation**: Runs multi-scale, multi-rotation normalized cross-correlation on blurred binary masks with aggressive duplicate suppression
4. **Top-N ranking**: Keeps several high-confidence candidates, attaches contours, and prepares data for plotting

### Configuration

The matcher can be configured via these constants in `matcher.py`:
- `COLS`, `ROWS`: Grid dimensions (36x28)
- `EST_SCALE_WINDOW`: Scale factors to test around the knob-aware estimate
- `ROTATIONS`: Rotation angles to test
- `TOP_MATCH_COUNT`: How many best matches to keep for review
- `KNOB_WIDTH_FRAC`: Contribution of each knob to the estimated full width/height

### Built-In Visual Debugging

The Gradio app renders the same diagnostic panels that the offline script produced (template, masks, zoom views, etc.) and lets you cycle through the ranked matches. There's no environment toggle needed—just upload a piece and use the on-screen arrows.

## Deployment

### Local Deployment

```bash
python app.py
```

### Docker (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t jigsaw-puzzle .
docker run -p 7860:7860 jigsaw-puzzle
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

See LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- Gradio for the interactive interface
- HuggingFace for deployment platform
