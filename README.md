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

# 🧩 Jigsaw Puzzle Solver

A Gradio web application that helps solve jigsaw puzzles by identifying where individual pieces fit in a template image using computer vision techniques.

## Features

- **Upload puzzle piece images** and automatically find their position in the template
- **Tab count configuration** with horizontal and vertical tab inputs for accurate scale estimation
- **Multi-match visualization** with the same diagnostic subplots as the CLI workflow
- **Zoomable/pannable plots** using Plotly for interactive exploration of match results
- **Confidence scoring** for match quality
- **Interactive Gradio interface** with modern UI/UX
- **HuggingFace Spaces ready**
- **Fully tested** with Playwright E2E tests

## How It Works

The application uses an improved pipeline inspired by `1.py`:

1. **Blue mask segmentation**: Uses HSV thresholds and morphological cleanup to extract the piece
2. **Knob-aware scaling**: Estimates scale using puzzle grid cell sizes and knob counts
3. **Binary template matching**: Runs multi-scale, multi-rotation correlation on binary patterns with top-N ranking
4. **Interactive diagnostics**: Displays the multi-panel plots from `show_debug_all()` directly in the web UI with next/previous navigation

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/wcmbotanicals/wcmbot.git
cd wcmbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Playwright browsers (for testing):
```bash
playwright install
```

## Usage

### Running the Gradio App

Simply run:
```bash
python app.py
```

The app will:
- Automatically create a default puzzle template if none exists
- Launch the Gradio interface in your browser
- Display the puzzle template and allow you to upload piece images

### Using the Interface

1. **View the template** - The puzzle template is displayed on the right side
2. **Upload a piece** - Click the upload area or drag and drop a puzzle piece image
3. **Set tab counts** - Enter the number of horizontal and vertical tabs (0-2) on your puzzle piece for accurate scale estimation
4. **Find the match** - Click "Find Piece Location" button
5. **View results** - See the highlighted position on the template with confidence score
6. **Explore matches** - Use zoomable/pannable Plotly views to inspect match details and navigate between top matches

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
├── app.py                   # Gradio interface
├── matcher.py               # Image matching algorithms
├── media/                   # Puzzle templates and pieces
│   ├── templates/           # Puzzle templates
│   │   └── sample_puzzle.png
│   └── pieces/              # Sample puzzle pieces
│       ├── piece_1.jpg
│       ├── piece_2.jpg
│       └── piece_3.jpg
├── test_gradio.py           # Playwright E2E tests
├── pytest.ini               # Pytest configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
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
