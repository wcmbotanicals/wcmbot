# 🧩 Jigsaw Puzzle Solver

A Gradio web application that helps solve jigsaw puzzles by identifying where individual pieces fit in a template image using computer vision techniques.

## Features

- **Upload puzzle piece images** and automatically find their position in the template
- **Visual highlighting** of the matched position on the template
- **Confidence scoring** for match quality
- **Interactive Gradio interface** with modern UI/UX
- **HuggingFace Spaces ready**
- **Fully tested** with Playwright E2E tests

## How It Works

The application uses OpenCV-based computer vision algorithms to match puzzle pieces:

1. **Edge Detection**: Converts template and pieces to edge maps using Canny edge detection
2. **Template Matching**: Uses normalized cross-correlation with multi-scale and multi-rotation matching
3. **Position Highlighting**: Marks the matched location with visual indicators

## Installation

### Prerequisites

- Python 3.8 or higher
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
3. **Find the match** - Click "Find Piece Location" button
4. **View results** - See the highlighted position on the template with confidence score

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
- **pytest**: Testing framework
- **Playwright**: Browser automation for E2E tests

## How the Matching Works

The matcher uses a sophisticated edge-based template matching approach:

1. **Background Removal**: Uses HSV saturation to segment the puzzle piece from background
2. **Edge Detection**: Applies Canny edge detection to both template and piece
3. **Multi-Scale Matching**: Tests multiple scales to account for size variations
4. **Multi-Rotation Matching**: Tests rotations (0°, 90°, 180°, 270°)
5. **Normalized Cross-Correlation**: Uses OpenCV's matchTemplate with masks
6. **Best Match Selection**: Returns the position with highest confidence score

### Configuration

The matcher can be configured via these constants in `matcher.py`:
- `COLS`, `ROWS`: Grid dimensions (36x28)
- `EST_SCALE_WINDOW`: Scale factors to test
- `ROTATIONS`: Rotation angles to test
- `CANNY_LOW`, `CANNY_HIGH`: Edge detection thresholds

### Debug Mode

Enable debug mode to save intermediate images:
```bash
export PUZZLE_MATCHER_DEBUG=1
python app.py
```

Debug images will be saved to `media/debug/` showing:
- Raw input images
- Edge detection results
- Match heatmaps
- Final highlighted results

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
