# Jigsaw Puzzle Solver - Feature Documentation

## Overview
A Gradio web application that uses computer vision to automatically identify where puzzle pieces fit in a template image.

## Core Features

### 1. Image Matching Algorithm
- **Blue Mask Segmentation**: Uses HSV thresholds and morphological cleanup to extract the piece
- **Knob-Aware Scale Estimation**: Estimates scale using puzzle grid cell sizes and inferred tab/knob counts
- **Multi-Scale Matching**: Tests multiple scale factors to handle size variations
- **Multi-Rotation Matching**: Tests 0°, 90°, 180°, and 270° rotations
- **Binary Template Matching**: Runs multi-scale, multi-rotation correlation on binary patterns with top-N ranking
- **Confidence Scoring**: Provides match quality metrics (0-1 scale)

### 2. Gradio Web Interface
- **HuggingFace Ready**: Deployable to HuggingFace Spaces
- **Side-by-Side View**: Template and upload area in one screen
- **Drag & Drop Upload**: Modern file upload with preview
- **Auto tab/knob inference**: No manual tab-count input required
- **Grid Overlay**: Visual grid showing row and column numbers (enabled by default, toggleable in Settings)
  - White dotted grid lines with black numbered labels
  - Configurable margins and visual styling
  - Works with template rotation
- **Template Rotation**: Display templates at 0°, 90°, 180°, or 270° angles
- **Zoomable Visualizations**: Interactive Plotly plots with zoom and pan controls
- **Visual Highlighting**: Shows matched position with colored circles
- **Real-time Processing**: Instant results with highlighted template
- **Multipiece Mode (Batch)**: Detect multiple pieces in a single upload and solve them in sequence
- **Simple API**: Easy to integrate and customize

## Technical Specifications

### Image Processing
- **Input Formats**: PNG, JPEG, JPG, GIF
- **Template Size**: Flexible (default 800x600px)
- **Piece Size**: Flexible (typically 100x100 to 500x500px)
- **Processing Time**: <2 seconds for typical images
- **Grid Configuration**: 36x28 cells by default

### Matching Algorithm Details
1. **Blue Mask Segmentation**: Two HSV ranges isolate blue plastic, followed by morphological cleanup
2. **Knob-Aware Scaling**: Estimates correct template scale from grid cell dimensions and inferred tab/knob counts
3. **Binary Correlation**: Multi-scale, multi-rotation normalized cross-correlation on blurred binary masks
4. **Top-N Ranking**: Keeps several high-confidence candidates for review
5. **Rotation Testing**: 4 cardinal directions (0°, 90°, 180°, 270°)
6. **Interactive Visualization**: Zoomable Plotly plots for detailed match inspection

### Matching Accuracy
- **Perfect Matches**: 90-100% confidence for exact extracts
- **Close Matches**: 70-90% for similar pieces
- **No Match**: <70% confidence indicates poor match

### Performance
- **Storage**: Local file system for templates and pieces
- **Concurrent Users**: Supports multiple simultaneous uploads via Gradio
- **Memory**: Efficient numpy/OpenCV operations

## Use Cases

### Educational
- Teaching computer vision concepts
- Demonstrating template matching algorithms
- Pattern recognition exercises
- OpenCV tutorial examples

### Entertainment
- Virtual puzzle solving
- Puzzle piece organization
- Online puzzle games

### Practical
- Quality control in manufacturing
- Part identification systems
- Pattern matching applications
- Image alignment tasks

## Debug Mode

The primary debugging workflow is via the Gradio UI's "Show diagnostic visualizations" toggle.

There are also a few environment variables that affect performance/debugging:

```bash
# Profile matching (timing logs)
export WCMBOT_PROFILE=1

# Torch acceleration (also set via `python app.py --gpu`)
export WCMBOT_USE_TORCH=1
export WCMBOT_TORCH_DEVICE=mps   # or cuda / cpu
```

## Configuration

Key parameters in `matcher.py`:
- `COLS = 36`: Grid columns
- `ROWS = 28`: Grid rows
- `EST_SCALE_WINDOW`: Scale factors to test around the knob-aware estimate
- `ROTATIONS = [0, 90, 180, 270]`: Rotation angles to test
- `TOP_MATCH_COUNT`: Number of best matches to keep for review
- `KNOB_WIDTH_FRAC`: Contribution of each knob to the estimated full width/height
- Knob/tab counts are inferred by default in the app pipeline

## 3. Template Grid Export

The `export_template_grid.py` script enables high-quality export of puzzle templates with grid overlays to PNG files:

### Export Features
- **DPI Configuration**: Specify output resolution (default 150 DPI)
- **Template Rotation**: Export at any orientation (0°, 90°, 180°, 270°)
- **Dimension Calculation**: Automatically calculates physical dimensions from template configuration
- **RGBA Handling**: Automatically converts RGBA images to RGB for PNG export
- **Custom Output Paths**: Specify where to save exported images
- **Template Listing**: Display all available templates

### Export Usage

```bash
# Export at default DPI (150) with no rotation
python export_template_grid.py sample_puzzle

# Export at custom DPI
python export_template_grid.py sample_puzzle --dpi 100

# Export with rotation
python export_template_grid.py sample_puzzle --rotation 90

# Export to custom location
python export_template_grid.py sample_puzzle --output ~/exports/

# List available templates
python export_template_grid.py --list
```

### Configuration

Templates support the `export_width_cm` parameter in `templates.json` to specify the physical width of the template **before margins are added**. The grid overlay adds 40-pixel margins on each side for row/column labels. The script calculates dimensions at the specified DPI:

```
template_pixels = (export_width_cm / 2.54) * dpi
total_pixels = template_pixels + margin_pixels
```

The export output shows both the template dimensions and the total dimensions including margins.

## 4. Testing Coverage


### E2E Tests (4 tests)
- App loads successfully
- Template image displays correctly
- Upload interface exists
- Complete upload and match workflow

### Test Metrics
- **Coverage**: All critical user workflows
- **Execution Time**: ~20 seconds for full suite
- **Reliability**: 100% pass rate with Playwright

## Deployment Options

### Local Development
```bash
python app.py
```

### HuggingFace Spaces
1. Create a new Space on HuggingFace
2. Push this repository
3. App automatically detected and launched

### Docker
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

## API Access

Gradio provides automatic API endpoints:
- `/api/predict` - Main prediction endpoint
- `/api/docs` - API documentation
- Enable with `api_name="predict"` in Gradio blocks

## Future Enhancements

### Planned Features
- [x] Multiple piece upload at once ("Multipiece mode (batch)")
- [ ] Custom template upload
- [ ] Progress tracking for full puzzle completion
- [ ] Piece rotation detection visualization
- [ ] Configurable matching parameters in UI
- [ ] Image preprocessing options

### Advanced Matching
- [ ] SIFT/SURF feature detection
- [ ] Deep learning-based matching
- [ ] Improved piece boundary detection
- [ ] Color histogram matching
- [ ] Jigsaw tab detection

### UI Improvements
- [ ] Dark mode
- [ ] Mobile responsive design
- [ ] Animation of piece placement
- [ ] Match confidence visualization
- [ ] Batch processing mode

## License
See LICENSE file for details.

## Support
For issues or questions, please open a GitHub issue.
