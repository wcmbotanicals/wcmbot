# 🧩 Jigsaw Puzzle Solver

A Django web application that helps solve jigsaw puzzles by identifying where individual pieces fit in a template image using computer vision techniques.

## Features

- **Upload puzzle piece images** and automatically find their position in the template
- **Visual highlighting** of the matched position on the template
- **Confidence scoring** for match quality
- **Upload history** tracking all previously matched pieces
- **Responsive web interface** with modern UI/UX
- **HuggingFace Spaces ready** with Gradio interface
- **Fully tested** with pytest and Playwright

## How It Works

The application uses OpenCV-based computer vision algorithms to match puzzle pieces:

1. **Template Matching**: Uses normalized cross-correlation to find similar regions
2. **Feature Detection**: Uses ORB (Oriented FAST and Rotated BRIEF) for more robust matching
3. **Position Highlighting**: Marks the matched location with visual indicators

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/wcmbotanicals/wcmbot.git
cd wcmbot
```

2. Run the setup script:
```bash
./setup.sh
```

3. Run the development server:
```bash
python manage.py runserver
```

4. Access the application at `http://127.0.0.1:8000/`

### Manual Setup

If you prefer to set up manually:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run migrations: `python manage.py migrate`
4. Create sample data: `python create_sample_images.py`
5. Run the server: `python manage.py runserver`

## Usage

### Django Web Interface

1. Navigate to the home page to see available puzzles
2. Click on a puzzle to open the detail page
3. Upload a puzzle piece image using the upload interface
4. The app will automatically find and highlight the piece location
5. View confidence score and exact coordinates of the match

### Gradio Interface (HuggingFace Spaces)

Run the Gradio app:
```bash
python app.py
```

Or deploy directly to HuggingFace Spaces by pushing this repository.

### Admin Interface

Access the Django admin at `http://127.0.0.1:8000/admin/` to:
- Add new puzzle templates
- View all uploaded pieces and their matches
- Manage puzzle data

Create a superuser first:
```bash
python manage.py createsuperuser
```

## Testing

The project includes comprehensive test coverage:

### Unit Tests

Run pytest tests for models, views, and matching logic:
```bash
pytest puzzle/tests.py -v
```

### End-to-End Tests

Run Playwright tests for full UI workflow:
```bash
pytest tests_e2e.py -v
```

### Run All Tests

```bash
pytest -v
```

## Project Structure

```
wcmbot/
├── jigsaw_project/          # Django project settings
│   ├── settings.py          # Configuration
│   ├── urls.py              # URL routing
│   └── wsgi.py              # WSGI application
├── puzzle/                  # Main Django app
│   ├── models.py            # Database models
│   ├── views.py             # View controllers
│   ├── matcher.py           # Image matching algorithms
│   ├── urls.py              # App URL patterns
│   ├── admin.py             # Admin configuration
│   ├── templates/           # HTML templates
│   └── tests.py             # Unit tests
├── media/                   # Uploaded images
│   ├── templates/           # Puzzle templates
│   └── pieces/              # Uploaded pieces
├── app.py                   # Gradio/HuggingFace interface
├── create_sample_images.py  # Sample data generator
├── tests_e2e.py             # Playwright E2E tests
├── conftest.py              # Pytest configuration
├── pytest.ini               # Pytest settings
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Technology Stack

- **Django 4.2**: Web framework
- **OpenCV**: Computer vision and image matching
- **Pillow**: Image processing
- **Gradio**: Alternative UI for HuggingFace
- **pytest**: Testing framework
- **Playwright**: Browser automation for E2E tests

## API Endpoints

- `GET /` - List all puzzle templates
- `GET /puzzle/<id>/` - Puzzle detail page with upload interface
- `POST /puzzle/<id>/upload/` - Upload and match a puzzle piece
- `GET /admin/` - Django admin interface

## Deployment

### HuggingFace Spaces

1. Create a new Space on HuggingFace
2. Push this repository to the Space
3. The `app.py` file will automatically be detected and run

### Docker (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py migrate

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
- Django for the web framework
- Gradio for the interactive interface
- HuggingFace for deployment platform