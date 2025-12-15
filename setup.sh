#!/bin/bash
# Setup script for the jigsaw puzzle solver

set -e

echo "🧩 Setting up Jigsaw Puzzle Solver..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Run migrations
echo "🔧 Running database migrations..."
python manage.py migrate

# Create sample data
echo "🎨 Creating sample puzzle data..."
python create_sample_images.py

# Create superuser (optional)
echo ""
echo "✅ Setup complete!"
echo ""
echo "To create an admin user, run:"
echo "  python manage.py createsuperuser"
echo ""
echo "To start the Django server, run:"
echo "  python manage.py runserver"
echo ""
echo "To start the Gradio interface, run:"
echo "  python app.py"
echo ""
echo "To run tests:"
echo "  pytest -v"
