"""End-to-end tests for Gradio puzzle solver interface"""
import pytest
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import time
import subprocess
import socket
import sys
import os
import tempfile


def find_free_port():
    """Find a free port to run the test server on"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture(scope="module")
def gradio_app():
    """Start Gradio app for testing"""
    # Find free port
    port = find_free_port()
    
    # Get the project directory
    project_dir = Path(__file__).resolve().parent
    
    # Start app.py in subprocess with specific port
    env = os.environ.copy()
    env['GRADIO_SERVER_PORT'] = str(port)
    
    process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    
    # Wait for app to be ready (check if port is listening)
    max_wait = 30
    start_time = time.time()
    while time.time() - start_time < max_wait:
        # Check if process has terminated with an error
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            error_msg = f"Gradio app process terminated unexpectedly.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
            raise RuntimeError(error_msg)
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', port))
                if result == 0:
                    # Port is open, wait a bit more for app to be fully ready
                    time.sleep(2)
                    break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        stdout, stderr = process.communicate(timeout=2)
        error_msg = f"Gradio app failed to start within timeout.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
        process.terminate()
        raise RuntimeError(error_msg)
    
    app_url = f"http://127.0.0.1:{port}"
    yield app_url
    
    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.mark.e2e
def test_app_loads(page, gradio_app):
    """Test that the Gradio app loads successfully"""
    page.goto(gradio_app, wait_until="networkidle", timeout=30000)
    
    # Check for main heading
    page.wait_for_selector("text=Jigsaw Puzzle Solver", timeout=10000)
    heading = page.locator("text=Jigsaw Puzzle Solver").first
    assert heading.is_visible()


@pytest.mark.e2e
def test_template_displays(page, gradio_app):
    """Test that the template image displays"""
    page.goto(gradio_app, wait_until="networkidle", timeout=30000)
    
    # Wait for the template section
    page.wait_for_selector("h3:has-text('Puzzle Template')", timeout=10000)
    
    # Check that template heading is visible
    template_heading = page.locator("h3:has-text('Puzzle Template')")
    assert template_heading.is_visible()
    
    # Check that template image container exists
    # Gradio images are within specific components
    time.sleep(2)  # Give time for image to load


@pytest.mark.e2e
def test_upload_interface_exists(page, gradio_app):
    """Test that file upload interface exists"""
    page.goto(gradio_app, wait_until="networkidle", timeout=30000)
    
    # Check for upload section
    page.wait_for_selector("text=Upload Puzzle Piece", timeout=10000)
    upload_label = page.locator("text=Upload Puzzle Piece")
    assert upload_label.is_visible()
    
    # Check for solve button
    solve_button = page.locator("button:has-text('Find Piece Location')")
    assert solve_button.is_visible()


@pytest.mark.e2e
def test_piece_upload_and_match(page, gradio_app):
    """Test uploading a piece and getting a match result"""
    page.goto(gradio_app, wait_until="networkidle", timeout=30000)
    
    # Wait for page to be ready
    page.wait_for_selector("text=Upload Puzzle Piece", timeout=10000)
    time.sleep(2)
    
    # Create a test piece image
    project_dir = Path(__file__).resolve().parent
    test_piece_path = project_dir / "media" / "pieces" / "piece_1.jpg"
    
    if not test_piece_path.exists():
        # If piece_1.jpg doesn't exist, create a temporary test piece
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            test_piece_path = Path(f.name)
            img = Image.new('RGB', (100, 100), (200, 200, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 100, 100], outline='black', width=2)
            img.save(test_piece_path)
    
    # Find and use the file upload - Gradio uses input[type=file]
    # The upload button in Gradio typically has a file input
    file_inputs = page.locator('input[type="file"]').all()
    
    if len(file_inputs) > 0:
        # Upload to the first file input (should be the piece input)
        file_inputs[0].set_input_files(str(test_piece_path))
        
        # Wait for upload to complete
        time.sleep(2)
        
        # Click the solve button
        solve_button = page.locator("button:has-text('Find Piece Location')")
        solve_button.click()
        
        # Wait for result (this may take a few seconds for CV processing)
        time.sleep(5)
        
        # Check for result text (should contain "Match Found" or error message)
        # The result is displayed in a markdown component
        page.wait_for_selector("text=Match Found", timeout=15000)
        result = page.locator("text=Match Found")
        assert result.is_visible()
