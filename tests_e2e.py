"""End-to-end tests using Playwright"""
import pytest
from playwright.sync_api import Page, expect
import subprocess
import time
import signal
import os


@pytest.fixture(scope="module")
def django_server():
    """Start Django development server for E2E tests"""
    # Get the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Make sure migrations are run
    subprocess.run(["python", "manage.py", "migrate"], cwd=project_dir, check=True)
    
    # Start the server
    process = subprocess.Popen(
        ["python", "manage.py", "runserver", "8000"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    yield "http://127.0.0.1:8000"
    
    # Stop the server
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=5)


@pytest.mark.e2e
def test_homepage_loads(page: Page, django_server):
    """Test that the homepage loads correctly"""
    page.goto(django_server)
    
    # Check for main heading
    expect(page.locator("h1")).to_contain_text("Jigsaw Puzzle Solver")


@pytest.mark.e2e
def test_puzzle_template_displays(page: Page, django_server):
    """Test that puzzle template is displayed"""
    page.goto(django_server)
    
    # Should see the sample puzzle
    expect(page.locator("text=Sample Puzzle")).to_be_visible()
    
    # Should have a solve button
    expect(page.locator("text=Solve Puzzle")).to_be_visible()


@pytest.mark.e2e
def test_puzzle_detail_page(page: Page, django_server):
    """Test navigating to puzzle detail page"""
    page.goto(django_server)
    
    # Click on solve puzzle button
    page.click("text=Solve Puzzle")
    
    # Should navigate to detail page
    expect(page.locator("h1")).to_contain_text("Sample Puzzle")
    
    # Should have upload section
    expect(page.locator("text=Upload Puzzle Piece")).to_be_visible()


@pytest.mark.e2e
def test_file_upload_interface(page: Page, django_server):
    """Test that file upload interface is present"""
    page.goto(f"{django_server}/puzzle/1/")
    
    # Should have file input
    file_input = page.locator("input[type='file']")
    expect(file_input).to_be_attached()
    
    # Should have submit button
    expect(page.locator("button[type='submit']")).to_contain_text("Find Piece Location")


@pytest.mark.e2e
def test_upload_piece_functionality(page: Page, django_server):
    """Test uploading a puzzle piece"""
    page.goto(f"{django_server}/puzzle/1/")
    
    # Upload a piece - use relative path from project root
    project_dir = os.path.dirname(os.path.abspath(__file__))
    piece_path = os.path.join(project_dir, "media", "pieces", "piece_1.png")
    
    if os.path.exists(piece_path):
        # Set up file input
        page.set_input_files("input[type='file']", piece_path)
        
        # Click submit
        page.click("button[type='submit']")
        
        # Wait for result
        page.wait_for_selector("#result", state="visible", timeout=10000)
        
        # Should show match result
        expect(page.locator("#result")).to_be_visible()
