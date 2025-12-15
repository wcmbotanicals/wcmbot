"""Image matching functionality for jigsaw puzzle pieces"""
import cv2
import numpy as np
from PIL import Image


def find_piece_in_template(piece_image_path, template_image_path):
    """
    Find the location of a puzzle piece in the template image.
    
    Args:
        piece_image_path: Path to the piece image
        template_image_path: Path to the template image
        
    Returns:
        tuple: (x, y, confidence) where x,y is the center position and confidence is the match score
    """
    # Load images
    piece = cv2.imread(str(piece_image_path))
    template = cv2.imread(str(template_image_path))
    
    if piece is None or template is None:
        raise ValueError("Could not load images")
    
    # Convert to grayscale
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Try template matching first (simple approach)
    result = cv2.matchTemplate(template_gray, piece_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Get the center of the matched region
    h, w = piece_gray.shape
    x = max_loc[0] + w // 2
    y = max_loc[1] + h // 2
    confidence = float(max_val)
    
    return x, y, confidence


def find_piece_with_features(piece_image_path, template_image_path):
    """
    Find piece location using feature matching (more robust to rotation/scale).
    
    Args:
        piece_image_path: Path to the piece image
        template_image_path: Path to the template image
        
    Returns:
        tuple: (x, y, confidence) where x,y is the center position and confidence is the match score
    """
    # Load images
    piece = cv2.imread(str(piece_image_path))
    template = cv2.imread(str(template_image_path))
    
    if piece is None or template is None:
        raise ValueError("Could not load images")
    
    # Convert to grayscale
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(piece_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # Fall back to template matching if not enough features
        return find_piece_in_template(piece_image_path, template_image_path)
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 4:
        # Fall back to template matching if not enough matches
        return find_piece_in_template(piece_image_path, template_image_path)
    
    # Get matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
    
    # Calculate center of matched region
    x = int(np.mean(dst_pts[:, 0, 0]))
    y = int(np.mean(dst_pts[:, 0, 1]))
    
    # Calculate confidence based on match quality
    confidence = min(1.0, len(matches) / 50.0)
    
    return x, y, confidence


def highlight_position(template_image_path, x, y, radius=30):
    """
    Create a highlighted version of the template with the matched position marked.
    
    Args:
        template_image_path: Path to the template image
        x, y: Position to highlight
        radius: Radius of the highlight circle
        
    Returns:
        PIL Image: Highlighted template image
    """
    # Load template
    template = cv2.imread(str(template_image_path))
    
    if template is None:
        raise ValueError("Could not load template image")
    
    # Draw a bright circle at the matched position
    cv2.circle(template, (int(x), int(y)), radius, (0, 255, 0), 3)
    cv2.circle(template, (int(x), int(y)), radius + 5, (255, 255, 0), 2)
    
    # Convert from BGR to RGB for PIL
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(template_rgb)
