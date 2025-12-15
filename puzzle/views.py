from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from .models import PuzzleTemplate, PuzzlePiece
from .matcher import find_piece_in_template, highlight_position
import io
import base64


def index(request):
    """Main page showing available puzzle templates"""
    templates = PuzzleTemplate.objects.all()
    return render(request, 'puzzle/index.html', {'templates': templates})


def puzzle_detail(request, template_id):
    """Puzzle detail page where users can upload pieces"""
    template = get_object_or_404(PuzzleTemplate, id=template_id)
    pieces = template.pieces.all().order_by('-uploaded_at')
    return render(request, 'puzzle/detail.html', {
        'template': template,
        'pieces': pieces
    })


def upload_piece(request, template_id):
    """Handle piece upload and matching"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    template = get_object_or_404(PuzzleTemplate, id=template_id)
    
    if 'piece_image' not in request.FILES:
        return JsonResponse({'error': 'No image provided'}, status=400)
    
    piece_file = request.FILES['piece_image']
    
    # Create piece object
    piece = PuzzlePiece(template=template, piece_image=piece_file)
    piece.save()
    
    try:
        # Find the piece location
        x, y, confidence = find_piece_in_template(
            piece.piece_image.path,
            template.template_image.path
        )
        
        # Update piece with matched location
        piece.matched_x = x
        piece.matched_y = y
        piece.confidence_score = confidence
        piece.save()
        
        # Generate highlighted template
        highlighted_img = highlight_position(template.template_image.path, x, y)
        
        # Convert to base64 for response
        buffer = io.BytesIO()
        highlighted_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return JsonResponse({
            'success': True,
            'x': x,
            'y': y,
            'confidence': confidence,
            'highlighted_image': f'data:image/png;base64,{img_str}',
            'piece_id': piece.id
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=500)
