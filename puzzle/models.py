from django.db import models


class PuzzleTemplate(models.Model):
    """Stores the template image of the complete jigsaw puzzle"""
    name = models.CharField(max_length=200)
    template_image = models.ImageField(upload_to='templates/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name


class PuzzlePiece(models.Model):
    """Stores uploaded puzzle pieces and their matched positions"""
    template = models.ForeignKey(PuzzleTemplate, on_delete=models.CASCADE, related_name='pieces')
    piece_image = models.ImageField(upload_to='pieces/')
    matched_x = models.IntegerField(null=True, blank=True)
    matched_y = models.IntegerField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Piece {self.id} for {self.template.name}"
