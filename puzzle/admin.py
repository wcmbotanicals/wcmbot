from django.contrib import admin
from .models import PuzzleTemplate, PuzzlePiece


@admin.register(PuzzleTemplate)
class PuzzleTemplateAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at', 'piece_count')
    search_fields = ('name',)
    
    def piece_count(self, obj):
        return obj.pieces.count()
    piece_count.short_description = 'Pieces'


@admin.register(PuzzlePiece)
class PuzzlePieceAdmin(admin.ModelAdmin):
    list_display = ('id', 'template', 'matched_x', 'matched_y', 'confidence_score', 'uploaded_at')
    list_filter = ('template', 'uploaded_at')
    search_fields = ('template__name',)
