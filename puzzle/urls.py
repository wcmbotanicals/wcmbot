from django.urls import path
from . import views

app_name = 'puzzle'

urlpatterns = [
    path('', views.index, name='index'),
    path('puzzle/<int:template_id>/', views.puzzle_detail, name='detail'),
    path('puzzle/<int:template_id>/upload/', views.upload_piece, name='upload'),
]
