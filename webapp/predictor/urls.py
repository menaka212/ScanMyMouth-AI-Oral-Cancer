from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('analyzing/', views.analyzing, name='analyzing'),
    path('analyze/', views.analyze, name='analyze'),
]
