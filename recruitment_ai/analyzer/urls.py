from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path("analyze-body/", views.analyze_body_language, name="analyze_body"),
]