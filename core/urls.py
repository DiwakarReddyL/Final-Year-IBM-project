from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('delete/<int:file_id>/', views.delete_csv, name='delete_csv'),
]
