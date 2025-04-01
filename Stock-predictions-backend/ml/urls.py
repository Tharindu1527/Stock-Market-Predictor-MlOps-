# Stock-predictions-backend/ml/urls.py
from django.urls import path
from ml import views

urlpatterns = [
    path('', views.Data.as_view(), name='data'),
    path('health/', views.HealthCheck.as_view(), name='health'),
    path('models/', views.ModelList.as_view(), name='models'),
    path('models/<str:model_id>/', views.ModelDetail.as_view(), name='model-detail'),
    path('monitor/predictions/', views.PredictionMonitor.as_view(), name='prediction-monitor')
]