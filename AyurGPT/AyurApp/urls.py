from django.urls import path
from . import views

urlpatterns = [
    path('api/register/', views.register_api, name='register'),
    path('api/login/', views.login_api, name='login'),
    path('api/user/', views.get_user_api, name='user'),
    path('api/chat-history/', views.get_chat_history, name='chat_history'),
    path('api/chat/', views.chat, name='chat'),
]