from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
   
   path('VI', views.VirtualAssistant, name="virtual_assistant"),
   
   path('virtual_assistant/chat_application', views.ChatApp, name="chat_app"),
   
   path('virtual_assistant/ML', views.MLApp, name="ml_app"),
   
   path('virtual_assistant/OCR', views.OCR, name="ocr_app"),
   
   path('virtual_assistant/dashboard', views.ChartsJs, name="dashboard_app"),

   path('virtual_assistant/chat_bot', views.ChatBot, name="chat_bot"),

   path('bot-response', views.bot_response, name='bot_response'),

   
   # path('save_modal', views.SaveModal, name="save_modal"),
   
   # path('edit_table', views.EditTable, name="edit_table"),
    
    
    
]