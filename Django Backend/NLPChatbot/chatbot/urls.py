from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('getIntent', views.get_intent_from_chat, name='getIntent'),
    path('getNER', views.get_ner_result, name='getNER'),
    path('getReply', views.get_response, name='getReply')
]
