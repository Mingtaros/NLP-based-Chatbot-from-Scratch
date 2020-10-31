from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('getIntent', views.get_intent_from_chat, name='getIntent')
]