from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import json

from .ml_helper.intent_classifier import predict

# Create your views here.
def index(request):
  try:
    return JsonResponse({
      "status": 200,
      "message": "sent"
    })
  except Exception as e:
    return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

@api_view(["GET"])
def get_intent_from_chat(request):
  data = json.loads(json.dumps(request.query_params))
  
  if ('text' not in data):
    return Response(status=status.HTTP_400_BAD_REQUEST)

  intent_ranking_prediction = predict(data['text'])
  chosen_intent = list(intent_ranking_prediction.keys())[0]

  return JsonResponse({
    "status": 200,
    "intent": chosen_intent,
    "intent_confidence": intent_ranking_prediction[chosen_intent],
    "intent_ranking": intent_ranking_prediction
  })