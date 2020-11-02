import json
import random
from re import template

from ..ml_helper.intent_classifier import predict_intent
from ..ml_helper.ner_labeler import predict_ner
from ..typo_correction.spell_check_module import loadSpellCheck
from ..constants import *


def load_response_constants():
  with open(RESPONSE_CONSTANTS, 'r') as f:
    content = f.read()

  content = json.loads(content)
  return content


# temp main
response_constants = load_response_constants()
spell_checker = loadSpellCheck(SPELL_CHECK_MODEL_PATH, SPELL_CHECK_DATA_PATH)


def get_reply(text):
  text = spell_checker.fix_sentence(text)

  intent_ranking_prediction = predict_intent(text)
  chosen_intent = list(intent_ranking_prediction.keys())[0]
  intent_confidence = intent_ranking_prediction[chosen_intent]

  bot_name = response_constants["bot_name"]
  
  if (intent_confidence > DEFAULT_FALLBACK_CONFIDENCE):
    response = random.choice(response_constants["template_response"][chosen_intent])

    if chosen_intent == "absence":
      ner_prediction = predict_ner(text)
      if (ner_prediction != []):
        response = response.replace("ner_datetime", ner_prediction[0])
      else: # Entity Undetected
        response = random.choice(response_constants["template_response"]["no_datetime"])

  else:
    response = random.choice(response_constants["template_response"]["fallback"])

  return response.replace("bot_name", bot_name)