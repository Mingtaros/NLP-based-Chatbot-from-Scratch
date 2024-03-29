import numpy as np
import pandas as pd
import re

import nltk
from nltk import word_tokenize
nltk.download('punkt')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

from ..constants import *


data = pd.read_csv(NLU_DATA_PATH)
# use only 5 intents
data = data[data['intent'].isin(INTENTS_USED)]
model = load_model(INTENT_MODEL_PATH)

# Clean data:
#   - Strip data from special characters
#   - Tokenize words
#   - Lowercase all word
def clean_data(text_data):
  words = []
  for sentence in text_data:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", sentence)
    tokenized_words = word_tokenize(clean)

    words.append([word.lower() for word in tokenized_words])

  return words


# Get max length of every word in words
def get_max_length(words):
  return len(max(words, key=len))


def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen=max_length, padding="post"))


def onehot_encode(data):
  encoder = OneHotEncoder(sparse=False)
  return encoder.fit_transform(data)


def create_tokenizer(words):
  filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token


def initialize_predictor():
  # clean text data
  cleaned_text_data = clean_data(data['text'])
  tokenizer = create_tokenizer(cleaned_text_data)
  max_length = get_max_length(cleaned_text_data)

  # onehot the intent map
  encoded_intent = onehot_encode(data['intent'].values.reshape(-1, 1))
  intent_map = {}
  for intent, onehot_intent in zip(data['intent'].values, encoded_intent):
    if (intent not in intent_map):
      intent_map[intent] = list(onehot_intent).index(1)

  return max_length, tokenizer, intent_map


# temp main
max_length, tokenizer, intent_map = initialize_predictor()


def clean_data(text, tokenizer):
  # clean the input text
  cleaned_text = re.sub(r'[^ a-z A-Z 0-9]', " ", text)

  test_word = word_tokenize(cleaned_text)
  test_word = [w.lower() for w in test_word]
  test_ls = tokenizer.texts_to_sequences(test_word)
  
  return test_ls


def predict_intent(text):
  test_ls = clean_data(text, tokenizer)

  # Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
  x = padding_doc(test_ls, max_length)

  prediction_result = model.predict(x)
  prediction_result = prediction_result[0]

  # convert prediction result to intent word
  unique_intents = intent_map.keys()

  # sort value by intent ranking
  intent_ranking = {}
  for each_intent in unique_intents:
    intent_ranking[each_intent] = float(prediction_result[intent_map[each_intent]])

  sorted_intent_ranking = sorted(intent_ranking.items(), key=lambda kv: kv[1], reverse=True)
  
  return dict(sorted_intent_ranking)