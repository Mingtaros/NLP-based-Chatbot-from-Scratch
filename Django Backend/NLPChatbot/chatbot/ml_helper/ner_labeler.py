from .enamex_reader import read_enamex_file
from ..constants import *

import re
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')

MAX_LEN = 10
nltk.download('punkt')
data_read = read_enamex_file(NER_DATA_PATH)
model = load_model(NER_MODEL_PATH)

def preprocessData(sentence):
  result = re.sub(r'[^ a-z A-Z 0-9]', " ", sentence.lower())
  result = nltk.word_tokenize(result)
  return result

def changeDataFormat(data_read):
  x_data = []
  y_data = []
  # Label 0 = other, 1 = date
  for data in data_read:
    words = preprocessData(data[0])
    label = [0 for i in range(len(words))]
    for entities in data[1]['entities']:

      ner_words = data[0][entities[0]:entities[1]]
      ner_words = preprocessData(ner_words)

      ner_count = 0
      idx_word = 0
      while (ner_count < len(ner_words) and idx_word < len(words)):
        if words[idx_word] == ner_words[ner_count]:
          label[idx_word] = 1
          ner_count += 1
        idx_word += 1
    x_data.append(words)
    y_data.append(label)
  return np.array(x_data), np.array(y_data)


# Temp Main
cleaned_x, y = changeDataFormat(data_read)
with open(NER_TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

def encodeXData(x):
  new_x = []
  for sentence in x:
    new_seq = []
    for word in sentence:
      try:
        new_seq.append(word2index[word.lower()])
      except KeyError:
        new_seq.append(word2index['OOVword'])
    new_x.append(new_seq)
  return new_x


def getIndex(data):
  for key in word2index:
    if word2index[key] == data:
      return key


def predict_ner(text):
  test_data = preprocessData(text)
  test_data = tokenizer.texts_to_sequences(test_data)
  oov_handler_data = []
  for data in test_data:
    if (len(data) == 0):
      oov_handler_data.append(0)
    else:
      oov_handler_data.append(data[0])
  padded_data = pad_sequences([oov_handler_data], maxlen=MAX_LEN, padding='post')
  pred = model.predict(padded_data[0])
  normalized_pred = [1 if a[0] > 0.5 else 0 for a in pred]

  labelled_word = []
  for token, label in zip(padded_data[0], normalized_pred):
    if label == 1:
      labelled_word.append(token)
  return tokenizer.sequences_to_texts([labelled_word])