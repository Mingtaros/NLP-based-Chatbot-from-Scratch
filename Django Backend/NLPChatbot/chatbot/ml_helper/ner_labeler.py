from .enamex_reader import read_enamex_file
from .constants import *
from ..typo_correction.spell_check_module import loadSpellCheck

import numpy as np
import nltk
from tensorflow.keras.models import load_model


nltk.download('punkt')
data_read = read_enamex_file(NER_DATA_PATH)
model = load_model(NER_MODEL_PATH)

def preprocessData(sentence):
  result = ' '.join(sentence.lower().split('-'))
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
x, y = changeDataFormat(data_read)
words = set([])
for sentence in x:
  for word in sentence:
    words.add(word.lower())
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['PADword'] = 0
word2index['OOVword'] = 1


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
  for i in range(NER_MAX_LEN - len(text.split(' '))):
    text += ' PADword'
  spell_checker = loadSpellCheck(SPELL_CHECK_MODEL_PATH, SPELL_CHECK_DATA_PATH)
  text = spell_checker.fix_sentence(text)

  test_data = preprocessData(text)
  test_data = encodeXData([test_data])[0]
  pred = model.predict(test_data)
  normalized_pred = [1 if a > 0.5 else 0 for a in pred]
  labelled_token = []
  for i, label in enumerate(normalized_pred):
    if label == 1:
      labelled_token.append(getIndex(test_data[i]))
  return labelled_token
