# File Paths
NLU_DATA_PATH = "../../Intent Classification/nlu.csv"
INTENT_MODEL_PATH = "../../Intent Classification/saved_model/intent_model_best"
NER_DATA_PATH = "../../NER Tanggal/ner_data.txt"
NER_MODEL_PATH = "../../NER Tanggal/saved_model/ner_model_best"
NER_TOKENIZER_PATH = "../../NER Tanggal/saved_model/tokenizer.pickle"
SPELL_CHECK_MODEL_PATH = "chatbot/typo_correction/model/typo-correction"
SPELL_CHECK_DATA_PATH = "chatbot/typo_correction/model/data.json"
RESPONSE_CONSTANTS = "chatbot/response/constants.json"

# ML Helper
INTENTS_USED = ['absence', 'thank_you', 'cancel', 'help', 'default_fallback_intent']
DEFAULT_FALLBACK_CONFIDENCE = 0.4
NER_MAX_LEN = 50

# HTTP statuses
HTTP_STATUS_OK = 200
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_NOT_FOUND = 404
