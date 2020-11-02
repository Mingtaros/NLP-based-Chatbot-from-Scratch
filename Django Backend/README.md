# Django Backend for NLP Chatbot

Use it in your frontend or something

## How To Run
1. Open NLPChatbot directory
2. do `python manage.py runserver`
3. open `localhost:8000` in your app

## Intent Classification
URL: <u>/chatbot/getIntent</u>

GET REQUEST PARAM:
```
text: <your text here>
```

RETURN BODY:
```
{
  status,
  intent,
  intent_confidence,
  intent_ranking: {
    intent_name_1: intent_confidence_1,
    intent_name_2: intent_confidence_2,
    intent_name_3: intent_confidence_3,
    intent_name_4: intent_confidence_4,
    .
    .
    .
  }
}
```

## Named Entity Recognition
URL: <u>/chatbot/getNER</u>

GET REQUEST PARAM:
```
text: <your text here>
```

RETURN BODY:
```
{
  status,
  ner: <list of entities found in text>
}
```