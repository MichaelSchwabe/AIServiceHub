import torch
from germansentiment import SentimentModel

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# standard BERT Model -> https://huggingface.co/oliverguhr/german-sentiment-bert
model1 = SentimentModel()
# news based model -> https://huggingface.co/mdraw/german-news-sentiment-bert
model2 = SentimentModel('mdraw/german-news-sentiment-bert')

app = FastAPI()

class Input(BaseModel):
    sentence: str

class Inputs(BaseModel):
    sentences: List[str]

class Output(BaseModel):
    sentiments1: List[str]
    sentiments2: List[str]


@app.post("/sentiment", response_model=Output)
def summerization(input: Input):
    sentence = []
    sentence.append(input.sentence)
    #print(text)
    return {"sentiments1": model1.predict_sentiment(sentence),
            "sentiments2": model2.predict_sentiment(sentence)}


@app.post("/sentiments", response_model=Output)
def summerization(inputs: Inputs):
    #text = generate_summary(input.sentence)
    #print(text)
    return {"sentiments1": model1.predict_sentiment(inputs.sentences),
            "sentiments2": model2.predict_sentiment(inputs.sentences)}

""" Test text
texts = [
    'Mit keinem guten Ergebniss',
    'Das ist gar nicht mal so gut',
    'Total awesome!',
    'nicht so schlecht wie erwartet',
    'Der Test verlief positiv.',
    'Sie fährt ein grünes Auto.',
    'schwärmt der parteilose Vizebürgermeister und Historiker Christian Matzka von der "tollen Helferszene".',
    'Flüchtlingsheim 11.05 Uhr: Massenschlägerei',
    'Rotterdam habe einen Migrantenanteil von mehr als 50 Prozent.']
"""