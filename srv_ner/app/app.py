import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

ner_model = spacy.load("de_core_news_lg")

app = FastAPI()

class Input(BaseModel):
    sentence: str

class Extraction(BaseModel):
    first_index: int
    last_index: int
    name: str
    content: str

class Output(BaseModel):
    extractions: List[Extraction]
    
class NER(BaseModel):
    #name: str
    content: str
    pos: str
    dep: str

class OutputNER(BaseModel):
    output_ners: List[NER]
    

@app.post("/extractions", response_model=Output)
def extractions(input: Input):
    document = ner_model(input.sentence)

    extractions = []
    for entity in document.ents:
        extraction = {}
        extraction["first_index"] = entity.start_char
        extraction["last_index"] = entity.end_char
        extraction["name"] = entity.label_
        extraction["content"] = entity.text
        extractions.append(extraction)
    return {"extractions": extractions}

@app.post("/ner", response_model=OutputNER)
def ner(input: Input):
    document = ner_model(input.sentence)

    output_ners = []
    for entity in document:
        output_ner = {}
        output_ner["content"] = entity.text
        output_ner["pos"] = entity.pos_
        output_ner["dep"] = entity.dep_
        output_ners.append(output_ner)
    return {"output_ners": output_ners}

