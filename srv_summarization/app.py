import torch
from transformers import BertTokenizerFast, EncoderDecoderModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'mrm8488/bert2bert_shared-german-finetuned-summarization'
tokenizer = BertTokenizerFast.from_pretrained(ckpt)
model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

app = FastAPI()

class Input(BaseModel):
    sentence: str

class Output(BaseModel):
    summerized_text: str


def generate_summary(text):
   inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
   input_ids = inputs.input_ids.to(device)
   attention_mask = inputs.attention_mask.to(device)
   output = model.generate(input_ids, attention_mask=attention_mask)
   return tokenizer.decode(output[0], skip_special_tokens=True)

@app.post("/summerization", response_model=Output)
def summerization(input: Input):
    #text = generate_summary(input.sentence)
    #print(text)
    return {"summerized_text": generate_summary(input.sentence)}



