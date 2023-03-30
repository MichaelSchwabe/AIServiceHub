import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de")


app = FastAPI()

class Input(BaseModel):
    sentence: str

class Output(BaseModel):
    trans_text: str

def generate_trans(text):
   input_ids = tokenizer([text], max_length= 1024, return_tensors="pt", add_special_tokens=False).input_ids
   output_ids = model.generate(input_ids)[0]
   #attention_mask = inputs.attention_mask.to(device)
   #output = model.generate(input_ids, attention_mask=attention_mask)
   return tokenizer.decode(output_ids, skip_special_tokens=True)

@app.post("/translation", response_model=Output)
def translation(input: Input):
    #text = generate_trans(input.sentence)
    #print(text)
    return {"trans_text": generate_trans(input.sentence)}


#sentence = "Would you like to grab a coffee with me this week?"

#input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).input_ids
#output_ids = model.generate(input_ids)[0]
#print(tokenizer.decode(output_ids, skip_special_tokens=True))