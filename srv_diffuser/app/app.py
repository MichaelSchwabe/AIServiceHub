'''
latent diffusion by prompt to generate an synthetic an image
Author: Michael Schwabe

FastAPI Tutorial: https://www.youtube.com/watch?v=-ykeT6kk4bk

'''
from typing import Optional
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Response
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    prompt: str
    inference_steps: Optional[int] = None
    eta: Optional[float] = None
    guidance_scale: Optional[int] = None

model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
@app.post("/generate")
def generate_image(item:Item):
    prompt = Item.prompt
    image = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]
    return Response(content=image, media_type="image/png") 

@app.post("/generate_advanced")
def generate_image(item:Item):
    prompt = Item.prompt
    image = ldm([prompt], num_inference_steps=Item.inference_steps, eta=Item.eta, guidance_scale=Item.guidance_scale)["sample"]
    return Response(content=image, media_type="image/png") 