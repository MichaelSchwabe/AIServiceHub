'''
latent diffusion by prompt to generate an synthetic an image
Author: Michael Schwabe

FastAPI Tutorial: https://www.youtube.com/watch?v=-ykeT6kk4bk

'''
from typing import Optional

from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import FileResponse

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device)

app = FastAPI()

class Input_simple(BaseModel):
    prompt: str
    
class Input(BaseModel):
    prompt: str
    inference_steps: Optional[int] = None
    eta: Optional[float] = None
    guidance_scale: Optional[int] = None


#model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
#ldm = DiffusionPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
@app.post("/generate")
def generate_image(input:Input_simple): #async 
    image = pipe(input.prompt, num_inference_steps=150, guidance_scale=6.5).images[0]
    #return Response(content=image, media_type="image/png")
    path = "temp.png"
    image.save(path)
    return FileResponse(path)

#@app.post("/generate_advanced")
#def generate_image(item:Item):
#    prompt = Item.prompt
#    image = ldm([prompt], num_inference_steps=Item.inference_steps, eta=Item.eta, guidance_scale=Item.guidance_scale)["sample"]
#    return Response(content=image, media_type="image/png") 
    