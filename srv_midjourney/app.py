'''
latent diffusion by prompt to generate an synthetic an image
Author: Michael Schwabe

FastAPI Tutorial: https://www.youtube.com/watch?v=-ykeT6kk4bk

'''
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import torch

model_id = "sd-dreambooth-library/cat-toy" #@param {type:"string"}
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

from typing import Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel


def inference(prompt, num_samples=1):
    all_images = [] 
    images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)
    return all_images
    #return pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images

app = FastAPI()

class Input_simple(BaseModel):
    prompt: str
    
class Input(BaseModel):
    prompt: str
    inference_steps: Optional[int] = None
    eta: Optional[float] = None
    guidance_scale: Optional[int] = None
    num_samples: Optional[int] = 1


#model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
#ldm = DiffusionPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
@app.post("/generate")
def generate_image(input:Input_simple):
    #image = pipe(input.prompt, guidance_scale=8.5)["sample"][0]
    #image = pipe(input.prompt, num_images_per_prompt=1, num_inference_steps=50, guidance_scale=8.5)["sample"][0]
    image = pipe(input.prompt, num_images_per_prompt=1, num_inference_steps=50, guidance_scale=8.5)

    #print("##########IMAGE###########",image)
    print("##########image###########",image[0])
    image[0].save("C:/image_midjourney.png")
    print("##########image###########",image[0][0])
    image[0][0].save("C:/image_midjourney1.png")
    return #Response(content=image[0], media_type="image/png") 

#@app.post("/generate_advanced")
#def generate_image(item:Item):
#    prompt = Item.prompt
#    image = ldm([prompt], num_inference_steps=Item.inference_steps, eta=Item.eta, guidance_scale=Item.guidance_scale)["sample"]
#    return Response(content=image, media_type="image/png") 


    