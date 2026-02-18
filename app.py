import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

def generate_image(prompt):
    image = pipe(
        prompt,
        num_inference_steps=40,
        guidance_scale=8.5
    ).images[0]
    return image

interface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="AI Image Generator",
    description="Enter a prompt and generate an AI image"
)

interface.launch()
