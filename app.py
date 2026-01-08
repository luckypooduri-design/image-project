import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Florence-2 model and processor
florence_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
).to(device)
florence_model.eval()

florence_processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
)

def generate_caption(image):
    if image is None:
        return "Please upload an image."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    inputs = florence_processor(
        text="<MORE_DETAILED_CAPTION>",
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3
        )

    generated_text = florence_processor.batch_decode(
        generated_ids,
