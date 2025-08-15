import os
import torch
from PIL import Image
from torchvision import transforms as T
import gradio as gr

# Example model class (replace with your actual models)
import sys
sys.path.append("./checkpoints")  # to import models if needed
from unet import UNet   # if you have UNet class in checkpoints/unet.py
from resnet_seg import ResNetSegmentation  # if you have ResNetSegmentation class

# Map model name prefix to classes
MODEL_CLASSES = {
    "unet": UNet,
    "resnet": ResNetSegmentation
}

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# Get list of model files in checkpoints
def get_model_files():
    folder = "checkpoints"
    files = [f for f in os.listdir(folder) if f.endswith(".pth")]
    return files

# Function to extract architecture from filename (adjust to your naming)
def get_arch_from_filename(filename):
    # Simple example: "unet_weights.pth" -> "unet"
    return filename.split("_")[0].lower()

def segment_image(image_path, model_filename):
    if not model_filename:
        return "⚠ No model selected yet."

    model_path = os.path.join("checkpoints", model_filename)
    arch_name = get_arch_from_filename(model_filename)

    if arch_name not in MODEL_CLASSES:
        return f"⚠ No model class found for architecture: {arch_name}"

    model_class = MODEL_CLASSES[arch_name]
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        mask = model(tensor).squeeze().numpy()

    mask_img = Image.fromarray((mask * 255).astype("uint8"))
    return mask_img

# Gradio interface
image_input = gr.Image(type="filepath", label="Upload Image")
model_dropdown = gr.Dropdown(choices=get_model_files(), label="Select Model")
output_image = gr.Image(label="Segmented Output")

app = gr.Interface(
    fn=segment_image,
    inputs=[image_input, model_dropdown],
    outputs=output_image,
    title="Segmentation App",
    description="Upload an image and select a model from the checkpoints folder."
)

if __name__ == "__main__":
    app.launch()
