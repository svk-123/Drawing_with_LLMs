from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import cairosvg
import os
import gc
import io

class SVGMetricEvaluator:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device=None):
        # Initialize the device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        
        # Load the model and processor
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def svg_metric(self, prompt, svg):
        try:
            # Convert SVG to PNG
            #cairosvg.svg2png(svg, write_to="temp.png")
            #image = Image.open('temp.png').convert("RGB")

            png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

            texts = ["SVG illustration of " + prompt]
            inputs = self.processor(text=texts, images=image, padding="max_length", return_tensors="pt").to(self.device)
            
            # Inference without gradient tracking
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)
            
            
            return probs[0][0].item()
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
