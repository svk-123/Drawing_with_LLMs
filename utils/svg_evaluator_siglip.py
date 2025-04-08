#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 22:49:45 2025

@author: vino
"""

from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import cairosvg
import os
import gc

class SVGMetricEvaluator:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device=None):
        # Initialize the device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"Using device: {self.device}")
        
        # Load the model and processor
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def svg_metric(self, prompt, svg):
        try:
            # Convert SVG to PNG
            cairosvg.svg2png(svg, write_to="./tmp/temp.png")
            
            # Open and process the image
            image = Image.open('./tmp/temp.png').convert("RGB")
            texts = ["SVG illustration of " + prompt]
            inputs = self.processor(text=texts, images=image, padding="max_length", return_tensors="pt")
            
            # Inference without gradient tracking
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)
            
            # Clean up temporary PNG file
            os.remove('./tmp/temp.png')
            
            return probs[0][0].item()
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def close_model(self):
        # Clean up to free memory
        del self.model
        gc.collect()

# Example of usage:
# evaluator = SVGMetricEvaluator()
# result = evaluator.svg_metric("some description", "<svg>...</svg>")
# print(result)
