from fastapi import FastAPI, HTTPException, Header, Request
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModel
from PIL import Image
import cairosvg
import io
import torch
import gc
import uvicorn
import subprocess
import time

API_KEY = "my-api-key"

class SVGMetricEvaluator:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"Using device: {self.device}")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def svg_metric(self, prompt: str, svg: str) -> float:
        try:

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
            print(f"Error during evaluation: {e}")
            raise e

evaluator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global evaluator
    evaluator = SVGMetricEvaluator()
    yield
    evaluator.close_model()

app = FastAPI(lifespan=lifespan)

class SVGRequest(BaseModel):
    prompt: str
    svg: str

@app.post("/evaluate_svg")
async def evaluate_svg(req: SVGRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        score = evaluator.svg_metric(req.prompt, req.svg)
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("siglip_eval_server:app", host="127.0.0.1", port=8000, reload=True)
