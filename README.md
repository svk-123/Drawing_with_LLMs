## 🔧 Skills & Tools Applied

1. **Foundation Model Inference** using `transformers` and `Unsloth`  
2. **Synthetic Data Generation** via OpenAI APIs
3. **Full Fine-Tuning (FFT)** with `Unsloth` 
4. **Parameter-Efficient Fine-Tuning (PEFT)** using LoRA with `Unsloth`  
5. **Fine-Tuned Model Inference** with `transformers` and `Unsloth`  
6. **Response Cleaning** using custom functions and criteria  
7. **Response Scoring** with `SigLIP` model  
8. **Design of Experiments (DOE)** using `MLflow` by varying model parameters such as:
   - `temperature`
   - `top_p`
   - `top_k`  
9. **Model Serving**:
   - Serving LLM via `vLLM` for high-throughput inference  
   - Serving `SigLIP` scoring model via `FastAPI`  
10. **End-to-End Inference-Scoring Workflow**:
   - Load input data  
   - Start inference server  
   - Perform predictions  
   - Stop inference server  
   - Clean inference outputs  
   - Start scoring server  
   - Perform scoring  
   - Stop scoring server  


## 🧩 To Be Done
	** compare models from 1B-7B with various finetunes, parameters, etc**
	** explore further instruction truning such as DPO & so on**
	** Write a clear report for documentation**
	** release improved models & train-validation dataset to kaggle along with notebooks for the benifit of the community**

> ⚠️ **Note:**  
> Due to limited VRAM (16 GB), running multiple servers (e.g., inference and scoring) **simultaneously** is not feasible.  
> The pipeline is designed to run these processes **sequentially** to manage memory efficiently.

---
