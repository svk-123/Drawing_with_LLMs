import os
os.environ["VLLM_USE_V1"] = "0"

import vllm
import torch
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from datasets import Dataset
from vllm.lora.request import LoRARequest
from get_dataset import build_dataset
from config import BASE_MODEL_PATH, LORA_PATH, DATA_PATH


def main():
    llm = vllm.LLM(
        BASE_MODEL_PATH,
        quantization="gptq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )
    
    tokenizer = llm.get_tokenizer()
# to be modified

if __name__ == "__main__":
    main()