import torch
from transformers import AutoModelForCausalLM

def load(info=False):
    base_model = "codellama/CodeLlama-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(base_model,
        torch_dtype=torch.float16,
        device_map="auto")

    print("Llam model loaded")
    if info:
        print(model)

    return model

if __name__ != "__main__":
    print(f"file: {__name__}")