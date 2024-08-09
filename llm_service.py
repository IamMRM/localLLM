from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import os
from transformers import BitsAndBytesConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, model_dir=""):
    global model, tokenizer
    print(f"Loading model {model_name}...")
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)


def generate_text(prompt, max_length=50):
    formatted_prompt = f"Q: {prompt}\nA:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=int(max_length)).to(model.device)
    attention_mask = inputs['attention_mask']
    input_ids = inputs['input_ids']
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=int(max_length), 
            num_return_sequences=1, 
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id  # Automatically handle end of sequence
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output