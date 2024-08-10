from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import os
from transformers import BitsAndBytesConfig
from llm_conversation import Conversation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, model_dir=""):
    global model, tokenizer
    print(f"Loading model {model_name}...")
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)


def generate_text(prompt, conversation=None):
    max_length=500 # hardcoding because of less GPU VRAM
    if conversation:
        context = conversation.get_context()
        formatted_prompt = f"{context}\nHuman: {prompt}\nAI:"
    else:
        formatted_prompt = f"Human: {prompt}\nAI:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=int(max_length)).to(model.device)
    attention_mask = inputs['attention_mask']
    input_ids = inputs['input_ids']
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=1024,  # Set a reasonable upper limit
            num_return_sequences=1, 
            do_sample=True,
            max_length=int(max_length),
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output.split("AI:")[-1].strip()

    if conversation:
        conversation.add_exchange(prompt, response)

    return response