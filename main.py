from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import gradio as gr
import os


def load_model(model_name, model_dir=""):
    print(f"Loading model {model_name}...")
    
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    return model, tokenizer

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

def main():
    global model, tokenizer, device
    model_name = "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "Meta-Llama-3-8B"
    model, tokenizer = load_model(model_name, model_dir)
    
    # Create Gradio interface
    interface = gr.Interface(fn=generate_text, 
                             inputs=[
                                 gr.Textbox(lines=2, placeholder="Enter your question here..."), 
                                 gr.Slider(10, 2000, value=500, step=10, label="Max Length")
                             ], 
                             outputs="text",
                             title="Wild AI",
                             description="Ask whatever!")

    interface.launch()

if __name__ == "__main__":
    main()
