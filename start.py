from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import gradio as gr
import os
from transformers import BitsAndBytesConfig
import threading
from fastapi.middleware.cors import CORSMiddleware
from llm_conversation import GenerateRequest

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and tokenizer globally
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, model_dir=""):
    print(f"Loading model {model_name}...")
    
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
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

@app.on_event("startup")
def startup_event():
    global model, tokenizer
    try:
        model_name = "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3"
        model_dir = "Meta-Llama-3-8B"
        model, tokenizer = load_model(model_name, model_dir)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise

@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        response = generate_text(request.prompt, request.max_length)
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "running"}

# Gradio Interface
def gradio_interface(prompt, max_length):
    return generate_text(prompt, max_length)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here..."),
        gr.Slider(10, 2000, value=100, step=10, label="Max Length")
    ],
    outputs="text",
    title="Wild AI",
    description="Ask whatever!"
)

app = gr.mount_gradio_app(app, iface, path="/gradio")
