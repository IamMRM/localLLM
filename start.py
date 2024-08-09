from fastapi import FastAPI, Request
import gradio as gr
#import threading
from fastapi.middleware.cors import CORSMiddleware
from llm_conversation import GenerateRequest
from llm_service import load_model, generate_text

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    global model, tokenizer
    try:
        model_name = "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3"
        model_dir = "Meta-Llama-3-8B"
        load_model(model_name, model_dir)
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
