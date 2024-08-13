from fastapi import FastAPI, Request
import gradio as gr
#import threading
from fastapi.middleware.cors import CORSMiddleware
from llm_conversation import GenerateRequest
from llm_service import load_model, generate_text
from gradio_interface import GradioConversation

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
        conversation = conversations.get(request.conversation_id)
        if not conversation and request.conversation_id:
            conversation = Conversation()
            conversations[request.conversation_id] = conversation

        response = generate_text(request.prompt, conversation)
        return {"generated_text": response, "conversation_id": request.conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_conversation/{conversation_id}")
def clear_conversation(conversation_id: str):
    if conversation_id in conversations:
        conversations[conversation_id].clear()
        return {"message": "Conversation cleared"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/")
def read_root():
    return {"status": "running"}


gradio_conv = GradioConversation()

with gr.Blocks() as iface:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    msg.submit(gradio_conv.generate, [msg, chatbot], [msg, chatbot])
    clear.click(gradio_conv.clear, outputs=[chatbot])

app = gr.mount_gradio_app(app, iface, path="/gradio")
