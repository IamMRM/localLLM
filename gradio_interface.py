import gradio as gr
from llm_conversation import Conversation
from llm_service import generate_text

"""# Gradio Interface
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
)"""

class GradioConversation:
    def __init__(self):
        self.conversation = Conversation()

    def generate(self, prompt,history):
        response = generate_text(prompt, self.conversation)
        history.append((prompt, response))
        return "", history

    def clear(self):
        self.conversation.clear()
        return []
