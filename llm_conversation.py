from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    conversation_id: str = None


class Conversation:
    def __init__(self, max_history=3):
        self.history = []
        self.max_history = max_history

    def add_exchange(self, prompt, response):
        self.history.append((prompt, response))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self):
        return "\n".join([f"Human: {p}\nAI: {r}" for p, r in self.history])

    def clear(self):
        self.history = []
