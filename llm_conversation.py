from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 2000

