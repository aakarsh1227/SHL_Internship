from pydantic import BaseModel
from typing import List

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

