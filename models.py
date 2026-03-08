from pydantic import BaseModel


class ChatMessageInput(BaseModel):
    role: str   # "system", "user", "assistant"
    content: str
