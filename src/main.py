from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from generator import DummyResponseGenerator, FastDummyGenerator
from pydantic import BaseModel
from typing import List, Union, Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
#generator = DummyResponseGenerator(model_name="gpt-3.5-turbo", max_rpm=600, max_tpm=1e8)
generator = FastDummyGenerator(model_name="gpt-3.5-turbo", max_rpm=600, max_tpm=100, session_token_limit=50)
security = HTTPBearer()
MOCK_API_KEY = "sk-mock-api-key-12345"

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != MOCK_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
    messages = [msg.dict() for msg in request.messages]
    session_id = request.user or api_key  # use `user` field or fallback to API key

    try:
        if request.stream:
            return StreamingResponse(
                generator.stream_response(messages, session_id=session_id),
                media_type="text/event-stream"
            )
        else:
            reply = generator.get_response(messages, session_id=session_id)
            return {
                "id": f"chatcmpl-{random.randint(1000,9999)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop"
                    }
                ]
            }
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))

