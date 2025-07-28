# main.py
import time
import uuid
import random
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from starlette.middleware.cors import CORSMiddleware
from generator import FastDummyGenerator
import json

# --- App Initialization ---
app = FastAPI(
    title="Mock OpenAI Chat Completion API",
    description="A mock implementation of the OpenAI Chat API with rate limiting.",
    version="1.0.0",
)

# --- CORS (Optional) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key Auth ---
MOCK_API_KEY = "sk-mock-api-key-12345"
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != MOCK_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# --- Pydantic Models ---
class Message(BaseModel):
    role: str
    content: str

    @validator("role")
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 256
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None

    @validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages array cannot be empty")
        return v


# --- Generator Setup ---
generator = FastDummyGenerator(
    model_name="gpt-4",
    max_rpm=3000,           # Allow up to 60 requests per minute
    tokens_per_sec=60     # Limit 10 tokens per second per session
)

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    return {
        "data": [
            {
                "id": generator.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
                "permission": [
                    {
                        "id": str(uuid.uuid4()),
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": False,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
            }
        ]
    }

# --- Completion Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
):
    session_id = request.user or api_key
    messages = [msg.dict() for msg in request.messages]

    try:
        if request.stream:
            return StreamingResponse(
                generator.stream_response(messages, session_id=session_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            reply = generator.get_response(messages, session_id=session_id)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": reply,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# --- Run with: uvicorn main:app --reload ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

