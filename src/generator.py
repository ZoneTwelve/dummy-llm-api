# generate.py
import time
import random
import json
from collections import deque
from typing import List, Dict, Any
from datasets import load_dataset
import tiktoken
from token_throttle import RPMRateLimiter, SessionTokenThrottle


class FastDummyGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", max_rpm=300, tokens_per_sec=10):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.rpm_limiter = RPMRateLimiter(max_rpm)
        self.token_throttle = SessionTokenThrottle(tokens_per_sec)
        self._load_dataset()
        self._index = random.randint(0, len(self.assistant_responses) - 1)  # random starting index

    def _load_dataset(self):
        dataset = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train")
        messages = list(dataset["messages"])  # list of conversations
        self.assistant_responses = []

        for convo in messages:
            for msg in convo:
                if msg["role"] == "assistant":
                    self.assistant_responses.append(msg["content"])
                    break

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        text = "".join(f"{m['role']}: {m['content']}" for m in messages)
        return len(self.tokenizer.encode(text))

    def get_response(self, messages: List[Dict[str, str]], session_id: str) -> str:
        self.rpm_limiter.check()
        response = self.assistant_responses.popleft()
        self.assistant_responses.append(response)
        return response

    def stream_response(self, messages: List[Dict[str, str]], session_id: str):
        index = int(time.time()) + 0
        self.rpm_limiter.check()
        response = self.assistant_responses.popleft()
        self.assistant_responses.append(response)

        tokens = self.tokenizer.encode(response)
        token_strings = self.tokenizer.decode_tokens_bytes(tokens)

        # Send role
        yield self._format_chunk({"role": "assistant"}, finish_reason=None, series_id=index)

        # Stream tokens one-by-one respecting throughput
        for i, token in enumerate(token_strings):
            self.token_throttle.sleep_until_next_token(session_id, tokens_to_emit=1)
            is_last = i == len(token_strings) - 1
            yield self._format_chunk({"content": token.decode("utf-8", errors="ignore")}, finish_reason="stop" if is_last else None, series_id=index+i)

        yield "data: [DONE]\n\n"

    def _format_chunk(self, delta: Dict[str, Any], finish_reason: str, series_id: int) -> str:
        payload = {
            "id": f"chatcmpl-{series_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason
                }
            ]
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
