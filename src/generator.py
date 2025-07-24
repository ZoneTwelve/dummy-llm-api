import time
import random
import json
from collections import deque
from typing import List, Dict, Any
from datasets import load_dataset
import tiktoken
from token_throttle import RPMRateLimiter, SessionTokenThrottle


class FastDummyGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", max_rpm=60, tokens_per_sec=10):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.rpm_limiter = RPMRateLimiter(max_rpm)
        self.token_throttle = SessionTokenThrottle(tokens_per_sec)
        self._load_dataset()

    def _load_dataset(self):
        dataset = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train")
        all_msgs = dataset["messages"]
        random.shuffle(all_msgs)

        self.assistant_responses = deque()
        for convo in all_msgs:
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
        self.rpm_limiter.check()
        response = self.assistant_responses.popleft()
        self.assistant_responses.append(response)

        tokens = self.tokenizer.encode(response)
        token_strings = self.tokenizer.decode_tokens_bytes(tokens)

        # Send role
        yield self._format_chunk({"role": "assistant"}, finish_reason=None)

        # Stream tokens one-by-one respecting throughput
        for i, token in enumerate(token_strings):
            self.token_throttle.sleep_until_next_token(session_id, tokens_to_emit=1)
            is_last = i == len(token_strings) - 1
            yield self._format_chunk({"content": token.decode("utf-8", errors="ignore")}, finish_reason="stop" if is_last else None)

        yield "data: [DONE]\n\n"

    def _format_chunk(self, delta: Dict[str, Any], finish_reason: str):
        payload = {
            "id": f"chatcmpl-{random.randint(1000, 9999)}",
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
