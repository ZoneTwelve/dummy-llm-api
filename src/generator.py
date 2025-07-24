import random
import threading
import time
import json
import tiktoken
from datasets import load_dataset
from typing import List, Dict, Any
from collections import deque, defaultdict

class RateLimiter:
    def __init__(self, max_rpm: int, max_tpm: int):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._lock = threading.Lock()
        self._requests = 0
        self._tokens = 0
        self._start_reset_timer()

    def _start_reset_timer(self):
        def reset():
            while True:
                time.sleep(60)
                with self._lock:
                    self._requests = 0
                    self._tokens = 0
        threading.Thread(target=reset, daemon=True).start()

    def check(self, tokens: int):
        with self._lock:
            if self._requests >= self.max_rpm:
                raise RuntimeError("Rate limit exceeded: RPM")
            if self._tokens + tokens > self.max_tpm:
                raise RuntimeError("Rate limit exceeded: TPM")
            self._requests += 1
            self._tokens += tokens


class ResponseGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", max_rpm=60, max_tpm=10000):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.ratelimiter = RateLimiter(max_rpm, max_tpm)

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        text = "".join([m["role"] + ": " + m["content"] for m in messages])
        return len(self.tokenizer.encode(text))

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    def stream_response(self, messages: List[Dict[str, str]]):
        raise NotImplementedError


class DummyResponseGenerator(ResponseGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train")
        self.messages_list = self.dataset["messages"]
        random.shuffle(self.messages_list)
        self.index = 0
        self.lock = threading.Lock()

    def _next_assistant_message(self) -> str:
        with self.lock:
            while True:
                convo = self.messages_list[self.index]
                self.index = (self.index + 1) % len(self.messages_list)
                for msg in convo:
                    if msg.get("role") == "assistant":
                        return msg.get("content", "")

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        token_count = self.count_tokens(messages)
        self.ratelimiter.check(token_count + 100)  # assume 100 tokens output
        return self._next_assistant_message()

    def stream_response(self, messages: List[Dict[str, str]]):
        token_count = self.count_tokens(messages)
        self.ratelimiter.check(token_count + 100)

        response = self._next_assistant_message()
        tokens = response.split()

        yield self._make_chunk({"role": "assistant"}, finish_reason=None)

        for i, token in enumerate(tokens):
            finish = "stop" if i == len(tokens) - 1 else None
            yield self._make_chunk({"content": token}, finish_reason=finish)
            time.sleep(0.05)

        yield "data: [DONE]\n\n"

    def _make_chunk(self, delta: Dict[str, str], finish_reason=None):
        payload = {
            "id": f"chatcmpl-{random.randint(1000,9999)}",
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
        return f"data: {json.dumps(payload)}\n\n"

class FastRateLimiter:
    def __init__(self, max_rpm: int, max_tpm: int):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._requests = 0
        self._tokens = 0
        self._reset_at = time.time() + 60

    def check(self, tokens: int):
        now = time.time()
        if now >= self._reset_at:
            self._requests = 0
            self._tokens = 0
            self._reset_at = now + 60

        if self._requests >= self.max_rpm:
            raise RuntimeError("Rate limit exceeded: RPM")
        if self._tokens + tokens > self.max_tpm:
            raise RuntimeError("Rate limit exceeded: TPM")

        self._requests += 1
        self._tokens += tokens

class FastDummyGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", max_rpm=60, max_tpm=10000, session_token_limit=5000):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.ratelimiter = FastRateLimiter(max_rpm, max_tpm)
        self.session_limits = defaultdict(int)  # session_id -> tokens used
        self.session_token_limit = session_token_limit
        self._load_and_prepare_dataset()

    def _load_and_prepare_dataset(self):
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
        text = "".join([f"{m['role']}: {m['content']}" for m in messages])
        return len(self.tokenizer.encode(text))

    def _check_session_limit(self, session_id: str, tokens_needed: int):
        used = self.session_limits[session_id]
        if used + tokens_needed > self.session_token_limit:
            raise RuntimeError(f"Session token limit exceeded ({used}/{self.session_token_limit})")
        self.session_limits[session_id] += tokens_needed

    def get_response(self, messages: List[Dict[str, str]], session_id: str) -> str:
        input_tokens = self.count_tokens(messages)
        output_tokens = 100  # estimate
        total = input_tokens + output_tokens

        self._check_session_limit(session_id, total)
        self.ratelimiter.check(total)

        response = self.assistant_responses.popleft()
        self.assistant_responses.append(response)
        return response

    def stream_response(self, messages: List[Dict[str, str]], session_id: str):
        input_tokens = self.count_tokens(messages)
        output_tokens = 100
        total = input_tokens + output_tokens

        self._check_session_limit(session_id, total)
        self.ratelimiter.check(total)

        response = self.assistant_responses.popleft()
        self.assistant_responses.append(response)

        tokens = self.tokenizer.encode(response)
        words = response.split()

        yield self._format_chunk(delta={"role": "assistant"}, finish_reason=None)

        for i in range(0, len(words), 8):
            chunk = " ".join(words[i:i+8])
            is_last = i + 8 >= len(words)
            finish = "stop" if is_last else None
            yield self._format_chunk(delta={"content": chunk}, finish_reason=finish)

        yield "data: [DONE]\n\n"

    def _format_chunk(self, delta: Dict[str, Any], finish_reason: str):
        payload = {
            "id": f"chatcmpl-{random.randint(1000,9999)}",
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

