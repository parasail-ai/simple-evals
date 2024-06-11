import atexit
import base64
import dataclasses
import pathlib
import threading
import time
import uuid
from typing import Any, Optional

import jsonlines

from ..types import MessageList, SamplerBase


@dataclasses.dataclass(frozen=True)
class _RequestBody:
    model: str
    messages: MessageList
    temperature: Optional[float]
    max_tokens: Optional[int]

    def __hash__(self) -> int:
        messages = tuple(frozenset(msg.items()) for msg in self.messages)
        return hash((self.model, messages, self.temperature, self.max_tokens))


class BatchChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str,
        working_dir: str,
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        instrument: bool = False,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.instrument = instrument
        self.responses = {}
        self.working_dir = working_dir

        if self.instrument:
            self.requests_writer = jsonlines.open(
                f"{working_dir}/requests.jsonl", mode="w"
            )
            self.requests_writer_lock = threading.Lock()
            atexit.register(lambda: self.requests_writer.close())
        else:
            self._load_responses(
                pathlib.Path(f"{working_dir}/requests.jsonl"),
                pathlib.Path(f"{working_dir}/responses.jsonl"),
            )

    def _write_request(self, request_body: _RequestBody):
        with self.requests_writer_lock:
            request = {
                "custom_id": uuid.uuid4().hex,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": dataclasses.asdict(request_body),
            }
            self.requests_writer.write(request)

    def _load_responses(
        self, requests_jsonl: pathlib.Path, responses_jsonl: pathlib.Path
    ):
        response_texts = {}
        with jsonlines.open(responses_jsonl) as responses:
            for resp in responses:
                text = resp["response"]["body"]["choices"][0]["message"]["content"]
                response_texts[resp["custom_id"]] = text

        with jsonlines.open(requests_jsonl) as requests:
            for req in requests:
                req_body = req["body"]
                key = _RequestBody(
                    model=req_body["model"],
                    messages=req_body["messages"],
                    temperature=req_body["temperature"],
                    max_tokens=req_body["max_tokens"],
                )
                self.responses[key] = response_texts[req["custom_id"]]

    def pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [
                self.pack_message("system", self.system_message)
            ] + message_list

        request_body = _RequestBody(
            model=self.model,
            messages=message_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if self.instrument:
            self._write_request(request_body)
            return ""

        return self.responses[request_body]
