"""Universal Local LLM Client supporting Ollama, llama-cpp-python, vLLM, and Mock.

All operations execute strictly on localhost with zero external cloud dependencies.
"""

import json
import logging
import uuid
from typing import AsyncGenerator, List, Optional, Dict, Any
import httpx

from antimatter.core.config import AntimatterSettings, LLMBackendType, get_settings
from antimatter.brain.types import (
    ChatMessage,
    ChatRole,
    CompletionChunk,
    ToolCall,
    ToolDefinition,
)
from antimatter.brain.tool_extractor import ToolCallExtractor

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """Universal client interface for sovereign local language models."""

    def __init__(self, settings: Optional[AntimatterSettings] = None):
        self.settings = settings or get_settings()
        self.backend = self.settings.llm_backend

    async def check_health(self) -> Dict[str, Any]:
        """Verify local model server connectivity and report available models."""
        if self.backend == LLMBackendType.MOCK:
            return {"status": "ok", "backend": "mock", "models": ["mock-model"]}

        if self.backend == LLMBackendType.OLLAMA:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m.get("name") for m in data.get("models", [])]
                        return {"status": "ok", "backend": "ollama", "models": models}
                    return {"status": "error", "message": f"Ollama HTTP {resp.status_code}"}
            except Exception as e:
                return {"status": "unreachable", "error": str(e), "target": self.settings.ollama_base_url}

        if self.backend == LLMBackendType.VLLM:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self.settings.vllm_base_url}/models")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m.get("id") for m in data.get("data", [])]
                        return {"status": "ok", "backend": "vllm", "models": models}
                    return {"status": "error", "message": f"vLLM HTTP {resp.status_code}"}
            except Exception as e:
                return {"status": "unreachable", "error": str(e), "target": self.settings.vllm_base_url}

        return {"status": "ready", "backend": str(self.backend)}

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Stream model responses token-by-token with tool call detection."""
        target_model = model or self.settings.planner_model

        if self.backend == LLMBackendType.MOCK:
            yield CompletionChunk(delta_text="Antimatter Mock Engine: Operational and offline.")
            yield CompletionChunk(finish_reason="stop")
            return

        if self.backend == LLMBackendType.OLLAMA:
            async for chunk in self._stream_ollama(messages, tools, target_model, temperature):
                yield chunk
            return

        if self.backend == LLMBackendType.VLLM:
            async for chunk in self._stream_vllm(messages, tools, target_model, temperature):
                yield chunk
            return

        raise NotImplementedError(f"Backend '{self.backend}' streaming is not yet initialized.")

    async def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ChatMessage:
        """Accumulate streaming chat response into a single ChatMessage."""
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        async for chunk in self.stream_chat(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
        ):
            if chunk.delta_text:
                text_parts.append(chunk.delta_text)
            if chunk.tool_call_delta:
                tool_calls.append(chunk.tool_call_delta)

        return ChatMessage(
            role=ChatRole.ASSISTANT,
            content="".join(text_parts),
            tool_calls=tool_calls if tool_calls else None,
        )

    @staticmethod
    def _normalize_tool_arguments(raw_args: Any) -> Dict[str, Any]:
        """Normalize tool call arguments across local models and serving backends.

        Handles JSON string parsing and extracts actual parameter values when small
        local models inadvertently mirror schema artifacts (e.g. nested 'properties').
        """
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                return {"raw": raw_args}

        if not isinstance(raw_args, dict):
            return {}

        args = dict(raw_args)
        schema_keys = {"type", "properties", "required", "$schema", "description"}
        real_top_level_keys = {k for k in args.keys() if k not in schema_keys}

        if real_top_level_keys:
            if "properties" in args and isinstance(args["properties"], dict):
                del args["properties"]
            if "type" in args and args["type"] == "object":
                del args["type"]
            if "required" in args and isinstance(args["required"], list):
                del args["required"]
            return args

        if "properties" in args and isinstance(args["properties"], dict):
            extracted = {}
            for k, v in args["properties"].items():
                if not isinstance(v, dict):
                    if v is not None:
                        extracted[k] = v
                elif "value" in v:
                    extracted[k] = v["value"]
            if extracted:
                return extracted

        return args

    async def _stream_ollama(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]],
        model: str,
        temperature: float,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Stream from Ollama /api/chat endpoint with hybrid tool-call extraction."""
        url = f"{self.settings.ollama_base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    **({"tool_calls": [tc.model_dump() for tc in m.tool_calls]} if m.tool_calls else {})
                }
                for m in messages
            ],
            "options": {"temperature": temperature},
            "stream": True,
        }

        if tools:
            payload["tools"] = [t.model_dump() for t in tools]

        accumulated_text = ""
        yielded_native_tool_call = False

        async with httpx.AsyncClient(timeout=self.settings.max_exec_timeout_sec) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err_msg = await response.aread()
                    raise RuntimeError(f"Ollama error ({response.status_code}): {err_msg.decode()}")

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message_dict = data.get("message", {})
                    content_delta = message_dict.get("content", "")
                    raw_tool_calls = message_dict.get("tool_calls")
                    done = data.get("done", False)

                    if raw_tool_calls:
                        for tc in raw_tool_calls:
                            func = tc.get("function", {})
                            name = func.get("name", "unknown_tool")
                            call_id = str(uuid.uuid4())[:8]
                            normalized_args = self._normalize_tool_arguments(func.get("arguments", {}))
                            
                            # Filter out hallucinated empty calls from tiny models
                            if not normalized_args:
                                continue

                            yielded_native_tool_call = True
                            yield CompletionChunk(
                                tool_call_delta=ToolCall(
                                    id=call_id,
                                    name=name,
                                    arguments=normalized_args,
                                )
                            )

                    if content_delta:
                        accumulated_text += content_delta
                        yield CompletionChunk(delta_text=content_delta)

                    if done:
                        # Fallback extraction if model output tool calls in freeform text
                        if not yielded_native_tool_call and tools and accumulated_text:
                            _, fallback_tcs = ToolCallExtractor.extract_from_text(accumulated_text, tools)
                            for tc in fallback_tcs:
                                if tc.arguments:
                                    yield CompletionChunk(tool_call_delta=tc)

                        yield CompletionChunk(finish_reason="stop")

    async def _stream_vllm(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]],
        model: str,
        temperature: float,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Stream from OpenAI-compatible vLLM /v1/chat/completions endpoint."""
        url = f"{self.settings.vllm_base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = [t.model_dump() for t in tools]

        async with httpx.AsyncClient(timeout=self.settings.max_exec_timeout_sec) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise RuntimeError(f"vLLM error ({response.status_code}): {err_text.decode()}")

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        yield CompletionChunk(finish_reason="stop")
                        break

                    data_str = line[6:]
                    try:
                        chunk_json = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        finish = choices[0].get("finish_reason")
                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            for tc in tool_calls:
                                func = tc.get("function", {})
                                normalized_args = self._normalize_tool_arguments(func.get("arguments", {}))
                                yield CompletionChunk(
                                    tool_call_delta=ToolCall(
                                        id=tc.get("id", str(uuid.uuid4())[:8]),
                                        name=func.get("name", "unknown_tool"),
                                        arguments=normalized_args,
                                    )
                                )
                        if text:
                            yield CompletionChunk(delta_text=text)
                        if finish:
                            yield CompletionChunk(finish_reason=finish)
