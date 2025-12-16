"""
Phi-3.5-Mini LLM wrapper + tools
Unified client supporting local Ollama, Azure ML, and Hugging Face API
"""

import os
import requests
from typing import Optional
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Default max tokens (31k for Phi-3.5)
DEFAULT_MAX_TOKENS = 31000


class Phi35MiniClient:
    """
    Unified client for Phi-3.5-Mini that works with:
    - Local LM Studio / Ollama server (OpenAI-compatible API)
    - Azure ML endpoint
    - Hugging Face Inference API
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.backend = backend or os.getenv("LLM_BACKEND", "local")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        
        # Support both OLLAMA_BASE_URL and LLM_BASE_URL for compatibility
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv("LLM_BASE_URL", "http://localhost:1234")
        
        # Ensure base_url has /v1 suffix for OpenAI compatibility
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

        # Model identifiers - support OLLAMA_MODEL env var
        default_model = os.getenv("OLLAMA_MODEL", "phi-3.5-mini-instruct")
        self.text_model = model or default_model
        self.vision_model = model or default_model
        
        # Max tokens configuration
        self.max_tokens = int(os.getenv("AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS))

        # Initialize client based on backend
        if self.backend == "hf":
            self.hf_endpoint = "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct"
            self.client = None
        elif self.backend == "azure":
            self.client = AzureOpenAI(
                azure_endpoint=self.base_url,
                api_key=self.api_key,
                api_version="2024-02-15-preview",
            )
            self.text_model = "phi-3.5-mini-instruct"
            self.vision_model = "phi-3.5-vision-instruct"
        else:  # local (LM Studio / Ollama)
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "lm-studio",  # LM Studio doesn't require a real key
            )

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert chat messages to a single prompt string for HF API."""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                # Handle multimodal content
                text_parts = [c["text"] for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)
            if role == "system":
                prompt += f"<|system|>\n{content}<|end|>\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}<|end|>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}<|end|>\n"
        prompt += "<|assistant|>\n"
        return prompt

    def generate(
        self,
        messages: list[dict],
        is_vision: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response from Phi-3.5-Mini.

        Args:
            messages: List of message dicts with 'role' and 'content'
            is_vision: Whether this is a vision request with images
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        model = self.vision_model if is_vision else self.text_model
        max_tokens = max_tokens or self.max_tokens

        if self.backend == "hf":
            return self._generate_hf(messages, max_tokens, temperature)
        else:
            return self._generate_openai_compatible(
                messages, model, max_tokens, temperature
            )

    def _generate_hf(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> str:
        """Generate using Hugging Face Inference API."""
        prompt = self._messages_to_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
                "do_sample": True,
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(self.hf_endpoint, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)

    def _generate_openai_compatible(
        self, messages: list[dict], model: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate using OpenAI-compatible API (Ollama, Azure)."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response with an image input.

        Args:
            prompt: Text prompt
            image_base64: Base64-encoded PNG image
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]
        return self.generate(messages, is_vision=True, max_tokens=max_tokens, temperature=temperature)


class BaseAgent:
    """Base class for all Brainnet agents."""

    def __init__(
        self,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.llm = Phi35MiniClient(backend=backend, api_key=api_key, base_url=base_url)

    def _create_system_message(self, content: str) -> dict:
        """Create a system message."""
        return {"role": "system", "content": content}

    def _create_user_message(self, content: str) -> dict:
        """Create a user message."""
        return {"role": "user", "content": content}

    def _create_assistant_message(self, content: str) -> dict:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}
