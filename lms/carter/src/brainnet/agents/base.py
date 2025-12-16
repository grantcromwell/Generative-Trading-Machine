"""
Phi-3.5-Vision-Instruct LLM wrapper via NVIDIA NIM API
Unified vision-capable client for GAF image analysis

NVIDIA API - no local GPU required, handles base64 PNGs natively.
Model: microsoft/phi-3.5-vision-instruct
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# NVIDIA NIM API endpoint
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "microsoft/phi-3.5-vision-instruct"

# Default max tokens
DEFAULT_MAX_TOKENS = 1024


class Phi35VisionClient:
    """
    Client for Phi-3.5-Vision-Instruct via NVIDIA NIM API.
    
    Model: microsoft/phi-3.5-vision-instruct
    
    Handles base64 PNGs of GAF images natively for research analysis.
    No local GPU required - runs on NVIDIA's infrastructure.
    
    Usage:
        export NVIDIA_API_KEY="nvapi-..."
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.model = model or os.getenv("NVIDIA_MODEL", NVIDIA_MODEL)
        self.stream = stream
        self.api_url = os.getenv("NVIDIA_API_URL", NVIDIA_API_URL)
        
        # Max tokens configuration
        self.max_tokens = int(os.getenv("AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS))

        if not self.api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY environment variable.\n"
                "Get your key at: https://build.nvidia.com/"
            )

    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream" if self.stream else "application/json",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        messages: list[dict],
        is_vision: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.20,
        top_p: float = 0.70,
    ) -> str:
        """
        Generate a response from Phi-3.5-Vision-Instruct via NVIDIA API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            is_vision: Whether this is a vision request (same model handles both)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (default 0.20 for precision)
            top_p: Top-p sampling (default 0.70)

        Returns:
            Generated text response
        """
        max_tokens = max_tokens or self.max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": self.stream,
        }

        response = requests.post(
            self.api_url,
            headers=self._get_headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        if self.stream:
            # Collect streamed response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode("utf-8")
                    if line_text.startswith("data: "):
                        try:
                            import json
                            data = json.loads(line_text[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    full_response += delta["content"]
                        except json.JSONDecodeError:
                            pass
            return full_response
        else:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 1024,
        temperature: float = 0.20,
        top_p: float = 0.70,
    ) -> str:
        """
        Generate a response analyzing an image (GAF pattern).
        
        NVIDIA's Phi-3.5-Vision-Instruct uses inline img tags for base64 images.

        Args:
            prompt: Text prompt for analysis
            image_base64: Base64-encoded PNG image (GAF)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            Generated text response with pattern analysis
        """
        # NVIDIA format: inline img tag in content
        content = f'{prompt} <img src="data:image/png;base64,{image_base64}" />'
        
        # Validate image size (NVIDIA limit)
        if len(image_base64) >= 180_000:
            raise ValueError(
                f"Image too large ({len(image_base64)} chars). "
                "NVIDIA API limit is 180,000 chars for inline base64. "
                "Reduce image size or use assets API."
            )
        
        messages = [{"role": "user", "content": content}]
        
        return self.generate(
            messages,
            is_vision=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )


# Backwards compatibility alias
Phi35MiniClient = Phi35VisionClient


class BaseAgent:
    """Base class for all Carter agents using Phi-3.5-Vision-Instruct via NVIDIA NIM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.llm = Phi35VisionClient(api_key=api_key)

    def _create_system_message(self, content: str) -> dict:
        """Create a system message."""
        return {"role": "system", "content": content}

    def _create_user_message(self, content: str) -> dict:
        """Create a user message."""
        return {"role": "user", "content": content}

    def _create_assistant_message(self, content: str) -> dict:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}
