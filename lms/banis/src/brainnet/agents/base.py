"""
DeepSeek-V3 LLM wrapper using HuggingFace Pipeline
Unified client supporting local HF pipeline, HF Inference API, and OpenAI-compatible endpoints
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default max tokens (128K context for DeepSeek-V3)
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONTEXT_LENGTH = 131072  # 128K context window


class DeepSeekV3Client:
    """
    Unified client for DeepSeek-V3 that works with:
    - Local HuggingFace Pipeline (with 4-bit quantization)
    - HuggingFace Inference API
    - OpenAI-compatible API (LM Studio, Ollama, vLLM)
    - DeepSeek API (api.deepseek.com)
    
    Model: deepseek-ai/DeepSeek-V3-0324 (685B MoE, 37B active params)
    Context: 128K tokens
    """

    # Default model IDs
    HF_MODEL_ID = "deepseek-ai/DeepSeek-V3-0324"
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        torch_dtype: Optional[str] = None,
    ):
        """
        Initialize DeepSeek-V3 client.
        
        Args:
            backend: One of "pipeline", "hf_api", "openai", "deepseek"
                     Defaults to env LLM_BACKEND or "openai"
            api_key: API key for HF/DeepSeek/OpenAI endpoints
            base_url: Base URL for OpenAI-compatible API
            model: Model ID override
            device: Device for local pipeline ("auto", "cuda", "cpu")
            load_in_4bit: Use 4-bit quantization for local pipeline
            torch_dtype: Torch dtype ("float16", "bfloat16", "auto")
        """
        self.backend = backend or os.getenv("LLM_BACKEND", "openai")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        self.model_id = model or os.getenv("DEEPSEEK_MODEL", self.HF_MODEL_ID)
        self.device = device or os.getenv("LLM_DEVICE", "auto")
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype or os.getenv("LLM_DTYPE", "auto")
        
        # Max tokens configuration
        self.max_tokens = int(os.getenv("AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        
        # Pipeline cache (lazy loaded)
        self._pipeline = None
        self._tokenizer = None
        self._openai_client = None
        
        # Initialize based on backend
        if self.backend == "pipeline":
            # Lazy load - will initialize on first use
            logger.info(f"DeepSeek-V3 pipeline mode (will load on first use)")
        elif self.backend == "hf_api":
            self.hf_endpoint = f"https://api-inference.huggingface.co/models/{self.model_id}"
            logger.info(f"DeepSeek-V3 HF Inference API: {self.model_id}")
        elif self.backend == "deepseek":
            self._init_openai_client(self.DEEPSEEK_API_URL)
            self.model_id = "deepseek-chat"  # DeepSeek API model name
            logger.info(f"DeepSeek API: {self.DEEPSEEK_API_URL}")
        else:  # openai-compatible (default)
            self._init_openai_client(self.base_url)
            logger.info(f"DeepSeek-V3 OpenAI-compatible: {self.base_url}")
    
    def _init_openai_client(self, base_url: str):
        """Initialize OpenAI-compatible client."""
        try:
            from openai import OpenAI
            
            # Ensure /v1 suffix
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            
            self._openai_client = OpenAI(
                base_url=base_url,
                api_key=self.api_key or "not-needed",
            )
        except ImportError:
            logger.warning("openai package not installed, falling back to requests")
            self._openai_client = None
    
    def _init_pipeline(self):
        """
        Initialize HuggingFace pipeline for local inference.
        
        Uses 4-bit quantization by default for memory efficiency.
        DeepSeek-V3 with 4-bit: ~100GB VRAM (8x A100 or similar)
        """
        if self._pipeline is not None:
            return
        
        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            logger.info(f"Loading DeepSeek-V3 pipeline: {self.model_id}")
            logger.info(f"  Device: {self.device}, 4-bit: {self.load_in_4bit}")
            
            # Determine torch dtype
            if self.torch_dtype == "auto":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            
            # Quantization config for 4-bit
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            )
            
            # Create pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
                device_map=self.device,
            )
            
            logger.info("DeepSeek-V3 pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format chat messages for DeepSeek-V3.
        
        DeepSeek uses a similar format to ChatML:
        <|begin▁of▁sentence|>System: {system}
        
        User: {user}
        
        Assistant: {assistant}<|end▁of▁sentence|>
        """
        formatted = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle multimodal content (extract text only)
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)
            
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        # Add final assistant prompt
        formatted += "Assistant:"
        
        return formatted
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from DeepSeek-V3.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated text response
        """
        max_tokens = max_tokens or self.max_tokens
        
        if self.backend == "pipeline":
            return self._generate_pipeline(messages, max_tokens, temperature, top_p, stop)
        elif self.backend == "hf_api":
            return self._generate_hf_api(messages, max_tokens, temperature, top_p)
        else:  # openai-compatible or deepseek
            return self._generate_openai(messages, max_tokens, temperature, top_p, stop)
    
    def _generate_pipeline(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> str:
        """Generate using local HuggingFace pipeline."""
        # Lazy load pipeline
        self._init_pipeline()
        
        # Format messages
        prompt = self._format_messages(messages)
        
        # Generate
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            return_full_text=False,
        )
        
        response = outputs[0]["generated_text"]
        
        # Apply stop sequences
        if stop:
            for s in stop:
                if s in response:
                    response = response.split(s)[0]
        
        return response.strip()
    
    def _generate_hf_api(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate using HuggingFace Inference API."""
        prompt = self._format_messages(messages)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False,
                "do_sample": temperature > 0,
            },
        }
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        response = requests.post(
            self.hf_endpoint,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return str(result)
    
    def _generate_openai(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> str:
        """Generate using OpenAI-compatible API."""
        if self._openai_client:
            response = self._openai_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            return response.choices[0].message.content or ""
        else:
            # Fallback to requests
            url = f"{self.base_url}/chat/completions"
            if not url.startswith("http"):
                url = f"http://{url}"
            
            payload = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop:
                payload["stop"] = stop
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response with an image input.
        
        Note: DeepSeek-V3 base model doesn't have vision capability.
        This falls back to text-only generation with image description prompt.
        For vision tasks, consider using DeepSeek-VL or similar.
        
        Args:
            prompt: Text prompt
            image_base64: Base64-encoded image (will be noted but not processed)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # DeepSeek-V3 doesn't have native vision - note the image in prompt
        messages = [
            {
                "role": "user",
                "content": f"[Note: An image was provided but this model processes text only. Analyzing based on the text description.]\n\n{prompt}",
            }
        ]
        
        return self.generate(messages, max_tokens=max_tokens, temperature=temperature)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
            except Exception:
                # Rough estimate: ~4 chars per token
                return len(text) // 4
        
        return len(self._tokenizer.encode(text))


class BaseAgent:
    """Base class for all Brainnet agents using DeepSeek-V3."""

    def __init__(
        self,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.llm = DeepSeekV3Client(
            backend=backend,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def _create_system_message(self, content: str) -> dict:
        """Create a system message."""
        return {"role": "system", "content": content}

    def _create_user_message(self, content: str) -> dict:
        """Create a user message."""
        return {"role": "user", "content": content}

    def _create_assistant_message(self, content: str) -> dict:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

