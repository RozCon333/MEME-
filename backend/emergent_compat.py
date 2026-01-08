"""
Compatibility layer for emergentintegrations package
Provides LlmChat and ImageGeneration classes
Uses Google Gemini API
"""

import os
from typing import Optional, List
import json

# Import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")


class UserMessage:
    """Simple user message wrapper"""
    def __init__(self, text: str = None, content: str = None):
        # Support both 'text' and 'content' for compatibility
        self.content = text or content
    
    def to_dict(self):
        return {"role": "user", "content": self.content}


class LlmChat:
    """
    Compatible replacement for emergentintegrations.llm.chat.LlmChat
    Supports both OpenAI and Google Gemini
    """
    
    def __init__(self, api_key: Optional[str] = None, session_id: Optional[str] = None, 
                 system_message: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("EMERGENT_LLM_KEY") or os.getenv("OPENAI_API_KEY")
        self.session_id = session_id
        self.system_message = system_message
        # Allow overriding model via environment; use latest Gemini model
        # Available models: models/gemini-2.5-flash, models/gemini-2.5-pro, models/gemini-2.0-flash
        self.model = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
        self.provider = "gemini"
        self.gemini_model = None
        
        if not self.api_key:
            raise ValueError("No API key provided. Set GEMINI_API_KEY environment variable or pass api_key parameter")
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel(self.model)

        
    def with_model(self, provider: str, model: str):
        """Set the Gemini model to use (keeps signature for compatibility).
        Expects a model string compatible with the Gemini client (e.g. "models/text-bison-001")."""
        # Keep the signature for backward compatibility but always use Gemini
        self.model = model
        self.gemini_model = genai.GenerativeModel(self.model)
        return self
    
    def chat(self, prompt: str) -> str:
        """
        Send a chat message and get response using Gemini
        Compatible with emergentintegrations API
        """
        full_prompt = prompt
        if self.system_message:
            full_prompt = f"{self.system_message}\n\n{prompt}"

        try:
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.9,
                    "max_output_tokens": 2000,
                }
            )
            return response.text

        except Exception as e:
            # If the error indicates the model is not found, attempt to list available models
            err_str = str(e).lower()
            if "not found" in err_str or "notfound" in err_str or "models/" in err_str:
                try:
                    available = genai.list_models()
                    raise RuntimeError(f"Model '{self.model}' not found or not supported by this API version. Available models: {available}")
                except Exception as list_err:
                    raise RuntimeError(f"Model '{self.model}' not found and listing models failed: {list_err}") from e
            # otherwise re-raise the original exception with context
            print(f"Error in LlmChat.chat: {e}")
            raise
    
    async def send_message(self, user_message: 'UserMessage') -> str:
        """
        Async version of chat - using Gemini API
        Compatible with emergentintegrations API
        """
        prompt = user_message.content
        if self.system_message:
            prompt = f"{self.system_message}\n\n{prompt}"

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.9,
                    "max_output_tokens": 2000,
                }
            )
            return response.text

        except Exception as e:
            err_str = str(e).lower()
            if "not found" in err_str or "notfound" in err_str or "models/" in err_str:
                try:
                    available = genai.list_models()
                    raise RuntimeError(f"Model '{self.model}' not found or not supported by this API version. Available models: {available}")
                except Exception as list_err:
                    raise RuntimeError(f"Model '{self.model}' not found and listing models failed: {list_err}") from e
            print(f"Error in LlmChat.send_message: {e}")
            raise


class ImageGeneration:
    """
    Image generation using Gemini API
    Compatible replacement for emergentintegrations image generation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("EMERGENT_LLM_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided for image generation")
        genai.configure(api_key=self.api_key)
    
    async def generate(self, prompt: str, model: str = "gemini-1.5-flash", size: str = "1024x1024", 
                      number_of_images: int = 1) -> List[bytes]:
        """
        Generate images using Gemini
        Note: Gemini doesn't support image generation via API yet
        This returns a placeholder response
        """
        try:
            print(f"Note: Gemini doesn't support image generation via API yet.")
            print(f"Prompt: {prompt}")
            return []
            
        except Exception as e:
            print(f"Error in ImageGeneration.generate: {e}")
            return []


# Keep old name for backwards compatibility
OpenAIImageGeneration = ImageGeneration

