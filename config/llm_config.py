import os
import streamlit as st # If you want Streamlit API key input here
from typing import List, Dict, Optional

class LLMConfig:
    base_url: str = 'https://api.openai.com/v1' # Or your specific base URL
    model: str = 'gpt-4o' # Or your preferred model like 'gpt-3.5-turbo'
    max_tokens: int = 1024
    default_temp: float = 0.5

# Global LLM instance and API key setup for both local and Streamlit runs
_llm_instance = None
OPENAI_KEY = None

def get_openai_key():
    global OPENAI_KEY
    if OPENAI_KEY is None:
        # Try environment variable first for local runs or deployed apps
        OPENAI_KEY = os.getenv("OPENAI_KEY")
        if OPENAI_KEY is None and 'streamlit' in os.environ.get('PYTHONDONTWRITEBYTECODE', ''):
            # If running in Streamlit, try Streamlit secrets or direct input
            OPENAI_KEY = st.secrets.get("OPENAI_KEY")
            if OPENAI_KEY is None:
                st.warning("OPENAI_KEY not found in environment or Streamlit secrets.")
                OPENAI_KEY = st.text_input("Enter your OpenAI API Key:", type="password")
                if not OPENAI_KEY:
                    st.stop() # Stop execution until key is provided
    return OPENAI_KEY

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        api_key = get_openai_key()
        if not api_key:
            raise ValueError("OpenAI API Key is not configured.")
        from openai import AsyncOpenAI # Import here to avoid circular dependencies if LLM uses other modules
        _llm_instance = AsyncOpenAI(
            base_url=LLMConfig.base_url,
            api_key=api_key
        )
    return _llm_instance

# This will be the LLM class that wraps AsyncOpenAI
class YourLLM:
    def __init__(self, api_key: str):
        self.config = LLMConfig()
        # Use the global get_llm to ensure consistent client
        self.client = get_llm()
        self._is_authenticated = False

    async def check_auth(self) -> bool:
        test_message = [{"role": "user", "content": "test"}]
        try:
            await self.agenerate(test_message, temperature=0.1)
            self._is_authenticated = True
            return True
        except Exception:
            # st.error(f'Authentication Failed: {str(e)}') # Use st.error if in Streamlit context
            return False

    async def agenerate(
            self,
            messages: List[Dict],
            temperature: Optional[float] = None
    ) -> str:
        completion = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.default_temp,
            max_tokens=self.config.max_tokens,
            stream=False
        )
        return completion.choices[0].message.content