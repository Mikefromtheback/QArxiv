import logging
from abc import ABC, abstractmethod
import re
from typing import Optional
from config import settings

log = logging.getLogger(__name__)
THINK_RE = re.compile(r'(?is)<think>.*?</think>')

def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    return THINK_RE.sub("", text).strip()

class LLMService(ABC):
    @abstractmethod
    def invoke(self, prompt: str, system: Optional[str] = None) -> str: ...



class LlamaCppService(LLMService):
    def __init__(self):
        from llama_cpp import Llama
        chat_format = settings.LLAMA_CPP_CHAT_FORMAT
        if chat_format and chat_format.lower() == "auto":
            chat_format = None 
        try:
            self.llm = Llama(
                model_path=settings.LLAMA_CPP_MODEL_PATH,
                n_ctx=settings.LLAMA_CPP_CTX,
                n_threads=settings.LLAMA_CPP_THREADS,
                n_gpu_layers=settings.LLAMA_CPP_GPU_LAYERS,
                chat_format=chat_format,
            )
        except Exception:
            log.exception("Не удалось инициализировать llama.cpp. Проверь путь к GGUF и колёса для платформы.")
            raise

        self.temperature = settings.LLAMA_CPP_TEMPERATURE
        self.max_tokens = settings.LLAMA_CPP_MAX_TOKENS
        self.top_p = settings.LLAMA_CPP_TOP_P
        self.top_k = settings.LLAMA_CPP_TOP_K

    def invoke(self, prompt: str, system: Optional[str] = None) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            res = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                stream=False,
            )
            return strip_think_blocks(res["choices"][0]["message"]["content"])
        except Exception as e:
            log.exception("Ошибка llama.cpp")
            return f"Ошибка: не удалось получить ответ от LLM (llama.cpp). Детали: {e}"

_LLM_SINGLETON: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    global _LLM_SINGLETON
    if _LLM_SINGLETON is None:
        provider = (settings.LLM_PROVIDER or "ollama").lower()
        if provider in ("llama.cpp", "llama_cpp"):
            _LLM_SINGLETON = LlamaCppService()
    return _LLM_SINGLETON