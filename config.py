import os
from dotenv import load_dotenv

load_dotenv()

def _as_bool(val: str, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

class Settings:
    LLM_PROVIDER: str

    LLAMA_CPP_MODEL_PATH: str
    LLAMA_CPP_CTX: int
    LLAMA_CPP_THREADS: int
    LLAMA_CPP_GPU_LAYERS: int
    LLAMA_CPP_CHAT_FORMAT: str
    LLAMA_CPP_TEMPERATURE: float
    LLAMA_CPP_MAX_TOKENS: int
    LLAMA_CPP_TOP_P: float
    LLAMA_CPP_TOP_K: int


    RAG_EMBED_MODEL: str
    RAG_RERANK_MODEL: str
    RAG_CHUNK_SIZE_TOKENS: int
    RAG_RETRIEVAL: str
    RAG_TOP_K: int
    RAG_HYBRID_ALPHA: float
    RAG_HYBRID_RRF_K: int
    RAG_MAX_CONTEXT_TOKENS: int
    RAG_USE_RERANKER: bool
    RAG_RERANKER_TOP_N: int

    def __init__(self):
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "llama.cpp")

        self.LLAMA_CPP_MODEL_PATH = os.getenv("LLAMA_CPP_MODEL_PATH", "models/Qwen3-1.7B-Q8_0.gguf")
        self.LLAMA_CPP_CTX = int(os.getenv("LLAMA_CPP_CTX", "4096"))
        self.LLAMA_CPP_THREADS = int(os.getenv("LLAMA_CPP_THREADS", str(os.cpu_count() or 4)))
        self.LLAMA_CPP_GPU_LAYERS = int(os.getenv("LLAMA_CPP_GPU_LAYERS", "0"))
        self.LLAMA_CPP_CHAT_FORMAT = os.getenv("LLAMA_CPP_CHAT_FORMAT", "auto")
        self.LLAMA_CPP_TEMPERATURE = float(os.getenv("LLAMA_CPP_TEMPERATURE", "0.6"))
        self.LLAMA_CPP_MAX_TOKENS = int(os.getenv("LLAMA_CPP_MAX_TOKENS", "500"))
        self.LLAMA_CPP_TOP_P = float(os.getenv("LLAMA_CPP_TOP_P", "0.95"))
        self.LLAMA_CPP_TOP_K = int(os.getenv("LLAMA_CPP_TOP_K", "20"))


        self.RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        self.RAG_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")

        self.RAG_CHUNK_SIZE_TOKENS = int(os.getenv("RAG_CHUNK_SIZE_TOKENS", "256"))
        self.RAG_RETRIEVAL = os.getenv("RAG_RETRIEVAL", "hybrid").lower()
        self.RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
        self.RAG_HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.75"))
        self.RAG_HYBRID_RRF_K = int(os.getenv("RAG_HYBRID_RRF_K", "60"))
        self.RAG_MAX_CONTEXT_TOKENS = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "1600"))
        self.RAG_USE_RERANKER = _as_bool(os.getenv("RAG_USE_RERANKER", "false"))
        self.RAG_RERANKER_TOP_N = int(os.getenv("RAG_RERANKER_TOP_N", "25"))

settings = Settings()