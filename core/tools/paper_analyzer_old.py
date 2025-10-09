import os
import re
import json
import math
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.htmlrag.splitter import universal_html_parser_to_markdown, split_large_chunks, get_token_count

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")



def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _sanitize(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', s)

def _bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w\-]+\b", (text or "").lower(), flags=re.UNICODE)

def _top_k_indices(scores: np.ndarray, k: int) -> List[int]:
    if scores.size == 0:
        return []
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()

def _rrf_fuse(ranks_a: Dict[int, int], ranks_b: Dict[int, int], k: int = 60, alpha: float = 0.5) -> List[int]:
    docs = set(ranks_a.keys()) | set(ranks_b.keys())
    scores = {}
    for d in docs:
        ra = ranks_a.get(d, 10**9)
        rb = ranks_b.get(d, 10**9)
        scores[d] = alpha * (1.0 / (k + ra)) + (1 - alpha) * (1.0 / (k + rb))
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]

def _as_ranks(order_ids: List[int]) -> Dict[int, int]:
    return {doc_id: i + 1 for i, doc_id in enumerate(order_ids)}


class EmbeddingBackend:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = device):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)
        self.tokenizer = self.model.tokenizer
        try:
            self.dim = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            test_vec = self.model.encode(["test"], normalize_embeddings=True)
            self.dim = int(test_vec.shape[-1])

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return vecs.astype(np.float32)


class QwenReranker:
    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        device: str = device,
        max_length: int = 8192,
        instruction: Optional[str] = 'Given a web search query, retrieve relevant passages that answer the query',
        use_half: bool = True,
        use_flash_attn2: bool = False
    ):
        dtype = torch.float16 if (use_half and device == "cuda") else None
        attn_impl = "flash_attention_2" if (use_flash_attn2 and device == "cuda") else None

        tok_kwargs = {"padding_side": "left", "trust_remote_code": True}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

        model_kwargs = {"trust_remote_code": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device).eval()
        self.device = device
        self.max_length = int(max_length)

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.instruction = instruction

    def _format_instruction(self, instruction: Optional[str], query: str, doc: str) -> str:
        inst = instruction if instruction is not None else self.instruction
        return f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        inner_max = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=inner_max
        )
        for i, ids in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ids + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for k in inputs:
            inputs[k] = inputs[k].to(self.model.device)
        return inputs

    @torch.no_grad()
    def score(self, query: str, docs: List[str], batch_size: int = 16) -> np.ndarray:
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        scores = []
        for i in range(0, len(docs), batch_size):
            pairs = [self._format_instruction(self.instruction, query, d) for d in docs[i:i+batch_size]]
            toks = self._process_inputs(pairs)
            out = self.model(**toks)
            last_logits = out.logits[:, -1, :]
            true_vec = last_logits[:, self.token_true_id]
            false_vec = last_logits[:, self.token_false_id]
            batch_scores = torch.stack([false_vec, true_vec], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            yes_scores = batch_scores[:, 1].exp().detach().float().cpu().numpy()
            scores.append(yes_scores)
        return np.concatenate(scores, axis=0).astype(np.float32)


def _build_or_load_bm25(index_dir: str, docs: List[str]) -> BM25Okapi:
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning(f"Не удалось загрузить BM25, пересобираю: {e}")

    tokens = [_bm25_tokenize(t) for t in docs]
    bm25 = BM25Okapi(tokens)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    return bm25

def _build_or_load_vec_index(index_dir: str, encoder: EmbeddingBackend, docs: List[str]) -> Tuple[faiss.IndexFlatIP, Optional[np.ndarray]]:
    _ensure_dir(index_dir)
    emb_path = os.path.join(index_dir, "embeddings.npy")
    faiss_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.json")

    def _emb_shape_ok() -> bool:
        if not os.path.exists(emb_path):
            return False
        try:
            mm = np.load(emb_path, mmap_mode="r")
            ok = (mm.shape[1] == encoder.dim and mm.shape[0] == len(docs))
            del mm
            return ok
        except Exception:
            return False

    need_build_emb = not _emb_shape_ok()

    if os.path.exists(faiss_path):
        try:
            index = faiss.read_index(faiss_path)
            if index.d == encoder.dim and index.ntotal == len(docs):
                return index, None  # важно: embeddings не грузим
        except Exception:
            pass

    if need_build_emb:
        vecs = encoder.encode(docs, batch_size=64, normalize=True)
        np.save(emb_path, vecs)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"embed_model": encoder.model_name, "dim": encoder.dim, "count": len(docs)}, f)
    else:
        vecs = np.load(emb_path, mmap_mode="r")

    index = faiss.IndexFlatIP(encoder.dim)
    index.add(np.asarray(vecs, dtype=np.float32))
    faiss.write_index(index, faiss_path)
    return index, None

# def _build_or_load_vec_index(index_dir: str, encoder: EmbeddingBackend, docs: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
#     _ensure_dir(index_dir)
#     emb_path = os.path.join(index_dir, "embeddings.npy")
#     faiss_path = os.path.join(index_dir, "faiss.index")
#     meta_path = os.path.join(index_dir, "meta.json")

#     need_build_emb = True
#     if os.path.exists(emb_path):
#         try:
#             emb = np.load(emb_path)
#             if emb.shape[1] == encoder.dim and emb.shape[0] == len(docs):
#                 need_build_emb = False
#         except Exception:
#             pass

#     if need_build_emb:
#         vecs = encoder.encode(docs, batch_size=64, normalize=True)
#         np.save(emb_path, vecs)
#         with open(meta_path, "w", encoding="utf-8") as f:
#             json.dump({"embed_model": encoder.model_name, "dim": encoder.dim, "count": len(docs)}, f)
#     else:
#         vecs = np.load(emb_path)

#     # Всегда FAISS
#     rebuild_faiss = True
#     if os.path.exists(faiss_path):
#         try:
#             index = faiss.read_index(faiss_path)
#             if index.d == encoder.dim and index.ntotal == len(docs):
#                 rebuild_faiss = False
#                 return index, vecs
#         except Exception:
#             pass

#     index = faiss.IndexFlatIP(encoder.dim)
#     index.add(vecs)  # косинус через IP, т.к. нормализовано
#     faiss.write_index(index, faiss_path)
#     return index, vecs

def _vec_search(query: str, encoder: EmbeddingBackend, faiss_index: faiss.IndexFlatIP, top_k: int) -> List[int]:
    q = encoder.encode([query], normalize=True)  # [1, dim]
    D, I = faiss_index.search(q, top_k)
    return [int(i) for i in I[0] if i >= 0]

def _bm25_search(query: str, bm25: BM25Okapi, top_k: int) -> List[int]:
    q_tokens = _bm25_tokenize(query)
    scores = np.array(bm25.get_scores(q_tokens))
    return _top_k_indices(scores, top_k)

# ---------------------- КОНТЕКСТ ----------------------

def _format_context(chosen: List[Tuple[int, Dict[str, Any]]], max_ctx_tokens: int, tokenizer: Any) -> str:
    parts = []
    total = 0
    for _, ch in chosen:
        block = ch["content"].strip()
        header = ch.get("metadata", {}).get("source", "")
        piece = f"[{header}]\n{block}"
        toks = get_token_count(piece, tokenizer)
        if total + toks > max_ctx_tokens:
            break
        parts.append(piece)
        total += toks
    return "\n\n---\n\n".join(parts)

# ---------------------- ВЫЗОВ LLM ----------------------

def _call_llm_direct(llm_service, prompt: str, system: Optional[str] = None) -> str:
    if hasattr(llm_service, "invoke") and callable(llm_service.invoke):
        # Возможно, ваш invoke не принимает system, поэтому просто передаем prompt
        return llm_service.invoke(prompt)
    raise RuntimeError("Не удалось вызвать LLM: неизвестный интерфейс у llm_service")

# ---------------------- ОСНОВНАЯ ФУНКЦИЯ ----------------------

from core.data.project_store import ProjectStore
from core.llm_services import get_llm_service
# --- Singletons ---
_ENCODER_SINGLETON = None
_RERANKER_SINGLETON = None

_INDEX_CACHE = {}
_BM25_CACHE = {}

def get_encoder_singleton() -> EmbeddingBackend:
    global _ENCODER_SINGLETON
    if _ENCODER_SINGLETON is None:
        _ENCODER_SINGLETON = EmbeddingBackend(EMBED_MODEL, device=device)
    return _ENCODER_SINGLETON

def get_reranker_singleton() -> QwenReranker:
    global _RERANKER_SINGLETON
    if _RERANKER_SINGLETON is None:
        _RERANKER_SINGLETON = QwenReranker(RERANK_MODEL, device=device, max_length=8192, use_half=True, use_flash_attn2=False)
    return _RERANKER_SINGLETON


def analyze_paper(paper_id: str, question: str, project_id: str) -> str:
    """
    Режимы поиска: bm25 | emb | hybrid (RRF).
    Реранкер: включается флагом.
    """
    try:
        project_store = ProjectStore(storage_path="projects_data")
        project = project_store.get_project(project_id)
        if not project:
            raise FileNotFoundError(f"Проект '{project_id}' не найден.")
        paper = next((p for p in project.get('papers', []) if p.get('id') == paper_id), None)
        if not paper or not paper.get('html_path'):
            raise FileNotFoundError(f"Статья '{paper_id}' не найдена.")
        full_path = os.path.join(project_store.storage_path, paper['html_path'])
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Файл HTML не найден: {full_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Настройки
        cfg = dict(project.get("rag_settings", {}))
        chunk_size_tokens = int(cfg.get("chunk_size_tokens", 256))
        retrieval = str(cfg.get("retrieval", "emb")).lower()
        top_k = int(cfg.get("top_k", 5))
        hybrid_alpha = float(cfg.get("hybrid_alpha", 0.75))
        hybrid_rrf_k = int(cfg.get("hybrid_rrf_k", 60))
        max_context_tokens = int(cfg.get("max_context_tokens", 1600))
        use_reranker = bool(cfg.get("use_reranker", False))
        reranker_top_n = int(cfg.get("reranker_top_n", max(25, top_k * 3)))
        rebuild_index = bool(cfg.get("rebuild_index", False))

        # Этап 1: чанкование
        encoder = get_encoder_singleton()
        embedder_tokenizer = encoder.tokenizer
        initial_chunks = universal_html_parser_to_markdown(html_content)
        final_chunks = split_large_chunks(semantic_chunks=initial_chunks, chunk_size=chunk_size_tokens, tokenizer=embedder_tokenizer)
        if not final_chunks:
            return "Не удалось получить содержимое статьи (пусто после парсинга/чанкования)."

        docs = [ch["content"] for ch in final_chunks]
        metas = [ch["metadata"] for ch in final_chunks]

        base_index_dir = os.path.join(
            project_store.storage_path,
            "indices",
            _sanitize(project_id),
            _sanitize(paper_id),
            _sanitize(EMBED_MODEL)
        )
        _ensure_dir(base_index_dir)

        # mapping (проверим длину — если изменилось, обновим)
        mapping_path = os.path.join(base_index_dir, "doc_mapping.pkl")
        need_update_mapping = True
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "rb") as f:
                    m = pickle.load(f)
                if len(m.get("docs", [])) == len(docs):
                    need_update_mapping = False
            except Exception:
                pass
        if rebuild_index or need_update_mapping:
            with open(mapping_path, "wb") as f:
                pickle.dump({"docs": docs, "metas": metas}, f)

        if rebuild_index:
            for fn in ("embeddings.npy", "faiss.index", "meta.json"):
                p = os.path.join(base_index_dir, fn)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        # в analyze_paper перед построением индекса:
        idx_key = (project_id, paper_id, EMBED_MODEL)

        faiss_index = _INDEX_CACHE.get(idx_key)
        if faiss_index is None:
            faiss_index, _ = _build_or_load_vec_index(base_index_dir, encoder, docs)
            _INDEX_CACHE[idx_key] = faiss_index

        bm25 = None
        if retrieval in {"bm25", "hybrid"}:
            bm25 = _BM25_CACHE.get(idx_key)
            if bm25 is None or rebuild_index:
                bm25 = _build_or_load_bm25(base_index_dir, docs)
                _BM25_CACHE[idx_key] = bm25

        # 4) Поиск кандидатов
        if retrieval == "emb":
            candidates = _vec_search(question, encoder, faiss_index, top_k=(reranker_top_n if use_reranker else top_k))
        elif retrieval == "bm25":
            candidates = _bm25_search(question, bm25, top_k=(reranker_top_n if use_reranker else top_k))
        elif retrieval == "hybrid":
            k_cand = max(reranker_top_n, top_k * 4)
            emb_ids = _vec_search(question, encoder, faiss_index, top_k=k_cand)
            bm25_ids = _bm25_search(question, bm25, top_k=k_cand)
            fused = _rrf_fuse(_as_ranks(emb_ids), _as_ranks(bm25_ids), k=hybrid_rrf_k, alpha=hybrid_alpha)
            candidates = fused[: (reranker_top_n if use_reranker else top_k)]
        else:
            return f"Неизвестный тип поиска: {retrieval}"

        # Уникализация
        seen, uniq = set(), []
        for i in candidates:
            if 0 <= i < len(docs) and i not in seen:
                seen.add(i)
                uniq.append(i)
        candidates = uniq

        # 5) Реранкер (опционально)
        if use_reranker and candidates:
            try:
                reranker = get_reranker_singleton()
                cand_texts = [docs[i] for i in candidates]
                scores = reranker.score(question, cand_texts, batch_size=4)
                order = np.argsort(-scores).tolist()
                chosen_ids = [candidates[i] for i in order[:top_k]]
            except Exception as e:
                log.warning(f"Реранкер не применился: {e}")
                chosen_ids = candidates[:top_k]
        else:
            chosen_ids = candidates[:top_k]

        # 6) Формирование контекста
        chosen_pairs = [(i, {"content": docs[i], "metadata": metas[i]}) for i in chosen_ids]
        context = _format_context(chosen_pairs, max_ctx_tokens=max_context_tokens, tokenizer=embedder_tokenizer)

        system_msg = (
            "You are an assistant for analyzing scientific papers. Answer in English, strictly based on the provided context. "
            "If the context is insufficient, state that. Where possible, cite sources in square brackets."
        )
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n```\n{context}\n```\n\n"
            "Provide a precise and concise answer:"
        )

        llm_service = get_llm_service()
        # Примечание: если ваш _call_llm_direct игнорирует system, можно склеить его с prompt вручную.
        answer = _call_llm_direct(llm_service, prompt=prompt, system=system_msg)
        log.info("Анализ завершён.")
        return answer

    except Exception as e:
        log.error(f"Непредвиденная ошибка во время выполнения analyze_paper: {e}", exc_info=True)
        return f"Ошибка анализа: Произошла непредвиденная ошибка. Детали: {e}"