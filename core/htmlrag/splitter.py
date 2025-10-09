import re
import math
from typing import Optional
from bs4 import BeautifulSoup, NavigableString
from transformers import AutoTokenizer
from typing import List, Dict, Any
from markdownify import markdownify as md
from collections import defaultdict
from core.htmlrag.html_utils import clean_html


def get_token_count(text: str, tokenizer: Any) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

MATH_SENTINEL = "§MATH§"

HEADING_TAG_RE = re.compile(r'^h[1-6]$')
EXCLUDE_SECTIONS = {'references', 'bibliography', 'acknowledgments', 'acknowledgements'}

def normalize_heading(text: str) -> str:
    s = re.sub(r'\s+', ' ', (text or '')).strip().lower()
    s = re.sub(r'[.:;—–-]+$', '', s)
    return s

def clip_at_first_lower_md_heading(md_text: str, level: int) -> str:
    if not md_text or level >= 6:
        return md_text
    pattern = re.compile(r'(?m)^\s*#{' + str(level + 1) + r',6}\s+.+$')
    m = pattern.search(md_text)
    if m:
        return md_text[:m.start()].rstrip()
    return md_text


def restore_math(text, placeholders):
    if not text: return text
    import base64
    for k, v in placeholders.items():
        text = text.replace(k,v)
    return text

EQNUM_RE = re.compile(r'^\(?\d+[a-z]?\)?$')


def postprocess_markdown_block(md_text: str) -> str:
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)
    md_text = re.sub(r'(?m)([^\n])\n\s*\n(\s*\|)', r'\1\n\2', md_text)
    md_text = re.sub(
        r'(?m)(:\s*)\n\s*\n(\s*(?:[-*+]|\d+\.)\s+)',
        r'\1\n\2', md_text
    )
    return md_text

TABLE_HEADER_SEP_RE = re.compile(
    r'^\s*(?=.*\|)\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*\|?\s*$'
)

def _is_table_header_sep(line: str) -> bool:
    return bool(TABLE_HEADER_SEP_RE.match(line))

def fix_empty_md_table_headers(md_text: str) -> str:
    if not md_text:
        return md_text

    def _split_md_row_to_cells(line: str) -> list:
        line = (line or '').strip().strip('|')
        if not line:
            return []
        return [c.strip() for c in line.split('|')]

    def _row_has_nonempty_text(line: str) -> bool:
        for c in _split_md_row_to_cells(line):
            if re.search(r'\S', c) and not re.fullmatch(r'[-:\s]*', c):
                return True
        return False

    def _make_md_header_sep_like(header_line: str) -> str:
        n = max(1, len(_split_md_row_to_cells(header_line)))
        return '| ' + ' | '.join(['---'] * n) + ' |'

    lines = md_text.splitlines()
    out = []
    i, n = 0, len(lines)

    def _is_table_like_line(ln: str) -> bool:
        return ('|' in (ln or '')) or _is_table_header_sep(ln or '')

    while i < n:
        line = lines[i]
        if _is_table_like_line(line):
            j = i
            block = []
            while j < n and _is_table_like_line(lines[j]) and lines[j].strip() != '':
                block.append(lines[j])
                j += 1

            pipe_lines = sum(1 for ln in block if '|' in ln)
            has_sep = any(_is_table_header_sep(ln) for ln in block[1:])
            if pipe_lines >= 2 or has_sep:
                sep_idx = None
                for k in range(1, len(block)):
                    if _is_table_header_sep(block[k]) and ('|' in block[k - 1]):
                        sep_idx = k
                        break

                header_idx = None
                if sep_idx is not None:
                    header_idx = sep_idx - 1
                else:
                    for k in range(len(block)):
                        if '|' in block[k]:
                            header_idx = k
                            break

                if header_idx is not None:
                    if not _row_has_nonempty_text(block[header_idx]):
                        start = (sep_idx + 1) if sep_idx is not None else (header_idx + 1)
                        repl_idx = None
                        for k in range(start, len(block)):
                            if '|' in block[k] and _row_has_nonempty_text(block[k]):
                                repl_idx = k
                                break
                        if repl_idx is not None:
                            block[header_idx] = block[repl_idx]
                            del block[repl_idx]

                    sep_line_needed = _make_md_header_sep_like(block[header_idx])
                    if header_idx + 1 >= len(block) or not _is_table_header_sep(block[header_idx + 1]):
                        block.insert(header_idx + 1, sep_line_needed)
                    else:
                        block[header_idx + 1] = sep_line_needed

                out.extend(block)
                while j < n and lines[j].strip() == '':
                    out.append(lines[j])
                    j += 1
                i = j
                continue

        out.append(line)
        i += 1

    return "\n".join(out)

def universal_html_parser_to_markdown(html: str) -> List[Dict[str, Any]]:
    cleaned_html, math_ph = clean_html(html)
    soup = BeautifulSoup(cleaned_html, 'html.parser')
    semantic_chunks = []
    article = soup.find('article') or soup.find('main') or soup.find('body') or soup
    if not article:
        return []

    main_headers = article.find_all(HEADING_TAG_RE)

    path_by_level = {i: None for i in range(1, 7)}

    for header in main_headers:
        level = int(header.name[1])
        section_title = restore_math(header.get_text(separator=' ', strip=True), math_ph)
        norm_title = normalize_heading(section_title)

        if any(k in norm_title for k in EXCLUDE_SECTIONS):
            continue

        section_html = []
        for elem in header.find_next_siblings():
            if getattr(elem, "name", None) and HEADING_TAG_RE.match(elem.name) and int(elem.name[1]) <= level:
                break
            section_html.append(str(elem))

        section_md = restore_math(md("".join(section_html), heading_style="ATX").strip(), math_ph)
        section_md = postprocess_markdown_block(section_md)
        section_md = fix_empty_md_table_headers(section_md)

        section_md = clip_at_first_lower_md_heading(section_md, level)

        if not section_md.strip():
            continue

        for L in range(level + 1, 7):
            path_by_level[L] = None
        path_by_level[level] = section_title

        full_path_for_metadata = [t for t in (path_by_level[1], path_by_level[2], path_by_level[3],
                                        path_by_level[4], path_by_level[5], path_by_level[6]) if t]
        source_metadata = " > ".join(full_path_for_metadata)

        if level == 1:
            prefix_path = [path_by_level[1]] if path_by_level[1] else []
        else:
            prefix_path = [t for t in (path_by_level[2], path_by_level[3],
                                path_by_level[4], path_by_level[5], path_by_level[6]) if t]

        full_title = " > ".join(prefix_path)

        semantic_chunks.append({
            "content": section_md,
            "prefix": f"{full_title}\n\n",
            "metadata": {
                "source": f"section: {source_metadata}",
                "level": level,
                "title": section_title,
                "path": full_path_for_metadata
            }
        })

    return semantic_chunks



def _lower_heading_re(level: Optional[int]):
    if isinstance(level, int) and 1 <= level < 6:
        return re.compile(r'(?m)^\s*#{' + str(level + 1) + r',6}\s+.+$')
    return None


CAPTION_RE = re.compile(r"^\s*Table\s+\d+:")

def is_caption_like(text: str) -> bool:
    if not text:
        return False
    return bool(CAPTION_RE.search(text.strip()))


def is_markdown_table_block(block: str) -> bool:
    if not block:
        return False
    lines = [ln for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    if any(_is_table_header_sep(ln) for ln in lines[1:]):
        return True
    pipe_lines = sum(1 for ln in lines if '|' in ln)
    return pipe_lines >= 2



def split_markdown_table_block(block_text: str, cap_tokens: int, tokenizer: Any) -> list:
    raw = block_text.strip('\n')
    if not raw:
        return []

    lines = raw.splitlines()
    n = len(lines)

    sep_idx = None
    header_row_idx = None
    for i in range(1, n):
        if _is_table_header_sep(lines[i]) and ('|' in lines[i-1]):
            sep_idx = i
            header_row_idx = i - 1
            break
    if header_row_idx is None:
        for j in range(n):
            if '|' in lines[j]:
                header_row_idx = j
                break

    if header_row_idx is None:
        return hard_split_tokens(block_text, cap_tokens, overlap_tokens=0, tokenizer=tokenizer)

    t_body_start = (sep_idx + 1) if sep_idx is not None else (header_row_idx + 1)
    t_end = t_body_start
    while t_end < n and ('|' in lines[t_end] or _is_table_header_sep(lines[t_end])):
        t_end += 1

    pre_lines = lines[:header_row_idx]                
    header_lines = [lines[header_row_idx]]
    if sep_idx is not None:
        header_lines.append(lines[sep_idx])
    body_lines = lines[t_body_start:t_end]
    post_lines = lines[t_end:]                        


    parts = []

    pre_text = "\n".join(pre_lines).strip()
    base_common_lines = []
    if pre_text:
        base_common_lines.append(pre_text)
    base_common_lines.extend(header_lines)
    base_text = "\n".join(base_common_lines).strip()

    if base_text and get_token_count(base_text, tokenizer) > cap_tokens:
        for p in hard_split_tokens(base_text, cap_tokens, overlap_tokens=0, tokenizer=tokenizer):
            if p.strip():
                parts.append(p.strip())
        base_text = "\n".join(header_lines).strip()

    def pack_part(base_text_: str, extra_lines_iter):
        taken = 0
        collected = [base_text_] if base_text_ else []
        for ln in extra_lines_iter:
            candidate = "\n".join(collected + [ln]) if collected else ln
            if get_token_count(candidate, tokenizer) <= cap_tokens:
                collected.append(ln)
                taken += 1
            else:
                break
        return ("\n".join(collected).strip(), taken)

    if not body_lines:
        if base_text:
            parts.append(base_text)
        if post_lines:
            post_text = "\n".join(post_lines).strip()
            if post_text:
                if parts and get_token_count(parts[-1] + "\n\n" + post_text, tokenizer) <= cap_tokens:
                    parts[-1] = (parts[-1] + "\n\n" + post_text).strip()
                else:
                    parts.extend([p for p in hard_split_tokens(post_text, cap_tokens, overlap_tokens=0, tokenizer=tokenizer) if p.strip()])
        return parts or [raw]

    idx = 0
    while idx < len(body_lines):
        part_text, taken = pack_part(base_text, body_lines[idx:])
        if taken == 0:
            room = max(1, cap_tokens - get_token_count(base_text, tokenizer)) if base_text else cap_tokens
            row = body_lines[idx]
            if room <= 0:
                room = max(1, cap_tokens - get_token_count(base_text, tokenizer))
            row_parts = hard_split_tokens(row, room, overlap_tokens=0, tokenizer=tokenizer)
            for rp in row_parts:
                assembled = (base_text + "\n" + rp) if base_text else rp
                if assembled.strip():
                    parts.append(assembled.strip())
            idx += 1
        else:
            parts.append(part_text)
            idx += taken

    if post_lines:
        post_text = "\n".join(post_lines).strip()
        if post_text:
            candidate = (parts[-1] + "\n\n" + post_text) if parts else post_text
            if parts and get_token_count(candidate, tokenizer) <= cap_tokens:
                parts[-1] = candidate.strip()
            else:
                parts.extend([p for p in hard_split_tokens(post_text, cap_tokens, overlap_tokens=0, tokenizer=tokenizer) if p.strip()])

    return parts or [raw]

SENTENCE_SPLIT_RE = re.compile(r'(?s).+?(?:[.!?](?=\s|$)|$)')

def split_into_sentences(text: str) -> list:
    sentences = [s for s in SENTENCE_SPLIT_RE.findall(text) if s and s.strip()]
    return sentences if sentences else ([text] if text.strip() else [])

def chunk_units_with_overlap(units: list, cap_tokens: int, overlap_tokens: int, joiner: str, tokenizer: Any) -> list:
    parts = []
    n = len(units)
    i = 0

    overlap_tokens = max(0, overlap_tokens)
    cap_tokens = max(1, cap_tokens)
    if overlap_tokens >= cap_tokens:
        overlap_tokens = max(0, cap_tokens // 2)

    while i < n:
        j = i
        best_text = ""
        while j < n:
            candidate = joiner.join(units[i:j+1])
            if get_token_count(candidate, tokenizer) <= cap_tokens:
                best_text = candidate
                j += 1
            else:
                break

        if not best_text:
            token_parts = hard_split_tokens(units[i], cap_tokens, overlap_tokens, tokenizer)
            parts.extend([p for p in token_parts if p.strip()])
            i += 1
            continue

        parts.append(best_text)

        used = j - i 
        tail_units = 0
        for k in range(1, used + 1):
            tail_candidate = joiner.join(units[j - k:j])
            if get_token_count(tail_candidate, tokenizer) >= overlap_tokens:
                tail_units = k
                break

        if tail_units == 0:
            next_i = j
        else:
            next_i = j - tail_units
            if next_i <= i:
                next_i = max(i + 1, j - max(1, tail_units - 1))

        i = next_i

    return parts


def hard_split_tokens(text: str, cap_tokens: int, overlap_tokens: int, tokenizer: Any) -> list:
    if not text:
        return []
    cap_tokens = max(1, cap_tokens)
    overlap_tokens = max(0, overlap_tokens)
    if overlap_tokens >= cap_tokens:
        overlap_tokens = max(0, cap_tokens // 2)

    toks = tokenizer.encode(text, add_special_tokens=False)
    parts = []
    n = len(toks)
    if n <= cap_tokens:
        return [text]

    stride = max(1, cap_tokens - overlap_tokens)
    start = 0
    while start < n:
        end = min(start + cap_tokens, n)
        part_toks = toks[start:end]
        part_text = tokenizer.decode(part_toks, skip_special_tokens=True)
        if part_text.strip():
            parts.append(part_text)
        if end >= n:
            break
        start = start + stride

    return parts

SENT_NORM_RE = re.compile(r'\s+')

def _norm_for_compare(s: str) -> str:
    return SENT_NORM_RE.sub(' ', (s or '').strip())

def strip_leading_duplicate_sentences(prev_text: str, curr_text: str, max_check: int = 5) -> str:
    if not prev_text or not curr_text:
        return curr_text

    prev_sents = [s for s in smart_sent_split(prev_text) if s.strip()]
    curr_sents = [s for s in smart_sent_split(curr_text) if s.strip()]
    if not prev_sents or not curr_sents:
        return curr_text

    prev_norm = [_norm_for_compare(x) for x in prev_sents]
    curr_norm = [_norm_for_compare(x) for x in curr_sents]

    cut = 0
    max_k = min(max_check, len(prev_norm), len(curr_norm))
    for k in range(max_k, 0, -1):
        if prev_norm[-k:] == curr_norm[:k]:
            cut = k
            break

    if cut:
        rest = curr_sents[cut:]
        return (' '.join(rest)).lstrip()
    return curr_text

def strip_char_overlap_suffix_prefix(prev_text: str, curr_text: str, min_overlap_chars: int = 20, limit: int = 2000) -> str:
    if not prev_text or not curr_text:
        return curr_text

    a = prev_text[-limit:]
    b = curr_text[:limit]
    max_len = min(len(a), len(b))

    for L in range(max_len, min_overlap_chars - 1, -1):
        if a.endswith(b[:L]):
            return curr_text[L:].lstrip()
    return curr_text

def smart_sent_split(text: str) -> list:
    if not text:
        return []
    units = []
    pat = re.compile(r'(?s)\$\$.*?\$\$')
    pos = 0
    for m in pat.finditer(text):
        before = text[pos:m.start()]
        if before.strip():
            units.extend(split_into_sentences(before.strip()))
        units.append(m.group(0))  # сам math-блок — отдельный юнит
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        units.extend(split_into_sentences(tail.strip()))
    return [u for u in units if u and u.strip()]

def split_long_block_with_overlap(text: str, cap_tokens: int, overlap_tokens: int, tokenizer: Any) -> list:
    if not text or get_token_count(text, tokenizer) <= cap_tokens:
        return [text] if text else []

    sentences = smart_sent_split(text)
    if len(sentences) > 1:
        parts = chunk_units_with_overlap(sentences, cap_tokens, overlap_tokens, joiner=' ', tokenizer=tokenizer)
        if all(get_token_count(p, tokenizer) <= cap_tokens for p in parts):
            return parts

    if '\n' in text:
        lines = text.split('\n')
        structured = any(re.match(r'^\s*(`{3,}|[-*+]\s+|\d+\.\s+)', ln) for ln in lines)
        parts = chunk_units_with_overlap(lines, cap_tokens, overlap_tokens, joiner='\n', tokenizer=tokenizer)
        if structured and all(get_token_count(p, tokenizer) <= cap_tokens for p in parts):
            return parts
        if all(get_token_count(p, tokenizer) <= cap_tokens for p in parts):
            return parts

    return hard_split_tokens(text, cap_tokens, overlap_tokens, tokenizer)

def strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s

def merge_adjacent_small_text_chunks(chunks: List[Dict[str, Any]], chunk_size: int, tokenizer: Any) -> List[Dict[str, Any]]:
    if not chunks:
        return chunks
    out = []
    for ch in chunks:
        if (
            out
            and ch['metadata'].get('source') == out[-1]['metadata'].get('source')
            and ch['metadata'].get('kind') == 'text'
            and out[-1]['metadata'].get('kind') == 'text'
            and ch['metadata'].get('prefix') == out[-1]['metadata'].get('prefix')
        ):
            prefix = ch['metadata']['prefix']
            a_body = strip_prefix(out[-1]['content'], prefix)
            b_body = strip_prefix(ch['content'], prefix)

            b_body = strip_leading_duplicate_sentences(a_body, b_body)
            b_body = strip_char_overlap_suffix_prefix(a_body, b_body)

            if not b_body.strip():
                continue

            merged_body = (a_body.rstrip() + "\n\n" + b_body.lstrip()).strip()
            candidate_text = prefix + merged_body
            if get_token_count(candidate_text, tokenizer) <= chunk_size:
                out[-1]['content'] = candidate_text
                continue
        out.append(ch)
    return out

def split_large_chunks(
    semantic_chunks: List[Dict[str, Any]],
    chunk_size: int, tokenizer: Any
) -> List[Dict[str, Any]]:
    final_chunks = []
    MIN_OVERLAP = int(math.ceil(0.2 * chunk_size))

    for chunk_data in semantic_chunks:
        prefix = chunk_data["prefix"]
        content = chunk_data["content"]
        level = chunk_data.get("metadata", {}).get("level")
        meta_base = {**chunk_data["metadata"]}

        prefix_tokens = get_token_count(prefix, tokenizer)
        effective_chunk_size = max(1, chunk_size - prefix_tokens)

        lower_re = _lower_heading_re(level)
        if lower_re:
            m = lower_re.search(content)
            if m:
                content = content[:m.start()].rstrip()

        if not content.strip():
            continue

        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        current_group: List[str] = []

        def flush_current_group():
            nonlocal current_group
            if current_group:
                body = "\n\n".join(current_group).strip()
                if body:
                    final_chunks.append({
                        "content": prefix + body,
                        "metadata": {**meta_base, "kind": "text", "prefix": prefix}
                    })
                current_group = []

        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]

            if lower_re and lower_re.search(paragraph):
                flush_current_group()
                break

            p_tokens = get_token_count(paragraph, tokenizer)

            if is_markdown_table_block(paragraph) and p_tokens > effective_chunk_size:
                caption_before = None
                if current_group and is_caption_like(current_group[-1]):
                    caption_before = current_group.pop()

                flush_current_group()

                full_table_block_lines = []
                if caption_before:
                    full_table_block_lines.append(caption_before)
                full_table_block_lines.append(paragraph)

                caption_after = None
                if i + 1 < len(paragraphs) and is_caption_like(paragraphs[i + 1]):
                    caption_after = paragraphs[i + 1]
                    i += 1
                if caption_after:
                    full_table_block_lines.append(caption_after)

                full_table_block_text = "\n".join(full_table_block_lines)

                table_parts = split_markdown_table_block(full_table_block_text, effective_chunk_size, tokenizer)
                for tp in table_parts:
                    if tp.strip():
                        final_chunks.append({
                            "content": prefix + tp.strip(),
                            "metadata": {**meta_base, "kind": "table", "prefix": prefix}
                        })
                
                i += 1
                continue

            if p_tokens > effective_chunk_size:
                flush_current_group()
                parts = split_long_block_with_overlap(
                    paragraph,
                    cap_tokens=effective_chunk_size,
                    overlap_tokens=MIN_OVERLAP,
                    tokenizer=tokenizer
                )
                for part in parts:
                    if part.strip():
                        final_chunks.append({
                            "content": prefix + part.strip(),
                            "metadata": {**meta_base, "kind": "text", "prefix": prefix}
                        })
                i += 1
                continue

            tentative = "\n\n".join(current_group + [paragraph])
            if get_token_count(tentative, tokenizer) <= effective_chunk_size:
                current_group.append(paragraph)
            else:
                flush_current_group()
                current_group = [paragraph]

            i += 1

        flush_current_group()

    final_chunks = merge_adjacent_small_text_chunks(final_chunks, chunk_size=chunk_size, tokenizer=tokenizer)

    source_totals = defaultdict(int)
    for ch in final_chunks:
        source_totals[ch['metadata']['source']] += 1
    source_counts = defaultdict(int)
    for chunk in final_chunks:
        source = chunk['metadata']['source']
        total = source_totals[source]
        if total > 1:
            source_counts[source] += 1
            chunk['metadata']['part'] = source_counts[source]
            chunk['metadata']['total_parts'] = total

    return final_chunks