"""
P4: Chunking

输入:
  data/cn/combined_content_list_v2.json   v2 per-page
  data/cn/merged_tables.json              P1 跨页表合并 (fragments 引用 v1 global_page+bbox)
  data/cn/merged_paragraphs.json          P2 跨页段落合并 (fragments 引用 v2 global_page+idx)

输出:
  data/cn/chunks.jsonl                    每行一个 chunk
  data/cn/chunking_report.txt             统计报告

规则:
  - 流式遍历 v2 (按 global_page), title level 1/2 维护章节层级树
  - paragraph/list/equation/algorithm -> 聚合进 text chunk
    (并入 merged_paragraph 时整组视为一个单元, 组内后续 fragment 跳过)
  - table/image/chart -> 原子 chunk; merged_table 整组一个 chunk (后续 fragment 跳过)
  - text chunk 目标 ~1024 token (中文按字符估算), 超限时按句子软切 (重叠 1 句)
  - title 自身并入 text 流 (作为章节标题行), 同时更新层级
  - 每个 chunk 带 section_path (章/节路径), global_pages, source_items
"""
import json
import os
import re
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
CN = os.path.join(BASE, "data", "cn")

MAX_TOKENS = 1024
SKIP_TYPES = {"page_header", "page_number", "page_footnote", "page_footer"}
TEXT_TYPES = {"paragraph", "list", "equation_interline", "algorithm"}
ATOMIC_TYPES = {"table", "image", "chart"}


# ---------- 文本抽取 ----------
def est_tokens(text):
    """中文 1 字 ≈ 1 token, ASCII 4 字符 ≈ 1 token。"""
    cjk = sum(1 for c in text if ord(c) > 0x2E7F)
    return cjk + (len(text) - cjk) // 4


def para_text(item):
    parts = []
    for inline in (item.get("content") or {}).get("paragraph_content") or []:
        if isinstance(inline, dict):
            parts.append(str(inline.get("content", "")))
    return "".join(parts)


def item_text(item):
    """把文本类 item 转成纯文本。"""
    t = item.get("type")
    c = item.get("content") or {}
    if t == "paragraph":
        return para_text(item)
    if t == "title":
        return "".join(str(x.get("content", ""))
                       for x in c.get("title_content") or []
                       if isinstance(x, dict))
    if t == "equation_interline":
        return f"$${c.get('math_content', '')}$$"
    if t == "algorithm":
        return str(c.get("algorithm_content", ""))
    if t == "list":
        lines = []
        for li in c.get("list_items") or []:
            li_text = "".join(str(x.get("content", ""))
                              for x in li.get("item_content") or []
                              if isinstance(x, dict))
            lines.append(li_text)
        return "\n".join(lines)
    return ""


def cap_text(items):
    """caption/footnote 元素可能是 str (v1/merged) 或 dict {'type':'text','content':...} (v2)。"""
    out = []
    for x in items or []:
        if isinstance(x, dict):
            out.append(str(x.get("content", "")))
        else:
            out.append(str(x))
    return out


def split_sentences(text):
    """按中文句末标点软切, 保留标点。"""
    parts = re.split(r"(?<=[。！？；])", text)
    return [p for p in parts if p.strip()]


# ---------- chunk 构建 ----------
class ChunkBuilder:
    def __init__(self):
        self.chunks = []
        self.buf = []          # (text, item_ref)
        self.buf_tokens = 0
        self.section_path = []
        self._seq = 0

    def _next_id(self):
        self._seq += 1
        return f"c_{self._seq:06d}"

    def _flush_text(self):
        if not self.buf:
            return
        full = "\n\n".join(t for t, _ in self.buf)
        refs = [r for _, r in self.buf]
        self._emit_text(full, refs)
        self.buf = []
        self.buf_tokens = 0

    def _emit_text(self, text, refs, chunk_type="text", extra=None):
        pages = sorted({r["global_page"] for r in refs})
        chunk = {
            "chunk_id": self._next_id(),
            "chunk_type": chunk_type,
            "text": text,
            "section_path": list(self.section_path),
            "global_pages": pages,
            "source_items": refs,
        }
        if extra:
            chunk.update(extra)
        self.chunks.append(chunk)

    def add_text_item(self, text, ref):
        """聚合文本, 单条超 MAX_TOKENS 按句子软切。"""
        n = est_tokens(text)
        if n > MAX_TOKENS:
            self._flush_text()
            sentences = split_sentences(text)
            cur, cur_refs = [], [ref]
            cur_n = 0
            for s in sentences:
                sn = est_tokens(s)
                if cur and cur_n + sn > MAX_TOKENS:
                    self._emit_text("".join(cur), cur_refs)
                    # 重叠 1 句
                    cur = [cur[-1]] if cur else []
                    cur_n = est_tokens(cur[0]) if cur else 0
                cur.append(s)
                cur_n += sn
            if cur:
                self._emit_text("".join(cur), cur_refs)
            return
        if self.buf and self.buf_tokens + n > MAX_TOKENS:
            self._flush_text()
        self.buf.append((text, ref))
        self.buf_tokens += n

    def add_atomic(self, chunk_type, text, ref, extra=None):
        self._flush_text()
        self._emit_text(text, [ref], chunk_type=chunk_type, extra=extra)

    def set_section(self, level, title):
        if level == 1:
            self.section_path = [title]
        elif level == 2:
            if not self.section_path:
                self.section_path = ["(未分章)"]
            self.section_path = self.section_path[:1] + [title]

    def finish(self):
        self._flush_text()
        return self.chunks


# ---------- main ----------
def main():
    v2 = json.load(open(os.path.join(CN, "combined_content_list_v2.json")))
    mt = json.load(open(os.path.join(CN, "merged_tables.json")))["merged_tables"]
    mp = json.load(open(os.path.join(CN, "merged_paragraphs.json")))["merged_paragraphs"]

    # fragment -> merged group 索引
    # table fragment: (global_page, bbox_tuple) -> group
    tbl_frag2grp = {}
    for g in mt:
        for fr in g["fragments"]:
            tbl_frag2grp[(fr["global_page"], tuple(fr["bbox"]))] = g
    # paragraph fragment: (global_page, idx_in_page) -> group
    par_frag2grp = {}
    for g in mp:
        for fr in g["fragments"]:
            par_frag2grp[(fr["global_page"], fr["idx_in_page"])] = g

    pages = sorted((p for p in v2 if p), key=lambda p: p[0]["global_page"])
    builder = ChunkBuilder()
    tbl_done, par_done = set(), set()

    for page in pages:
        gpage = page[0]["global_page"]
        for idx, it in enumerate(page):
            t = it.get("type")
            if t in SKIP_TYPES:
                continue
            ref = {"global_page": gpage, "idx_in_page": idx, "type": t}

            if t == "title":
                text = item_text(it)
                level = (it.get("content") or {}).get("level", 2)
                builder._flush_text()
                builder.set_section(level, text)
                builder.add_text_item(f"## {text}" if level == 2 else f"# {text}", ref)
                continue

            if t == "table":
                key = (gpage, tuple(it.get("bbox") or [0, 0, 0, 0]))
                grp = tbl_frag2grp.get(key)
                if grp is not None:
                    if grp["merged_id"] in tbl_done:
                        continue  # 后续 fragment 跳过
                    tbl_done.add(grp["merged_id"])
                    cap = " ".join(grp.get("table_caption") or [])
                    fn = "\n".join(grp.get("table_footnote") or [])
                    text = (cap + ("\n" + fn if fn else "")).strip() or "(跨页合并表)"
                    refs = [{"global_page": f["global_page"], "type": "table",
                             "merged_table": grp["merged_id"]} for f in grp["fragments"]]
                    builder._flush_text()
                    builder._emit_text(text, refs, chunk_type="table", extra={
                        "merged_table_id": grp["merged_id"],
                        "table_html": grp["merged_html"],
                        "table_caption": grp.get("table_caption") or [],
                        "table_footnote": grp.get("table_footnote") or [],
                        "img_paths": grp.get("img_paths") or [],
                    })
                else:
                    c = it.get("content") or {}
                    cap = " ".join(cap_text(c.get("table_caption")))
                    fn = "\n".join(cap_text(c.get("table_footnote")))
                    text = (cap + ("\n" + fn if fn else "")).strip() or "(表格)"
                    builder.add_atomic("table", text, ref, extra={
                        "table_html": c.get("html", ""),
                        "table_caption": c.get("table_caption") or [],
                        "table_footnote": c.get("table_footnote") or [],
                        "img_paths": [c.get("image_source", {}).get("path", "")]
                                      if c.get("image_source") else [],
                    })
                continue

            if t in ("image", "chart"):
                c = it.get("content") or {}
                cap = " ".join(cap_text(c.get("image_caption")))
                fn = "\n".join(cap_text(c.get("image_footnote")))
                text = (cap + ("\n" + fn if fn else "")).strip() or f"({t})"
                builder.add_atomic(t, text, ref, extra={
                    "img_paths": [c.get("image_source", {}).get("path", "")]
                                  if c.get("image_source") else [],
                    "image_caption": c.get("image_caption") or [],
                    "image_footnote": c.get("image_footnote") or [],
                })
                continue

            if t in TEXT_TYPES:
                key = (gpage, idx)
                grp = par_frag2grp.get(key)
                if grp is not None:
                    if grp["merged_id"] in par_done:
                        continue
                    par_done.add(grp["merged_id"])
                    refs = [{"global_page": f["global_page"], "idx_in_page": f["idx_in_page"],
                             "type": "paragraph", "merged_paragraph": grp["merged_id"]}
                            for f in grp["fragments"]]
                    # 跨页段落作为整体进入文本流
                    text = grp["text"]
                    n = est_tokens(text)
                    if builder.buf and builder.buf_tokens + n > MAX_TOKENS:
                        builder._flush_text()
                    builder.buf.append((text, refs[0]))
                    builder.buf[-1] = (text, refs[0])  # 单 ref 记首 fragment
                    builder.buf_tokens += n
                else:
                    text = item_text(it)
                    if text.strip():
                        builder.add_text_item(text, ref)
                continue

    chunks = builder.finish()

    # 统计
    type_counts = Counter(c["chunk_type"] for c in chunks)
    stats = {
        "total_chunks": len(chunks),
        "type_counts": dict(type_counts),
        "merged_tables_used": len(tbl_done),
        "merged_paragraphs_used": len(par_done),
        "text_chunk_tokens": {
            "min": min((est_tokens(c["text"]) for c in chunks if c["chunk_type"] == "text"), default=0),
            "max": max((est_tokens(c["text"]) for c in chunks if c["chunk_type"] == "text"), default=0),
            "avg": int(sum(est_tokens(c["text"]) for c in chunks if c["chunk_type"] == "text")
                       / max(1, type_counts.get("text", 1))),
        },
        "chunks_without_section": sum(1 for c in chunks if not c["section_path"]),
    }

    with open(os.path.join(CN, "chunks.jsonl"), "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lines = ["P4 Chunking Report", "=" * 60]
    for k, v in stats.items():
        lines.append(f"{k}: {v}")
    report = "\n".join(lines)
    with open(os.path.join(CN, "chunking_report.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
