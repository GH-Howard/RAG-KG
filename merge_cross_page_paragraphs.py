"""
P2: 跨页段落合并器

输入:
  data/cn/combined_content_list_v2.json   v2 per-page (39 folder 合并, global_page 从 1 开始)

输出:
  data/cn/merged_paragraphs.json          合并后的段落组
  data/cn/merge_paragraphs_report.txt     统计报告

判定逻辑 (page G 最后一个内容项 与 page G+1 第一个内容项):
  1. 两者都是 paragraph 类型
  2. 页 G 末段 bbox bottom > 75% 页高, 页 G+1 首段 bbox top < 25% 页高
  3. 页 G 末段不以终止标点结尾 (。!?;:… 及配对引号/括号)
跳过 page_header / page_number / page_footnote / page_footer。
支持链式合并 (跨 3+ 页段落), 跨 folder 边界同样适用 (global_page 相邻即可)。
fragments 记录 (global_page, 页内 index) 供 P4 chunking 跳过原始段落原子。
"""
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
CN = os.path.join(BASE, "data", "cn")

PAGE_H = 967
BOTTOM_THRESHOLD = 0.75
TOP_THRESHOLD = 0.25

TERMINAL = tuple("。！？；：…．.!?;:\"'”’）)]」』")
SKIP_TYPES = {"page_header", "page_number", "page_footnote", "page_footer"}


def para_text(item):
    """拼接 paragraph 的内联内容为纯文本。"""
    parts = []
    for inline in (item.get("content") or {}).get("paragraph_content") or []:
        if isinstance(inline, dict):
            parts.append(str(inline.get("content", "")))
    return "".join(parts)


def page_content_items(page):
    """页内内容项 (跳过页眉页脚页码), 返回 (页内 index, item) 列表。"""
    return [(i, it) for i, it in enumerate(page) if it.get("type") not in SKIP_TYPES]


def main():
    v2 = json.load(open(os.path.join(CN, "combined_content_list_v2.json")))
    pages = sorted(
        (p for p in v2 if p),
        key=lambda p: p[0]["global_page"],
    )
    by_page = {p[0]["global_page"]: p for p in pages}

    # 每个段落的唯一 id: (global_page, idx_in_page)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    boundaries = 0
    for g in sorted(by_page):
        if g + 1 not in by_page:
            continue
        items_a = page_content_items(by_page[g])
        items_b = page_content_items(by_page[g + 1])
        if not items_a or not items_b:
            continue
        ia, last = items_a[-1]
        ib, first = items_b[0]
        if last.get("type") != "paragraph" or first.get("type") != "paragraph":
            continue
        ba = last.get("bbox") or [0, 0, 0, 0]
        bb = first.get("bbox") or [0, 0, 0, 0]
        if ba[3] <= PAGE_H * BOTTOM_THRESHOLD or bb[1] >= PAGE_H * TOP_THRESHOLD:
            continue
        text = para_text(last).rstrip()
        if not text or text.endswith(TERMINAL):
            continue
        boundaries += 1
        ka, kb = (g, ia), (g + 1, ib)
        parent.setdefault(ka, ka)
        parent.setdefault(kb, kb)
        union(ka, kb)

    groups = defaultdict(list)
    for k in parent:
        groups[find(k)].append(k)

    merged = []
    toc_skipped = 0
    for _, keys in sorted(groups.items(),
                          key=lambda kv: min(k[0] for k in kv[1])):
        frags = sorted(keys)  # (global_page, idx) 字典序即阅读顺序
        raw_texts = [para_text(by_page[g][idx]) for g, idx in frags]
        # 目录页噪音: 全部 fragment 都含点线引导符 "...." 的组不合并
        if raw_texts and all("...." in t for t in raw_texts):
            toc_skipped += 1
            continue
        text_parts = []
        frag_records = []
        folders = set()
        for (g, idx), raw in zip(frags, raw_texts):
            item = by_page[g][idx]
            text_parts.append(raw)
            folders.add(item["_source_folder"])
            frag_records.append({
                "global_page": g,
                "idx_in_page": idx,
                "bbox": item.get("bbox"),
                "source_folder": item["_source_folder"],
            })
        merged.append({
            "merged_id": f"mp_{len(merged):04d}",
            "global_pages": [g for g, _ in frags],
            "page_span": [frags[0][0], frags[-1][0]],
            "source_folders": sorted(folders),
            "n_fragments": len(frags),
            "text": "".join(text_parts),
            "fragments": frag_records,
        })

    stats = {
        "merge_boundaries": boundaries,
        "merged_groups": len(merged),
        "toc_groups_skipped": toc_skipped,
        "total_fragments": sum(m["n_fragments"] for m in merged),
        "max_fragments_in_group": max((m["n_fragments"] for m in merged), default=0),
        "cross_folder_groups": sum(1 for m in merged if len(m["source_folders"]) > 1),
    }
    with open(os.path.join(CN, "merged_paragraphs.json"), "w") as f:
        json.dump({"stats": stats, "merged_paragraphs": merged}, f, ensure_ascii=False)

    lines = ["P2 Merge Cross-Page Paragraphs Report", "=" * 60]
    for k, v in stats.items():
        lines.append(f"{k}: {v}")
    dist = defaultdict(int)
    for m in merged:
        dist[m["n_fragments"]] += 1
    lines.append("\nfragments-per-group distribution:")
    for n in sorted(dist):
        lines.append(f"  {n} fragments: {dist[n]} groups")
    report = "\n".join(lines)
    with open(os.path.join(CN, "merge_paragraphs_report.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
