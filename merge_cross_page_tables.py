"""
P1: 跨页表合并器

输入:
  data/cn/combined_content_list.json   v1 原子项 (table_body HTML)
  data/cn/cross_page_tables.json       451 对 folder 内跨页表 (bbox 启发式, rebuild_cn.py 产出)

输出:
  data/cn/merged_tables.json           合并后的表 (merged_html + fragments 引用)
  data/cn/merge_tables_report.txt      合并统计报告

逻辑:
  1. union-find 把共享端点的 pair 归组, 支持跨 3+ 页的链式表 (200 个共享端点)
  2. 跨 folder 检测: global_page 相邻 + folder 不同 + bbox 启发式 -> 额外 pair
     (12 处跨 folder 相邻表中 4 处通过启发式)
  3. 组内按 (global_page, bbox_top) 排序, 首个 fragment 为 head, 依次拼接 table_body:
     - 去掉 fragment 开头与 head 前 2 行文本重复的 "续表表头" 行
     - 列数不一致只记 warning 仍合并 (跨 folder candidate 列数不一致则拒绝)
  4. fragments 记录 (global_page, bbox, source_folder) 供 P4 chunking 跳过原始 table 原子
"""
import json
import os
from collections import defaultdict

from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.abspath(__file__))
CN = os.path.join(BASE, "data", "cn")

PAGE_H = 967
BOTTOM_THRESHOLD = 0.85
TOP_THRESHOLD = 0.15

# ---------- union-find ----------
_parent = {}


def find(x):
    while _parent[x] != x:
        _parent[x] = _parent[_parent[x]]
        x = _parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        _parent[ra] = rb


# ---------- html helpers ----------
def norm(text):
    return "".join(text.split())


def row_text(tr):
    return norm("".join(td.get_text() for td in tr.find_all(["td", "th"])))


def col_count(soup):
    n = 0
    for tr in soup.find_all("tr"):
        c = sum(int(td.get("colspan", 1)) for td in tr.find_all(["td", "th"]))
        n = max(n, c)
    return n


def merge_fragment(head_soup, frag_html, header_texts):
    """把 fragment 的 <tr> 追加到 head_soup 的 <table>。
    去掉 fragment 开头与 head 表头 (前 2 行文本) 重复的行。
    返回 (dropped_rows, added_rows, frag_col_count)。"""
    frag_soup = BeautifulSoup(frag_html or "", "html.parser")
    frag_rows = frag_soup.find_all("tr")
    dropped = 0
    i = 0
    while i < len(frag_rows) and row_text(frag_rows[i]) in header_texts:
        i += 1
        dropped += 1
    table = head_soup.find("table")
    added = 0
    for tr in frag_rows[i:]:
        table.append(tr)
        added += 1
    return dropped, added, col_count(frag_soup)


# ---------- main ----------
def main():
    v1 = json.load(open(os.path.join(CN, "combined_content_list.json")))
    pairs = json.load(open(os.path.join(CN, "cross_page_tables.json")))
    tables = [it for it in v1 if it.get("type") == "table"]

    # v1 table 索引: (global_page, bbox_tuple) -> item
    by_key = {}
    for t in tables:
        key = (t["global_page"], tuple(t.get("bbox") or [0, 0, 0, 0]))
        by_key[key] = t

    # 1. 收集 folder 内 pair 的端点
    raw_pairs = []  # (key_a, key_b, kind)
    unmatched = 0
    for p in pairs:
        ka = (p["first_table"]["global_page"], tuple(p["first_table"]["bbox"]))
        kb = (p["second_table"]["global_page"], tuple(p["second_table"]["bbox"]))
        if ka not in by_key or kb not in by_key:
            unmatched += 1
            continue
        raw_pairs.append((ka, kb, "intra_folder"))

    # 2. 跨 folder 检测
    by_page = defaultdict(list)
    for t in tables:
        by_page[t["global_page"]].append(t)
    cross_folder_adj = 0  # 所有跨 folder 相邻 (不过滤 bbox)
    cross_folder_cands = []  # (key_a, key_b)
    for g in sorted(by_page):
        if g + 1 not in by_page:
            continue
        fa = by_page[g][0]["_source_folder"]
        fb = by_page[g + 1][0]["_source_folder"]
        if fa == fb:
            continue
        # folder 边界
        if by_page[g] and by_page[g + 1]:
            cross_folder_adj += 1
        bottoms = [t for t in by_page[g]
                   if (t.get("bbox") or [0] * 4)[3] > PAGE_H * BOTTOM_THRESHOLD]
        tops = [t for t in by_page[g + 1]
                if (t.get("bbox") or [0] * 4)[1] < PAGE_H * TOP_THRESHOLD]
        for a in bottoms:
            for b in tops:
                cross_folder_cands.append((
                    (a["global_page"], tuple(a["bbox"])),
                    (b["global_page"], tuple(b["bbox"])),
                ))

    # 跨 folder candidate 列数校验, 不一致则拒绝
    rejected_cf = []
    for ka, kb in cross_folder_cands:
        ca = col_count(BeautifulSoup(by_key[ka].get("table_body") or "", "html.parser"))
        cb = col_count(BeautifulSoup(by_key[kb].get("table_body") or "", "html.parser"))
        if ca != cb:
            rejected_cf.append({"a": ka, "b": kb, "col_a": ca, "col_b": cb})
        else:
            raw_pairs.append((ka, kb, "cross_folder"))

    # 3. union-find 归组
    for ka, kb, _ in raw_pairs:
        _parent.setdefault(ka, ka)
        _parent.setdefault(kb, kb)
        union(ka, kb)
    groups = defaultdict(list)
    for k in _parent:
        groups[find(k)].append(k)
    # 组 -> kind (含 cross_folder pair 的组标记为 cross_folder)
    group_kind = {}
    for ka, kb, kind in raw_pairs:
        r = find(ka)
        if kind == "cross_folder":
            group_kind[r] = "cross_folder"
        else:
            group_kind.setdefault(r, "intra_folder")

    # 4. 逐组合并
    merged = []
    total_dropped = 0
    col_mismatch_groups = 0
    for i, (root, keys) in enumerate(sorted(groups.items(),
                                            key=lambda kv: min(k[0] for k in kv[1]))):
        frags = sorted(keys, key=lambda k: (k[0], k[1][1]))  # global_page, bbox_top
        head_item = by_key[frags[0]]
        head_soup = BeautifulSoup(head_item.get("table_body") or "", "html.parser")
        head_rows = head_soup.find_all("tr")
        header_texts = {row_text(tr) for tr in head_rows[:2]}
        head_cols = col_count(head_soup)

        frag_records = [{
            "global_page": frags[0][0],
            "bbox": list(frags[0][1]),
            "source_folder": head_item["_source_folder"],
            "img_path": head_item.get("img_path") or "",
            "dropped_header_rows": 0,
            "rows_added": len(head_rows),
        }]
        warnings = []
        total_rows = len(head_rows)

        for fk in frags[1:]:
            item = by_key[fk]
            dropped, added, fcols = merge_fragment(
                head_soup, item.get("table_body"), header_texts)
            total_dropped += dropped
            total_rows += added
            if fcols != head_cols:
                warnings.append(
                    f"col_mismatch page {fk[0]}: head={head_cols} frag={fcols}")
            frag_records.append({
                "global_page": fk[0],
                "bbox": list(fk[1]),
                "source_folder": item["_source_folder"],
                "img_path": item.get("img_path") or "",
                "dropped_header_rows": dropped,
                "rows_added": added,
            })
        if warnings:
            col_mismatch_groups += 1

        caption = next((by_key[fk].get("table_caption") for fk in frags
                        if by_key[fk].get("table_caption")), [])
        footnote = []
        for fk in frags:
            for fn in by_key[fk].get("table_footnote") or []:
                if fn not in footnote:
                    footnote.append(fn)

        merged.append({
            "merged_id": f"mt_{i:04d}",
            "merge_kind": group_kind[root],
            "global_pages": [fk[0] for fk in frags],
            "page_span": [frags[0][0], frags[-1][0]],
            "source_folders": sorted({r["source_folder"] for r in frag_records}),
            "img_paths": [r["img_path"] for r in frag_records if r["img_path"]],
            "table_caption": caption,
            "table_footnote": footnote,
            "col_count": head_cols,
            "row_count": total_rows,
            "n_fragments": len(frags),
            "warnings": warnings,
            "fragments": frag_records,
            "merged_html": str(head_soup.find("table")),
        })

    # 5. 输出
    stats = {
        "intra_folder_pairs": len(pairs),
        "unmatched_pairs": unmatched,
        "cross_folder_adjacent_pages": cross_folder_adj,
        "cross_folder_candidates": len(cross_folder_cands),
        "cross_folder_rejected": len(rejected_cf),
        "merged_groups": len(merged),
        "cross_folder_groups": sum(1 for m in merged if m["merge_kind"] == "cross_folder"),
        "total_fragments": sum(m["n_fragments"] for m in merged),
        "max_fragments_in_group": max(m["n_fragments"] for m in merged),
        "total_header_rows_dropped": total_dropped,
        "col_mismatch_groups": col_mismatch_groups,
    }
    out = {"stats": stats, "merged_tables": merged,
           "rejected_cross_folder": rejected_cf}
    with open(os.path.join(CN, "merged_tables.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)

    lines = ["P1 Merge Cross-Page Tables Report", "=" * 60]
    for k, v in stats.items():
        lines.append(f"{k}: {v}")
    if rejected_cf:
        lines.append("\nrejected cross-folder candidates:")
        for r in rejected_cf:
            lines.append(f"  {r['a']} -> {r['b']} cols {r['col_a']} vs {r['col_b']}")
    dist = defaultdict(int)
    for m in merged:
        dist[m["n_fragments"]] += 1
    lines.append("\nfragments-per-group distribution:")
    for n in sorted(dist):
        lines.append(f"  {n} fragments: {dist[n]} groups")
    report = "\n".join(lines)
    with open(os.path.join(CN, "merge_tables_report.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
