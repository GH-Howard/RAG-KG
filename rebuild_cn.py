"""
从 MinerU 原始输出重建 data/cn/。

废弃旧 cn/ 的合并产物（combined_content_list.json / _v2.json / combined_full.md / images/），
直接从 data/MinerU/ 的 39 个钢质海船文件夹重建。

产出：
  data/cn/combined_content_list.json        v1 原子项（保留 page_idx, bbox, img_path, table_body）
  data/cn/combined_content_list_v2.json     v2 per-page 段落组（保留 page 层级，不 flatten）
  data/cn/combined_full.md                  合并 markdown
  data/cn/images/                           合并的图（保留原始 hash 文件名，不加页面前缀）
  data/cn/cross_page_tables.json            跨页表检测（bbox 启发式）
  data/cn/manifest.json                     重建元信息
  data/cn/rebuild_report.txt                重建统计报告

设计决策（修复旧 cn/ 的三个结构性问题）：
1. v2 保留 per-page 结构：list-of-lists 外层 3120 个 page（39 folder × 80 page），
   每个 item 显式打 page_idx/global_page 标签。不 flatten。
2. images 保留原始 hash 文件名，JSON 引用路径不变，路径天然匹配。
3. v1/v2 各自保留原生字段名（v1 用 table_body, v2 用 content.html），不做字段扁平化。
"""
import json
import os
import shutil
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
MINERU = os.path.join(BASE, "data", "MinerU")
CN_OUT = os.path.join(BASE, "data", "cn")
IMAGES_OUT = os.path.join(CN_OUT, "images")

# 跨页表检测的 bbox 启发式参数
PAGE_H = 967
BOTTOM_THRESHOLD = 0.85  # 表 bbox bottom > 85% 页面高度
TOP_THRESHOLD = 0.15     # 表 bbox top < 15% 页面高度


def find_ship_folders():
    """列出所有钢质海船入级规范的 MinerU 输出文件夹，按页码排序。"""
    folders = []
    for d in os.listdir(MINERU):
        full = os.path.join(MINERU, d)
        if not os.path.isdir(full):
            continue
        if "钢质海船入级规范" not in d:
            continue
        # folder name starts with page_start: "1_PDFsam_ship..." -> 1
        page_start = int(d.split("_", 1)[0])
        folders.append((page_start, d))
    folders.sort()
    return folders


def find_content_list_file(folder_path, suffix):
    """找到 folder 里的 content_list 文件，返回完整路径。"""
    for f in os.listdir(folder_path):
        if f.endswith(suffix):
            return os.path.join(folder_path, f)
    return None


def merge_images(ship_folders):
    """把 39 个 images/ 文件夹的内容拷贝到 cn/images/，保留原始 hash 文件名。
    冲突时保留先到的文件（不同 folder 的同 hash 文件大概率是同一张图，因为 hash 来自图片内容）。"""
    os.makedirs(IMAGES_OUT, exist_ok=True)
    stats = {"copied": 0, "skipped_dup": 0, "conflict_different": 0}
    for _, d in ship_folders:
        src_dir = os.path.join(MINERU, d, "images")
        if not os.path.isdir(src_dir):
            continue
        for f in os.listdir(src_dir):
            src = os.path.join(src_dir, f)
            dst = os.path.join(IMAGES_OUT, f)
            if os.path.exists(dst):
                # 按内容大小粗判，如果大小相同视为同一张图
                if os.path.getsize(src) == os.path.getsize(dst):
                    stats["skipped_dup"] += 1
                else:
                    stats["conflict_different"] += 1
                    # 冲突时保留先到的，记录但不覆盖
                continue
            shutil.copy2(src, dst)
            stats["copied"] += 1
    return stats


def merge_v1(ship_folders):
    """合并所有 folder 的 v1 content_list，给每项打 source_folder/page_start/global_page 标签。"""
    all_items = []
    for page_start, d in ship_folders:
        path = os.path.join(MINERU, d)
        cl_file = find_content_list_file(path, "_content_list.json")
        if not cl_file:
            print(f"WARN: {d} 没有 v1 content_list")
            continue
        with open(cl_file) as f:
            data = json.load(f)
        for it in data:
            if not isinstance(it, dict):
                continue
            it["_source_folder"] = d
            it["_page_start"] = page_start
            page_idx = it.get("page_idx", 0)
            it["global_page"] = page_start + page_idx
            all_items.append(it)
    return all_items


def merge_v2(ship_folders):
    """合并所有 folder 的 v2 content_list，保留 per-page 结构。
    每个 page list 内的 item 显式打 page_idx/global_page 标签。"""
    all_pages = []
    for page_start, d in ship_folders:
        path = os.path.join(MINERU, d)
        v2_file = find_content_list_file(path, "_content_list_v2.json")
        if not v2_file:
            print(f"WARN: {d} 没有 v2 content_list")
            continue
        with open(v2_file) as f:
            pages = json.load(f)
        # pages 是 list-of-lists，外层 index 就是 page_idx
        for page_idx, page_items in enumerate(pages):
            if not isinstance(page_items, list):
                continue
            for it in page_items:
                if not isinstance(it, dict):
                    continue
                it["_source_folder"] = d
                it["_page_start"] = page_start
                it["page_idx"] = page_idx
                it["global_page"] = page_start + page_idx
            all_pages.append(page_items)
    return all_pages


def merge_md(ship_folders):
    """合并所有 folder 的 full.md，加分页注释。"""
    parts = []
    for page_start, d in ship_folders:
        path = os.path.join(MINERU, d)
        md_file = os.path.join(path, "full.md")
        if not os.path.exists(md_file):
            print(f"WARN: {d} 没有 full.md")
            continue
        with open(md_file) as f:
            content = f.read()
        header = f"<!-- ===== PAGE_RANGE_START: {page_start} (source: {d}) ===== -->\n\n"
        parts.append(header + content)
    return "\n\n".join(parts)


def detect_cross_page_tables(v1_items):
    """bbox 启发式检测跨页表：表 A bbox 底部 > 85% 页面高度，紧邻的下一页表 B bbox 顶部 < 15%。
    返回跨页表对列表。"""
    tables = [it for it in v1_items if it.get("type") == "table"]
    # 按 (source_folder, page_idx, bbox_top) 排序
    tables.sort(key=lambda t: (
        t.get("_source_folder", ""),
        t.get("page_idx", -1),
        (t.get("bbox") or [0, 0, 0, 0])[1]
    ))
    pairs = []
    for i in range(len(tables) - 1):
        a, b = tables[i], tables[i + 1]
        if a.get("_source_folder") != b.get("_source_folder"):
            continue
        pai = a.get("page_idx", -1)
        pbi = b.get("page_idx", -1)
        if pbi - pai != 1:
            continue
        ba = a.get("bbox") or [0, 0, 0, 0]
        bb = b.get("bbox") or [0, 0, 0, 0]
        if ba[3] > PAGE_H * BOTTOM_THRESHOLD and bb[1] < PAGE_H * TOP_THRESHOLD:
            pairs.append({
                "first_table": {
                    "global_page": a.get("global_page"),
                    "bbox": ba,
                    "img_path": a.get("img_path"),
                    "table_caption": a.get("table_caption"),
                },
                "second_table": {
                    "global_page": b.get("global_page"),
                    "bbox": bb,
                    "img_path": b.get("img_path"),
                    "table_caption": b.get("table_caption"),
                },
            })
    return pairs


def compute_stats(v1_items, v2_pages, img_stats, cross_pairs):
    """生成重建报告。"""
    lines = []
    lines.append("Rebuild Report: data/cn from data/MinerU")
    lines.append("=" * 60)
    lines.append(f"\nv1 total items: {len(v1_items)}")
    types = Counter(it.get("type", "?") for it in v1_items)
    lines.append("v1 type distribution:")
    for t, c in types.most_common():
        lines.append(f"  {t}: {c}")

    lines.append(f"\nv2 total pages: {len(v2_pages)}")
    v2_total_items = sum(len(p) for p in v2_pages)
    lines.append(f"v2 total items (per-page sum): {v2_total_items}")

    lines.append(f"\nImages:")
    lines.append(f"  copied:  {img_stats['copied']}")
    lines.append(f"  skipped (same-hash dup): {img_stats['skipped_dup']}")
    lines.append(f"  conflict (same-name diff-size): {img_stats['conflict_different']}")

    lines.append(f"\nCross-page table pairs: {len(cross_pairs)}")
    if cross_pairs:
        lines.append("  first 3:")
        for p in cross_pairs[:3]:
            lines.append(f"    p{p['first_table']['global_page']} -> p{p['second_table']['global_page']}")

    # 检查 v1 items with img_path 是否都对应磁盘文件
    img_refs = [it["img_path"] for it in v1_items if it.get("img_path")]
    lines.append(f"\nv1 img_path refs: {len(img_refs)}")
    missing = 0
    for ref in img_refs:
        # ref 形如 'images/<hash>.jpg'
        fname = ref.rsplit("/", 1)[-1]
        if not os.path.exists(os.path.join(IMAGES_OUT, fname)):
            missing += 1
    lines.append(f"  missing on disk: {missing}")

    return "\n".join(lines)


def main():
    ship_folders = find_ship_folders()
    print(f"Found {len(ship_folders)} ship spec folders, pages {ship_folders[0][0]}..{ship_folders[-1][0]}")

    # 1. 合并 images
    print("Merging images...")
    img_stats = merge_images(ship_folders)

    # 2. 合并 v1
    print("Merging v1 content_list...")
    v1_items = merge_v1(ship_folders)

    # 3. 合并 v2 (per-page)
    print("Merging v2 content_list (per-page)...")
    v2_pages = merge_v2(ship_folders)

    # 4. 合并 markdown
    print("Merging full.md...")
    md = merge_md(ship_folders)

    # 5. 检测跨页表
    print("Detecting cross-page tables...")
    cross_pairs = detect_cross_page_tables(v1_items)

    # 6. 写文件
    print("Writing outputs...")
    with open(os.path.join(CN_OUT, "combined_content_list.json"), "w") as f:
        json.dump(v1_items, f, ensure_ascii=False)
    with open(os.path.join(CN_OUT, "combined_content_list_v2.json"), "w") as f:
        json.dump(v2_pages, f, ensure_ascii=False)
    with open(os.path.join(CN_OUT, "combined_full.md"), "w") as f:
        f.write(md)
    with open(os.path.join(CN_OUT, "cross_page_tables.json"), "w") as f:
        json.dump(cross_pairs, f, ensure_ascii=False, indent=2)

    manifest = {
        "title": "钢质海船入级规范 2026 - Rebuilt from MinerU raw output",
        "source_pdf": "《钢质海船入级规范》2026综合文本.pdf",
        "rebuild_date": "2026-10-08",
        "total_folders": len(ship_folders),
        "page_ranges": [
            {"page_start": ps, "source_folder": d} for ps, d in ship_folders
        ],
        "output_files": {
            "combined_markdown": "combined_full.md",
            "combined_content_list": "combined_content_list.json",
            "combined_content_list_v2": "combined_content_list_v2.json",
            "cross_page_tables": "cross_page_tables.json",
            "images_dir": "images/",
        },
        "total_images": img_stats["copied"],
        "cross_page_table_pairs": len(cross_pairs),
        "notes": [
            "v2 preserves per-page structure (list-of-page-lists), page_idx is explicit",
            "images keep original MinerU hash filenames, no page prefix",
            "v1 keeps table_body, v2 keeps content.html (raw field names preserved)",
        ],
    }
    with open(os.path.join(CN_OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 7. 报告
    report = compute_stats(v1_items, v2_pages, img_stats, cross_pairs)
    with open(os.path.join(CN_OUT, "rebuild_report.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
