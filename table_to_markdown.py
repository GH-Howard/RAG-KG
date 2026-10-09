# P5: table chunk 的 table_html → markdown，写回 chunks（table_markdown 字段）
# 设计：markdown 进索引 + HTML 进 metadata；pandas.read_html（rowspan 下填充/colspan 右复制）
import json
import re
from io import StringIO

import pandas as pd

SRC = 'data/cn/chunks.jsonl'
DST = 'data/cn/chunks.jsonl.tmp'
REPORT = 'data/cn/p5_report.txt'


def html_to_markdown(html: str):
    """HTML table → markdown。返回 (markdown, n_rows, n_cols)。"""
    # 空表检测：所有单元格无文本（lxml 对全空 td 会解析失败）
    cells = re.findall(r'<td[^>]*>(.*?)</td>', html, flags=re.S)
    if not any(c.strip() for c in cells):
        return '', 0, 0
    dfs = pd.read_html(StringIO(html))
    parts = []
    max_r, max_c = 0, 0
    for df in dfs:
        df = df.fillna('')
        # 清洗单元格：竖线转全角、换行折叠为空格，避免破坏 markdown 表格
        df = df.astype(str).apply(lambda col: col.map(
            lambda s: re.sub(r'\s*\n\s*', ' ', s).replace('|', '｜').strip()))
        df.columns = [''] * len(df.columns)  # 本数据无 <th>，清掉 read_html 的整数列名噪音
        parts.append(df.to_markdown(index=False))
        max_r, max_c = max(max_r, df.shape[0]), max(max_c, df.shape[1])
    return '\n\n'.join(parts), max_r, max_c


def main():
    total = table_n = converted = empty_html = failed = 0
    fail_detail = []
    max_r = max_c = 0
    with open(SRC) as fin, open(DST, 'w') as fout:
        for line in fin:
            total += 1
            c = json.loads(line)
            if c['chunk_type'] in ('table', 'chart'):
                table_n += 1
                html = c.get('table_html') or ''
                if not html.strip() or html.strip() == 'None':  # 已知数据特性:个别为字面量 'None'
                    c['table_markdown'] = ''
                    empty_html += 1
                else:
                    try:
                        md, r, cc = html_to_markdown(html)
                        c['table_markdown'] = md
                        converted += 1
                        max_r, max_c = max(max_r, r), max(max_c, cc)
                    except Exception as e:  # noqa: BLE001 — 单表失败不中断全量
                        c['table_markdown'] = ''
                        failed += 1
                        fail_detail.append((c['chunk_id'], repr(e)[:120]))
            fout.write(json.dumps(c, ensure_ascii=False) + '\n')

    with open(REPORT, 'w') as f:
        f.write(f'total_chunks={total}\ntable_chunks={table_n}\n')
        f.write(f'converted={converted}\nempty_html={empty_html}\nfailed={failed}\n')
        f.write(f'max_rows={max_r}\nmax_cols={max_c}\n')
        for cid, err in fail_detail:
            f.write(f'FAIL {cid}: {err}\n')
    print(open(REPORT).read())

    import os
    os.replace(DST, SRC)


if __name__ == '__main__':
    main()
