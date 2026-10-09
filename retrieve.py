# P6: 检索验证 — BM25 + bge-m3 dense 双路 → RRF 融合 → 可选 bge-reranker-v2-m3 重排
import argparse
import json
import pickle
import os

import numpy as np

IDX = 'index'
BGE_M3 = os.path.expanduser('~/.cache/modelscope/models/BAAI--bge-m3/snapshots/master')
RERANKER = os.path.expanduser('~/.cache/modelscope/models/BAAI--bge-reranker-v2-m3/snapshots/master')
CHUNKS = 'data/cn/chunks.jsonl'


def bm25_tokenize(s):
    import jieba
    return [t for t in jieba.lcut(s.lower()) if t.strip()]


def index_text(c: dict) -> str:
    """与 build_index.py 保持一致：table/chart 用 text + table_markdown"""
    if c['chunk_type'] in ('table', 'chart'):
        md = c.get('table_markdown') or ''
        return (c['text'] + ('\n\n' + md if md else '')).strip()
    return c['text']


def load_chunks():
    m = {}
    with open(CHUNKS) as f:
        for line in f:
            c = json.loads(line)
            m[c['chunk_id']] = c
    return m


def rrf(rankings, k=60, topk=10):
    score = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r + 1)
    return [c for c, _ in sorted(score.items(), key=lambda x: -x[1])[:topk]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('query')
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--rerank', action='store_true')
    args = ap.parse_args()

    chunks = load_chunks()
    doc_ids = json.load(open(f'{IDX}/doc_ids.json'))
    emb = np.load(f'{IDX}/dense.npy')
    bm25 = pickle.load(open(f'{IDX}/bm25.pkl', 'rb'))['bm25']

    # BM25 路
    bm_rank = [doc_ids[i] for i in np.argsort(bm25.get_scores(bm25_tokenize(args.query)))[::-1][:50]]
    # dense 路
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_M3, device='cuda')
    q = model.encode([args.query], normalize_embeddings=True)
    sims = emb @ q[0].astype(np.float32)
    dn_rank = [doc_ids[i] for i in np.argsort(sims)[::-1][:50]]
    print('BM25 top5:', bm_rank[:5])
    print('Dense top5:', dn_rank[:5])

    merged = rrf([bm_rank, dn_rank], topk=max(args.topk, 20))
    results = merged

    if args.rerank:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(RERANKER, device='cuda', max_length=8192)
        pairs = [[args.query, index_text(chunks[cid])[:6000]] for cid in merged]
        scores = ce.predict(pairs)
        order = np.argsort(scores)[::-1][:args.topk]
        results = [merged[i] for i in order]

    for r, cid in enumerate(results, 1):
        c = chunks[cid]
        preview = c['text'].replace('\n', ' ')[:80]
        print(f'{r:2d}. [{c["chunk_type"]}] {cid} p{c["global_pages"]} {c["section_path"][-1][:30]} | {preview}')


if __name__ == '__main__':
    main()
