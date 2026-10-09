# P6: 构建检索索引 — BM25(jieba+rank_bm25) + bge-m3 dense
# 索引文本：text chunk 用 text；table/chart chunk 用 text(caption+footnote) + table_markdown
import json
import os
import pickle

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS = 'data/cn/chunks.jsonl'
OUT = 'index'
BGE_M3 = os.path.expanduser('~/.cache/modelscope/models/BAAI--bge-m3/snapshots/master')
BATCH = 16  # 长表 markdown 可达 8k tokens，batch 64 会 OOM


def index_text(c: dict) -> str:
    if c['chunk_type'] in ('table', 'chart'):
        md = c.get('table_markdown') or ''
        return (c['text'] + ('\n\n' + md if md else '')).strip()
    return c['text']


def bm25_tokenize(s: str) -> list:
    return [t for t in jieba.lcut(s.lower()) if t.strip()]


def main():
    os.makedirs(OUT, exist_ok=True)
    chunks, doc_ids, texts = [], [], []
    with open(CHUNKS) as f:
        for line in f:
            c = json.loads(line)
            t = index_text(c)
            if not t:
                continue  # 空 chunk（如无 caption 的 image）不入索引
            chunks.append(c)
            doc_ids.append(c['chunk_id'])
            texts.append(t)

    # BM25
    corpus_tok = [bm25_tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tok)
    with open(f'{OUT}/bm25.pkl', 'wb') as f:
        pickle.dump({'bm25': bm25, 'doc_ids': doc_ids}, f)
    print('BM25 done:', len(doc_ids))

    # dense
    model = SentenceTransformer(BGE_M3, device='cuda',
                                model_kwargs={'torch_dtype': 'float16'})
    model.max_seq_length = 8192
    emb = model.encode(texts, batch_size=BATCH, normalize_embeddings=True,
                       show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(f'{OUT}/dense.npy', emb)
    with open(f'{OUT}/doc_ids.json', 'w') as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    print('dense done:', emb.shape)

    with open(f'{OUT}/meta.json', 'w') as f:
        json.dump({'n_docs': len(doc_ids), 'dense_dim': int(emb.shape[1]),
                   'model': 'BAAI/bge-m3', 'normalize': True,
                   'batch_size': BATCH}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
