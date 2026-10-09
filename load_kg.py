# P8 Neo4j 入库：读 kg_out/triples_*.jsonl，实体合并 + 关系写入
# 节点 (Entity {name, type, source_chunk_id, global_page})；边 -[REL {type, source_chunk_id, global_page, confidence}]- (
# 设计见 pipeline.md「已定稿设计」)
import argparse
import json
import os
from collections import Counter

from neo4j import GraphDatabase

DEFAULT_FILES = ['kg_out/triples_a_text_table.jsonl', 'kg_out/triples_b_image_chart.jsonl']


def load_triples(paths):
    """读三元组，行级去重（同 s/p/o/source_chunk_id 视为重复）"""
    seen, rows = set(), []
    for path in paths:
        if not os.path.exists(path):
            print(f'skip (not found): {path}')
            continue
        n = 0
        with open(path) as f:
            for line in f:
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (t['s'], t['p'], t['o'], t.get('source_chunk_id'))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(t)
                n += 1
        print(f'{path}: {n} triples')
    return rows


def majority_type(votes):
    """同名实体多类型投票；全为空/其他时归「其他」"""
    cnt = Counter(t for t in votes if t and t != '其他')
    if cnt:
        return cnt.most_common(1)[0][0]
    return '其他' if not votes or all(v == '其他' for v in votes) else (votes[0] or '其他')


def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', default=DEFAULT_FILES)
    ap.add_argument('--uri', default='bolt://localhost:7688')
    ap.add_argument('--user', default='neo4j')
    ap.add_argument('--password', default=os.environ.get('NEO4J_PASSWORD', ''))
    ap.add_argument('--batch', type=int, default=5000)
    ap.add_argument('--wipe', action='store_true', help='入库前清空库')
    args = ap.parse_args()

    rows = load_triples(args.files)
    if not rows:
        print('no triples to load')
        return

    # 实体类型投票
    votes = {}
    for t in rows:
        votes.setdefault(t['s'], []).append(t.get('s_type'))
        votes.setdefault(t['o'], []).append(t.get('o_type'))
    name_type = {name: majority_type(v) for name, v in votes.items()}
    print(f'entities: {len(name_type)}, edges (raw): {len(rows)}')

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    def wipe_tx(tx):
        tx.run('MATCH (n) DETACH DELETE n').consume()

    def constraint_tx(tx):
        tx.run('CREATE CONSTRAINT entity_name IF NOT EXISTS '
               'FOR (e:Entity) REQUIRE e.name IS UNIQUE').consume()

    def entity_tx(tx, batch):
        tx.run('''
            UNWIND $rows AS row
            MERGE (e:Entity {name: row.name})
            ON CREATE SET e.type = row.type
        ''', rows=batch).consume()

    def edge_tx(tx, batch):
        tx.run('''
            UNWIND $rows AS row
            MATCH (a:Entity {name: row.s}), (b:Entity {name: row.o})
            MERGE (a)-[r:REL {type: row.p}]->(b)
            ON CREATE SET r.source_chunk_id = row.sid, r.global_page = row.page,
                          r.confidence = row.conf, r.count = 1
            ON MATCH SET r.count = coalesce(r.count, 1) + 1
        ''', rows=batch).consume()

    with driver.session() as ses:
        if args.wipe:
            ses.execute_write(wipe_tx)
            print('database wiped')
        ses.execute_write(constraint_tx)

        ents = [{'name': n, 'type': ty} for n, ty in name_type.items()]
        for batch in batched(ents, args.batch):
            ses.execute_write(entity_tx, batch)
        print(f'entities written: {len(ents)}')

        edges = [{'s': t['s'], 'p': t['p'], 'o': t['o'],
                  'sid': t.get('source_chunk_id'), 'page': t.get('global_page'),
                  'conf': t.get('confidence', 0.5)} for t in rows]
        for batch in batched(edges, args.batch):
            ses.execute_write(edge_tx, batch)
        stat = ses.run('MATCH (n:Entity) WITH count(n) AS nodes '
                       'MATCH ()-[r:REL]->() RETURN nodes, count(r) AS rels').single()
        print(f'NEO4J DONE: {stat["nodes"]} nodes, {stat["rels"]} rels')
    driver.close()


if __name__ == '__main__':
    main()
