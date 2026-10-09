# P7 双轨 KG 抽取
# 轨 A (text/table) → Qwen3-32B-FP8；轨 B (image/chart) → Qwen3-VL-32B-Instruct-FP8
# vLLM 离线批推理, temp=0, JSON schema 引导, 分窗口写盘可断点续跑
import argparse
import json
import os
import re
import time
from datetime import datetime

CHUNKS = 'data/cn/chunks.jsonl'
IMG_DIR = 'data/cn'
OUT_DIR = 'kg_out'

ENTITY_TYPES = ['设备', '结构', '检验', '参数', '材料', '组织', '标准', '规范条款', '过程', '其他']

SCHEMA = {
    'type': 'object',
    'properties': {
        'triples': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    's': {'type': 'string'},
                    's_type': {'type': 'string', 'enum': ENTITY_TYPES},
                    'p': {'type': 'string'},
                    'o': {'type': 'string'},
                    'o_type': {'type': 'string', 'enum': ENTITY_TYPES},
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                },
                'required': ['s', 's_type', 'p', 'o', 'o_type', 'confidence'],
            },
        }
    },
    'required': ['triples'],
}

SYS_PROMPT = (
    '你是《钢质海船入级规范》知识图谱构建专家。从给定片段中抽取知识三元组，要求：\n'
    '1. 只依据片段内容，不得编造或引入外部知识；\n'
    '2. s/o 为简短实体名（设备、结构、检验要求、参数、材料、组织、标准、条款等），'
    '保留规范编号与术语原样；\n'
    '3. p 为简短关系短语（如：适用于、应进行、应满足、定义为、属于、包含、间隔期为、不大于）；\n'
    '4. s_type/o_type 从 [' + ', '.join(ENTITY_TYPES) + '] 中选择；\n'
    '5. confidence 为 0~1 的置信度；\n'
    '6. 不要输出片段中没有依据的泛化事实；片段无可靠事实时输出空数组。\n'
    '只输出 JSON。'
)

SYS_PROMPT_B = (
    '你是《钢质海船入级规范》知识图谱构建专家。你需要从图片中抽取知识三元组。\n'
    '图片可能是：表单模板、流程图、示意图、曲线图、结构图、设备图等。\n'
    '要求：\n'
    '1. 仔细观察图片内容，结合章节路径和图片说明，抽取其中蕴含的知识三元组；\n'
    '2. s/o 为简短实体名（设备、结构、检验要求、参数、材料、组织、标准、条款等），保留编号与术语；\n'
    '3. p 为简短关系短语（如：包含、属于、适用于、定义为、位于、由…组成、标注为）；\n'
    '4. s_type/o_type 从 [' + ', '.join(ENTITY_TYPES) + '] 中选择；\n'
    '5. confidence 为 0~1 的置信度；\n'
    '6. 即使图片是表单模板，也要抽取其字段和结构关系；\n'
    '7. 无可靠事实时输出空数组。\n'
    '示例——若图片是一个检查记录表单（含字段：船名、IMO号、地点、日期、检查人员），可抽取：\n'
    '{"triples":[{"s":"检查记录表单","s_type":"规范条款","p":"包含字段","o":"船名","o_type":"参数","confidence":0.9},'
    '{"s":"检查记录表单","s_type":"规范条款","p":"包含字段","o":"IMO号","o_type":"参数","confidence":0.9},'
    '{"s":"检查人员","s_type":"组织","p":"填写","o":"检查记录表单","o_type":"规范条款","confidence":0.8}]}\n'
    '只输出 JSON。'
)


def index_text(c: dict) -> str:
    """与 build_index.py 一致：table/chart 用 text + table_markdown"""
    if c['chunk_type'] in ('table', 'chart'):
        md = c.get('table_markdown') or ''
        return (c['text'] + ('\n\n' + md if md else '')).strip()
    return c['text']


def build_messages(c: dict, track: str):
    section = ' > '.join(c.get('section_path') or [])
    pages = ','.join(map(str, c.get('global_pages') or []))
    if track == 'a':
        body = index_text(c)
        # 超长表格保护：截到 9000 字符（中文≈1 token/字），为提示+输出留余量
        if len(body) > 9000:
            body = body[:9000] + '\n……（片段过长，已截断）'
        user = f'章节路径：{section}\n页码：{pages}\n片段：\n{body}'
        return [{'role': 'system', 'content': SYS_PROMPT},
                {'role': 'user', 'content': user}]
    # track b: 图/表图片端到端抽取
    content = []
    for p in (c.get('img_paths') or []):
        ap = os.path.abspath(os.path.join(IMG_DIR, p))
        if os.path.exists(ap):
            content.append({'type': 'image_url', 'image_url': {'url': 'file://' + ap}})
    if not content:
        return None
    # 提取 caption/footnote 文本（caption 是 [{'type':'text','content':...}] 格式）
    def cap_text(items):
        if not items:
            return ''
        parts = []
        for it in items:
            if isinstance(it, dict):
                parts.append(it.get('content', ''))
            elif isinstance(it, str):
                parts.append(it)
        return ' '.join(p for p in parts if p)
    cap = cap_text(c.get('image_caption') or [])
    footnote = cap_text(c.get('image_footnote') or [])
    parts = [f'章节路径：{section}', f'页码：{pages}']
    if cap:
        parts.append(f'图片标题：{cap}')
    if footnote:
        parts.append(f'图片注记：{footnote}')
    parts.append('请仔细观察图片内容，结合上述上下文抽取知识三元组，只输出 JSON。')
    user_text = '\n'.join(parts)
    content.append({'type': 'text', 'text': user_text})
    return [{'role': 'system', 'content': SYS_PROMPT_B},
            {'role': 'user', 'content': content}]


def clean_triples(raw: dict, c: dict, extractor: str):
    out, seen = [], set()
    now = datetime.now().isoformat(timespec='seconds')
    pages = c.get('global_pages') or []
    for t in (raw or {}).get('triples', []):
        s, p, o = (str(t.get(k, '')).strip() for k in ('s', 'p', 'o'))
        if not s or not p or not o or s == o or len(s) > 120 or len(o) > 120:
            continue
        key = (s, p, o)
        if key in seen:
            continue
        seen.add(key)
        try:
            conf = min(1.0, max(0.0, float(t.get('confidence', 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        out.append({
            's': s, 'p': p, 'o': o,
            's_type': str(t.get('s_type', '其他'))[:20],
            'o_type': str(t.get('o_type', '其他'))[:20],
            'source_chunk_id': c['chunk_id'],
            'global_page': pages[0] if pages else None,
            'confidence': round(conf, 2),
            'extractor': extractor,
            'extract_time': now,
        })
    return out


def parse_json_loose(text: str):
    """兜底解析：去掉 thinking 标签、markdown 围栏，提取 JSON"""
    text = text.strip()
    # 去掉 Qwen3 thinking 标签内容（</think> 之后才是正式回复）
    think_end = text.rfind('</think>')
    if think_end != -1:
        text = text[think_end + 8:].strip()
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def make_sampling_params(n_sp_cls, schema):
    """兼容不同 vLLM 版本的 structured output API"""
    for mod, name in (('vllm.sampling_params', 'StructuredOutputsParams'),
                      ('vllm.sampling_params', 'GuidedDecodingParams'),
                      ('vllm.guided_decoding', 'GuidedDecodingParams')):
        try:
            cls = getattr(__import__(mod, fromlist=[name]), name)
            param = cls(json=schema)
            key = 'guided_decoding' if 'Guided' in name else 'structured_outputs'
            return n_sp_cls(**{key: param}), key
        except (ImportError, AttributeError):
            continue
    return n_sp_cls(), 'prompt_only'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', choices=['a', 'b'], required=True,
                    help='a: text+table; b: image+chart')
    ap.add_argument('--model', required=True)
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--window', type=int, default=256)
    ap.add_argument('--limit', type=int, default=0, help='仅处理前 N 条（冒烟）')
    ap.add_argument('--max-model-len', type=int, default=12288)
    ap.add_argument('--gpu-util', type=float, default=0.92)
    args = ap.parse_args()

    # 形如 .../models/Qwen--Qwen3-32B-FP8/snapshots/master → Qwen3-32B-FP8
    parts = args.model.rstrip('/').split('/')
    extractor = parts[-3].replace('Qwen--', '', 1) if parts[-2] == 'snapshots' else parts[-1]
    suffix = 'a_text_table' if args.track == 'a' else 'b_image_chart'
    out_path = os.path.join(OUT_DIR, f'triples_{suffix}.jsonl')
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['source_chunk_id'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f'resume: {len(done)} chunks already extracted')

    todo = []
    with open(CHUNKS) as f:
        for line in f:
            c = json.loads(line)
            ok_type = (c['chunk_type'] in ('text', 'table')) if args.track == 'a' \
                else (c['chunk_type'] in ('image', 'chart'))
            if ok_type and c['chunk_id'] not in done:
                todo.append(c)
    if args.limit:
        todo = todo[:args.limit]
    print(f'track {args.track}: {len(todo)} chunks to extract (extractor={extractor})')
    if not todo:
        return

    from vllm import LLM, SamplingParams
    # track a: structured output + no thinking（文本抽取已验证良好）
    # track b: 无 structured output + 开 thinking（VL 视觉推理需要思考链）
    if args.track == 'a':
        sp, mode = make_sampling_params(SamplingParams, SCHEMA)
        sp.temperature = 0
        sp.max_tokens = 1536
        chat_kwargs = {'chat_template_kwargs': {'enable_thinking': False}}
    else:
        sp = SamplingParams(temperature=0, max_tokens=4096)
        mode = 'prompt_only'
        chat_kwargs = {'chat_template_kwargs': {'enable_thinking': True}}
    print('structured output mode:', mode)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              enforce_eager=False, trust_remote_code=True,
              limit_mm_per_prompt={'image': 8},
              allowed_local_media_path=os.path.abspath(IMG_DIR))

    t0 = time.time()
    n_trip = 0
    for w_start in range(0, len(todo), args.window):
        batch = todo[w_start:w_start + args.window]
        convs = [(c, build_messages(c, args.track)) for c in batch]
        convs = [(c, m) for c, m in convs if m]
        try:
            outs = llm.chat([m for _, m in convs], sp, **chat_kwargs)
        except TypeError:
            outs = llm.chat([m for _, m in convs], sp)
        with open(out_path, 'a') as f:
            for (c, _), o in zip(convs, outs):
                raw = parse_json_loose(o.outputs[0].text)
                if raw is None:
                    continue
                for t in clean_triples(raw, c, extractor):
                    f.write(json.dumps(t, ensure_ascii=False) + '\n')
                    n_trip += 1
        done_n = min(w_start + args.window, len(todo))
        speed = done_n / (time.time() - t0)
        eta = (len(todo) - done_n) / speed / 60 if speed > 0 else 0
        print(f'[track {args.track}] {done_n}/{len(todo)} chunks, '
              f'{n_trip} triples, {speed:.1f} chunk/s, ETA {eta:.0f} min', flush=True)

    print(f'ALL DONE: {n_trip} triples -> {out_path}')


if __name__ == '__main__':
    main()
