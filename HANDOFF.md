# RAG-KG 项目交接说明

> 面向第三方接手者。读完本文应能：① 理解并复现整条流水线；② 知道当前图谱的实际效果；③ 清楚已知缺陷与改进方向。
> 工作目录：`/home/mountDisk2/user/guhao/code/RAG-KG/`

---

## 1. 项目目标与一句话方案

把《钢质海船入级规范》2026 综合文本（PDF，约 3,060 页）自动转成知识图谱并存入 Neo4j。

**方案**：MinerU 版面解析 → 跨页表/段落合并 → 分块（chunking）→ 构建混合检索索引（BM25 + bge-m3）→ 双轨大模型抽取三元组 → 实体/关系合并后写入 Neo4j。

- 文本/表格块 → **Qwen3-32B-FP8**（JSON schema 约束）
- 图片/图表块 → **Qwen3-VL-32B-Instruct-FP8**（直接看图抽取，无 caption 中间层）

---

## 2. 流水线总览

```
 PDF
  │  MinerU 解析（39 个 80 页分片文件夹）
  ▼
data/MinerU/                         ── 原始解析（含 origin.pdf 分片、HTML、图片）
  │ P0 rebuild_cn.py
  ▼
combined_content_list(_v2).json      ── v1 原子项 / v2 per-page
  │ P1 merge_cross_page_tables.py     ── 跨页表合并（union-find）
  │ P2 merge_cross_page_paragraphs.py ── 跨页段落合并
  ▼
merged_tables.json / merged_paragraphs.json
  │ P4 build_chunks.py                ── 分块，维护章/节路径
  ▼
chunks.jsonl (12,076 chunks)
  │ P5 table_to_markdown.py           ── HTML 表格 → markdown 写回
  │ P6 build_index.py                 ── BM25 + bge-m3 向量索引
  ▼
index/  +  chunks.jsonl（带 table_markdown）
  │ P7 extract_kg.py（双轨，vLLM 离线批推理）
  ▼
kg_out/triples_a_text_table.jsonl     ── 轨 A：文本/表格
kg_out/triples_b_image_chart.jsonl    ── 轨 B：图片/图表
  │ P8 load_kg.py
  ▼
Neo4j: 68,384 实体节点 / 67,790 关系边
```

> P3（VLM caption 中间层）已在设计阶段取消：最终由 Qwen3-VL 端到端看图抽三元组，不单独生成图片描述。

---

## 3. 目录结构

```
RAG-KG/
├── HANDOFF.md                     ← 本文
├── pipeline.md                    ← 内部执行日志/踩坑记录（细节补充）
│
├── rebuild_cn.py                  ← P0 从 MinerU 重建中文基础数据
├── merge_cross_page_tables.py     ← P1 跨页表合并
├── merge_cross_page_paragraphs.py ← P2 跨页段落合并
├── build_chunks.py                ← P4 分块
├── table_to_markdown.py           ← P5 表格 HTML→markdown
├── build_index.py                 ← P6 构建 BM25+dense 索引
├── retrieve.py                    ← 检索验证脚本（RRF 融合 + 可选重排）
├── extract_kg.py                  ← P7 双轨三元组抽取（支持断点续跑）
├── load_kg.py                     ← P8 三元组合并入 Neo4j
├── p7_run.sh                      ← P7 vLLM 启动封装（固化 CUDA13 环境）
│
├── data/
│   ├── MinerU/   (~787MB) 39 个规范分片解析 + 每分片 origin.pdf
│   ├── cn/        (~337MB) 流水线核心数据（见下）
│   └── en/        (~339MB) 英文版解析，本项目未使用（保留备用）
├── index/         BM25(bm25.pkl) + 向量(dense.npy) + doc_ids + meta
└── kg_out/        双轨三元组 jsonl（最终产出）
```

`data/cn/` 关键文件：

| 文件 | 说明 |
|---|---|
| `combined_content_list.json` | v1 原子项 59,856 条（table 字段 `table_body` HTML，1,506 表） |
| `combined_content_list_v2.json` | v2 per-page，3,063 页 list-of-lists；标题层级在 `content.level` |
| `merged_tables.json` | P1：253 组合并表 / 707 fragments，最长 19 页链 |
| `merged_paragraphs.json` | P2：271 组合并段落 |
| `chunks.jsonl` | P4/P5：12,076 chunk；table 块另带 `table_html`/`table_markdown`，image 块带 `img_paths` |
| `images/` | 7,339 张图（hash 文件名，JSON 中 `images/<hash>.jpg` 直接引用） |
| `*_report.txt`、`manifest.json` | 各阶段统计报告 |

---

## 4. 运行环境

### 4.1 硬件
- 本机 8× RTX 3090（24GB）。本项目实际占用：
  - 轨 A：GPU 2,3（TP=2）
  - 轨 B：GPU 0,1,2,3（TP=4，见 §7 踩坑）
- 3090（sm_86）**无原生 FP8**，vLLM 自动回退 Marlin weight-only 内核，速度略低于原生 FP8，属正常。

### 4.2 conda 环境
- **`rag-kg`**（P0/P5/P6/P8 + 检索）：torch 2.6.0+cu124、sentence-transformers、transformers、rank-bm25、jieba、pandas、beautifulsoup4、**neo4j 6.4.0**。
- **`rag-kg-vllm`**（P7）：vllm 0.31.0、torch 2.13.0+cu130、flashinfer 0.7.0.post1、ninja。
  - 本机无系统 CUDA toolkit，vLLM/flashinfer 首次运行需 JIT；所需 CUDA13 nvcc（统一 13.0.88）与 PATH 已在 `p7_run.sh` 固化，**直接用该脚本启动即可**，不要手动配环境。

### 4.3 模型（ModelScope 下载；本机 HuggingFace 不可达）
缓存于 `~/.cache/modelscope/models/`：
- `BAAI--bge-m3`（稠密向量，1024 维）
- `BAAI--bge-reranker-v2-m3`（重排）
- `Qwen--Qwen3-32B-FP8`（轨 A）
- `Qwen--Qwen3-VL-32B-Instruct-FP8`（轨 B）

### 4.4 数据库
- Neo4j 实例：**bolt://localhost:7688**（注意 7687 是他人实例，勿连）
- 账号 `neo4j` / 密码经环境变量 `NEO4J_PASSWORD` 注入（不写入代码与文档；向服务器管理员获取）
- 启动：`~/neo4j/bin/neo4j start`

---

## 5. 数据 Schema

**三元组**（`kg_out/triples_*.jsonl` 每行一条）：
```json
{
  "s": "防撞舱壁", "p": "应位于", "o": "干舷甲板",
  "s_type": "结构", "o_type": "结构",
  "source_chunk_id": "c_001234",
  "global_page": 185,
  "confidence": 0.95,
  "extractor": "Qwen3-32B-FP8",
  "extract_time": "2026-10-09T12:00:00"
}
```

**受控实体类型（10 类）**：设备、结构、检验、参数、材料、组织、标准、规范条款、过程、其他。

**Neo4j 图模型**：
- 节点：`(Entity {name, type})`；同名实体类型按**多数投票**决定。
- 边：`(Entity)-[REL {type, source_chunk_id, global_page, confidence, count}]->(Entity)`
  - `type` = 关系短语；`count` = 该 (s,p,o) 被多少个不同 chunk 支持（跨块证据数）。

---

## 6. 复现步骤

所有命令在 `RAG-KG/` 下执行。P0–P2 已完成且脚本幂等/可重复，**通常只需重跑 P4 之后的阶段**。

```bash
# P0 从 MinerU 重建中文数据（已完成，勿重跑）
~/.conda/envs/rag-kg/bin/python rebuild_cn.py

# P1 跨页表合并
~/.conda/envs/rag-kg/bin/python merge_cross_page_tables.py
# P2 跨页段落合并
~/.conda/envs/rag-kg/bin/python merge_cross_page_paragraphs.py

# P4 分块
~/.conda/envs/rag-kg/bin/python build_chunks.py
# P5 表格转 markdown（原位写回 chunks.jsonl）
~/.conda/envs/rag-kg/bin/python table_to_markdown.py
# P6 构建索引
~/.conda/envs/rag-kg/bin/python build_index.py

# P7 轨 A（文本/表格）——约 3.6h
bash p7_run.sh a Qwen--Qwen3-32B-FP8 2,3 --tp 2 --max-model-len 12288
# P7 轨 B（图片/图表）——约 47min
bash p7_run.sh b Qwen--Qwen3-VL-32B-Instruct-FP8 0,1,2,3 \
     --tp 4 --gpu-util 0.90 --max-model-len 8192
#   用法: bash p7_run.sh <a|b> <模型目录名> <gpu_ids> [额外参数]
#   extract_kg.py 按 chunk 分窗写盘、记录已处理 chunk，中断后重跑自动续跑。

# P8 入库（--wipe 先清空；先确认 Neo4j 7688 已启动）
~/.conda/envs/rag-kg/bin/python load_kg.py \
     --files kg_out/triples_a_text_table.jsonl kg_out/triples_b_image_chart.jsonl --wipe
```

**检索验证**（验证索引/重排链路，非必须）：
```bash
~/.conda/envs/rag-kg/bin/python retrieve.py "防撞舱壁的位置要求" --rerank
```
检索链路：BM25 top50 + dense top50 → RRF 融合 →（可选）bge-reranker-v2-m3 精排。

---

## 7. 当前效果

### 7.1 抽取与入库规模

| 指标 | 轨 A（文本/表格） | 轨 B（图片/图表） | 合计 |
|---|---|---|---|
| 输入 chunk | 10,678 | 1,398 | 12,076 |
| 有产出 chunk | 7,997 | 1,213 | 9,210 |
| 三元组 | 56,875 | 13,265 | **70,140** |
| 三元组/chunk | 5.33 | 10.93 | — |
| 耗时 / 速度 | 3.6 h / 0.8 chunk/s | 47 min / 0.5 chunk/s | — |

**Neo4j 最终图谱：68,384 节点 / 67,790 边。**
- 70,140 三元组在合并阶段去重 2,350 条；
- 1,496 条关系由多个 chunk 共同支持（`count>1`），最高 `count=124`；
- 0 个孤立节点；节点平均度数 1.98，最大 310；所有节点均有类型。

### 7.2 内容分布
- 节点类型 Top：参数 20,726、结构 12,799、规范条款 12,717、过程 7,639、其他 6,296、设备 3,607。
- 关系类型 Top：包含 7,942、定义为 7,626、适用于 3,470、应进行 3,057、应满足 2,683。
- 图片轨有效补出了文本轨拿不到的结构/示意信息（如装载计算仪示意、钢板换新区域、缺陷面积确定流程）。

### 7.3 质量印象（人工抽检，非严谨评测）
- 轨 A 三元组术语、规范编号保留较好，句式规范，是图谱主体可信来源。
- 轨 B 对真正含信息的示意图/流程图能抽取出结构化关系；对封面装饰图、空白表单能正确输出空数组（184 个图片块无产出）。

---

## 8. 已知缺陷与局限

> 以下为接手后应优先知悉/改进的问题，按影响排序。

1. **没有准确率评测，精度/召回未知（最大局限）。**
   全流程无人工标注金标准（gold set），所有质量判断仅来自少量人工抽检。`confidence` 是模型自评、未经校准（实际分布集中在 0.9–0.95，最低仅 0.7，区分度很低），不能直接当作可靠概率使用。**建议先抽样标注 200–500 条建立评测集再迭代。**

2. **存在受控词表外的实体类型。**
   轨 B 为 prompt-only（无 schema 约束），模型造了 10 类之外的类型。Neo4j 中共 **238 个越界类型节点**：示意图 187、公式 17、符号 15，另有少量 位置/条件/方向/单位 等。
   - 修复：轨 B 也启用 guided JSON；或在 `load_kg.py` 入库前把越界类型映射到受控类型（当前只做投票、未做归一）。

3. **图片轨含"画面描述"噪音。**
   部分三元组描述的是图像视觉元素而非规范事实，如"图 2.6.2 -包含-> 四个箭头指示圆角位置"。这类信息对规范问答价值低。可用规则或二次过滤剔除"箭头/角标/标注位置"类对象。

4. **实体仅按名称精确匹配，无实体链接/归一。**
   同一实体的不同字面写法（如"货舱"与"货物舱"、全称与简称、带不带编号）会被建成不同节点，造成碎片化与潜在重复。需要别名/同义词归并或实体消歧。

5. **绝大多数边只有单一证据。**
   67,790 条边中 **66,294 条（97.8%）count=1**，跨块交叉验证稀疏；高 count 关系多为高频通用关系（包含/定义为），对条款级事实的置信增强有限。

6. **跨页合并基于固定 bbox 阈值的启发式**，在排版异常页可能误合并或漏合并（段落合并还依赖"末段无终止标点"，中文标点使用不规范时会误判）。P1 中 227 组列数不符警告多数是"表格被识别为整页图片、仅首页有 HTML"所致，属数据特性而非合并错误，但也意味着这部分大表只有图片、无可用结构化文本。

7. **其他工程细节**
   - 无 tiktoken，token 数用"中文字符 + ASCII/4"估算，分块边界是近似值。
   - 约 119 个表格块无 markdown（空表或 HTML 字面量 `None`），这些表只能依赖轨 B 看图。
   - 硬编码了本机绝对路径与 Neo4j 口令；迁移到新机器需改 `p7_run.sh`、`build_index.py`/`retrieve.py` 的模型路径和 `load_kg.py` 的连接参数。

---

## 9. 建议的后续工作

1. 建小规模人工评测集，给出三元组级 Precision/Recall，作为后续迭代基线。
2. 轨 B 启用 schema 约束 + 入库前类型归一，消除越界类型节点。
3. 加实体归一/别名层（可先做规则，再考虑向量聚类）。
4. 过滤图片轨画面描述噪音；对 count=1 的关键条款关系考虑二次抽取确认。
5. 将路径、口令等配置抽到配置文件/环境变量，便于脱离本机复现。

---

## 附：关键文件速查

- 抽取逻辑与 Prompt：[extract_kg.py](file:///home/mountDisk2/user/guhao/code/RAG-KG/extract_kg.py)
- 入库（类型投票 + count 合并）：[load_kg.py](file:///home/mountDisk2/user/guhao/code/RAG-KG/load_kg.py)
- 分块：[build_chunks.py](file:///home/mountDisk2/user/guhao/code/RAG-KG/build_chunks.py)
- 检索：[retrieve.py](file:///home/mountDisk2/user/guhao/code/RAG-KG/retrieve.py)
- 内部执行/踩坑细节：[pipeline.md](file:///home/mountDisk2/user/guhao/code/RAG-KG/pipeline.md)
