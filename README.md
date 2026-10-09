# RAG-KG

从《钢质海船入级规范》2026 综合文本（约 3,060 页 PDF）自动构建知识图谱：MinerU 版面解析 → 跨页表/段落合并 → 分块 → 混合检索索引 → 双轨大模型抽取三元组 → Neo4j 入库。

最终图谱：**68,384 实体节点 / 67,790 关系边**（70,140 条原始三元组合并去重）。

## 方案

双轨抽取，按 chunk 类型分流：

| 轨道 | 输入 | 模型 | 三元组 | 耗时 |
|---|---|---|---|---|
| A 文本/表格 | 10,678 chunks | Qwen3-32B-FP8（JSON schema 约束） | 56,875 | ~3.6 h |
| B 图片/图表 | 1,398 chunks | Qwen3-VL-32B-Instruct-FP8（端到端看图） | 13,265 | ~47 min |

推理硬件：8× RTX 3090，vLLM 离线批推理（轨 A TP=2，轨 B TP=4）。

## 流水线

```text
PDF (3,063 页)
  │  MinerU 版面解析（39 个 80 页分片）
  ▼
P0 rebuild_cn.py ──► 基础数据 combined_content_list(_v2).json
  │
  ├─ P1 merge_cross_page_tables.py ──► merged_tables.json（253 组，最长 19 页链）
  └─ P2 merge_cross_page_paragraphs.py ──► merged_paragraphs.json（271 组）
  ▼
P4 build_chunks.py ──► chunks.jsonl（12,076 chunks，含章节路径）
  │
  ├─ P5 table_to_markdown.py ──► 表格 HTML→markdown 写回
  └─ P6 build_index.py ──► index/（BM25 + bge-m3，1024 维）
  ▼
P7 extract_kg.py（双轨，支持断点续跑）──► kg_out/triples_*.jsonl
  ▼
P8 load_kg.py ──► Neo4j (Entity {name,type}) -[REL {…,count}]->
```

## 仓库结构

```text
├── HANDOFF.md          完整交接文档（环境、复现步骤、Schema、效果、缺陷）
├── pipeline.md         内部执行日志与踩坑记录
├── *.py / p7_run.sh    P0–P8 流水线脚本 + vLLM 启动封装
├── data/cn/            中文核心数据（chunks、合并结果、7,339 张图片）
├── index/              BM25 + dense 检索索引
└── kg_out/             双轨三元组产出
```

## 快速开始

环境搭建与逐阶段复现命令见 [HANDOFF.md](HANDOFF.md)。常用命令：

```bash
# 构建索引
~/.conda/envs/rag-kg/bin/python build_index.py

# 检索验证（BM25+dense RRF 融合，可选重排）
~/.conda/envs/rag-kg/bin/python retrieve.py "防撞舱壁的位置要求" --rerank

# 三元组入库 Neo4j（bolt://localhost:7688）
~/.conda/envs/rag-kg/bin/python load_kg.py \
    --files kg_out/triples_a_text_table.jsonl kg_out/triples_b_image_chart.jsonl --wipe
```

## 说明

- 本仓库**不含** `data/MinerU/`（781MB 原始解析，可用 MinerU 重新生成）与 `data/en/`（339MB 英文版解析，本项目未使用）；克隆后如需从零复现请先补充原始 PDF 并重新解析。
- 实体类型为受控 10 类：设备、结构、检验、参数、材料、组织、标准、规范条款、过程、其他。
- 已知缺陷（无金标准评测、图片轨类型越界、实体未归一等）与改进建议见 [HANDOFF.md](HANDOFF.md)。
- 部分脚本含本机路径与数据库口令，迁移部署前需修改（HANDOFF.md §7）。
