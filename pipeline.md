# RAG-KG Pipeline 交接文档

> 目标：从《钢质海船入级规范》2026 PDF 用 RAG 辅助 LLM 抽取 KG 存入 Neo4j。
> 工作目录：`/home/mountDisk2/user/guhao/code/RAG-KG/`
> 新窗口开场白：「读 pipeline.md，按进度继续执行」。

## 进度状态

| 阶段 | 状态 | 产出 |
|---|---|---|
| P0 数据重建 | ✅ | `data/cn/` 全部基础数据 |
| P1 跨页表合并 | ✅ | `merged_tables.json`（253 组 / 707 fragments，最大 19 页链） |
| P2 跨页段落合并 | ✅ | `merged_paragraphs.json`（271 组，全部 2 页；滤掉 7 组目录噪音） |
| ~~P3 VLM caption~~ | ❌ 已取消 | 双轨方案定稿后由 Qwen3-VL-32B 端到端看图抽三元组，无 caption 中间层 |
| P4 Chunking | ✅ | `chunks.jsonl`（12,076 chunks） |
| P5 HTML→markdown | ✅ | `chunks.jsonl` 原位写回 `table_markdown`（1038 转换 / 119 空表 / 0 失败） |
| P6 索引构建 | ✅ | `index/`（bm25.pkl + dense.npy 12076×1024 + doc_ids.json + meta.json） |
| P7 双轨 KG 抽取 | ✅ | 轨 A 56,875 三元组（7,997 chunk）；轨 B 13,265 三元组（1,213/1,397 chunk）；合计 70,140 |
| P8 Neo4j 入库(user neo4j，密码经 `NEO4J_PASSWORD` 环境变量注入，不入库) | ✅ | 68,384 节点 / 67,790 边（70,140 三元组→2,350 跨 chunk 重复合并）；1,496 合并关系 max count=124 |

## 数据文件（data/cn/）

- `combined_content_list.json` — v1 原子项 59,856 条；table 字段 `table_body`(HTML)，1,506 表
  - table keys: `type, img_path, table_caption[], table_footnote[], table_body, bbox, page_idx, _source_folder, _page_start, global_page`
  - caption/footnote 元素是**纯字符串**
- `combined_content_list_v2.json` — v2 per-page，3,063 页 list-of-lists
  - type 分布: paragraph 39,510 / title 8,611 / equation_interline 3,516 / page_number 3,054 / page_header 1,847 / table 1,506 / image 1,293 / page_footnote 331 / chart 105 / list 66 / algorithm 15 / page_footer 2
  - table 在 `content.html` 下；caption/footnote 元素是 **dict** `{'type':'text','content':...}`（build_chunks.py 的 `cap_text()` 已处理两种形态）
  - title 层级在 `content.level`（1=章 110 个 / 2=节 8,501 个）
- `merged_tables.json` — P1 产出。每组: `merged_id, merge_kind, global_pages, page_span, merged_html, img_paths, table_caption, table_footnote, fragments[{global_page, bbox, source_folder, ...}], warnings`
- `merged_paragraphs.json` — P2 产出。每组: `merged_id, global_pages, text, fragments[{global_page, idx_in_page, bbox, source_folder}]`
- `chunks.jsonl` — P4 产出。每行: `chunk_id, chunk_type(text/table/image/chart), text, section_path[], global_pages[], source_items[]`；table chunk 另带 `table_html, table_caption, table_footnote, img_paths, merged_table_id?`；image/chart 另带 `img_paths, image_caption, image_footnote`
- `images/` — 7,339 张图（hash 文件名，JSON 中 `images/<hash>.jpg` 直接引用）
- `cross_page_tables.json`、`manifest.json`、`rebuild_report.txt`、`merge_tables_report.txt`、`merge_paragraphs_report.txt`、`chunking_report.txt`

## 脚本（RAG-KG/ 下）

- `rebuild_cn.py` — P0 已完成，勿重跑
- `merge_cross_page_tables.py` — P1。union-find 归组（451 对 + 跨 folder 启发式 3 组，1 组列数不符拒绝）；bs4 拼接 HTML 并去续表重复表头（head 前 2 行文本匹配）
- `merge_cross_page_paragraphs.py` — P2。判定：页末段 bottom>75% + 下页首段 top<25% + 末段无终止标点；全组含 `....` 判为目录跳过
- `build_chunks.py` — P4。流式遍历 v2，title 维护 section_path；merged 单元整组消费、后续 fragment 跳过；text chunk 上限 1024（`est_tokens` 中文按字符估算，**无 tiktoken**），超长单段按句子软切重叠 1 句
- `table_to_markdown.py` — P5。`pandas.read_html`（rowspan 下填充/colspan 右复制）+ tabulate 转 markdown 写回 `table_markdown`；空表头（数据无 `<th>`）、`|`→全角｜、空表前置检测；报告 `data/cn/p5_report.txt`
- `build_index.py` — P6。BM25（jieba+rank_bm25）+ bge-m3 dense（fp16, batch 16, 8192 tokens, 归一化）；索引文本：text chunk 用 `text`，table/chart 用 `text`+`table_markdown`；空 chunk 不入索引（入索引 12,076 条）
- `retrieve.py` — P6 检索验证。双路 top50 → RRF 融合 → 可选 `--rerank`（bge-reranker-v2-m3，CrossEncoder，输入用同款 index_text）

## 已定稿设计（P5-P8 必须遵守）

- **表格双写**：markdown 进索引 + HTML 进 metadata。P5 用 `pandas.read_html`（rowspan 下填充）把每个 table chunk 的 `table_html` 转 markdown，写回 chunk（如 `table_markdown` 字段）
- **检索**：BM25 + bge-m3 dense 统一索引 + bge-reranker-v2-m3 重排
- **抽取双轨**：
  - text/table chunk → **Qwen3-32B-FP8**（vLLM, temp=0, JSON schema 引导；设计原定 Qwen3-27B，但 ModelScope 无 27B，2026-10-09 定稿替换为 32B-FP8）
  - image/chart chunk（1,398 张）→ **Qwen3-VL-32B-Instruct-FP8**（端到端看图抽三元组，不走 caption）
- **三元组**：`{s, p, o, s_type, o_type, source_chunk_id, global_page, confidence, extractor, extract_time}`
- **Neo4j**：节点 `(Entity {name, type})`（同名实体类型多数投票），边 `-[REL {type, source_chunk_id, global_page, confidence, count}]-`（count=跨 chunk 重复证据数）

## 环境备忘

- Python 环境已有 bs4 4.12、pandas 2.2；**无 tiktoken**（用字符估算，勿装）
- **conda env `rag-kg`**（P6 起，克隆自 magix）：`~/.conda/envs/rag-kg`，torch 2.6.0+cu124 / sentence-transformers 6.1.0 / transformers 5.17.0 / rank-bm25 0.2.2 / jieba / modelscope；CUDA 验证通过（8×3090）
- **模型缓存**（ModelScope 下载，HF 不可达）：`~/.cache/modelscope/models/BAAI--bge-m3`、`BAAI--bge-reranker-v2-m3`
- **P6 教训**：bge-m3 编码 batch 64 会 OOM（长表 markdown 接近 8k tokens 时单次 attention 要 16GB），fp16 + batch 16 即可（12,076 条 69s）
- 本机 8×RTX 3090；GPU 4-7 常被占用，0-3 空闲；P7 起 vLLM 前先用 **gpu-torch-bootstrap skill** 确认 GPU/vLLM 环境
- 数据库：Neo4j **bolt://localhost:7688**（本机 `~/neo4j` 实例，注意 7687 是他人实例勿连） user:neo4j，密码见环境变量 `NEO4J_PASSWORD`；已验证连通、库为空
- **conda env `rag-kg-vllm`**（P7）：`~/.conda/envs/rag-kg-vllm`，vllm 0.31.0 / torch 2.13.0+cu130 / flashinfer 0.7.0.post1 / ninja；neo4j 6.4.0 装在 rag-kg（P8 用）
- **P7 启动**：`bash p7_run.sh <a|b> <Qwen--模型目录名> <gpu_ids> [额外参数]`（已固化 CUDA_HOME/PATH）；轨 A：GPU 2,3 TP=2（已完成）；轨 B：**TP=4 GPU 0,1,2,3**（以下为踩坑后定稿配置）
  - ~~TP=3~~ 不可行：Qwen3-VL-32B 有 64 attention heads，64%3≠0（VllmConfig ValidationError）
  - ~~TP=2~~ 不可行：模型权重 17.5 GiB/卡，KV cache 仅 0.36 GiB，max-model-len 8192 需 1.0 GiB（ValueError）
  - **TP=4 可行**：权重 9.02 GiB/卡，KV cache 9.09 GiB，最大并发 18.18x
  - vLLM 0.31 新增安全限制：加载本地 `file://` 图片必须设 `allowed_local_media_path`（已在 extract_kg.py 中添加 `LLM(allowed_local_media_path=os.path.abspath(IMG_DIR))`）
  - **轨 B 最终启动命令**：`bash p7_run.sh b Qwen--Qwen3-VL-32B-Instruct-FP8 0,1,2,3 --tp 4 --gpu-util 0.90 --max-model-len 8192`
  - 注意：GPU 1 偶有他人小任务（~952 MiB），TP=4 时 vLLM 仍可正常运行（实测无影响）
- **P7 CUDA13 JIT 踩坑（本机无系统 CUDA toolkit，vLLM/flashinfer 首次运行要 JIT）**：
  - pip 的 nvcc 在 `envs/rag-kg-vllm/lib/python3.10/site-packages/nvidia/cu13/{bin,include,lib,nvvm}`，需 `export CUDA_HOME=<该cu13目录>` 且 PATH 前置其 bin
  - pip 默认装的 nvidia-cuda-nvcc/nvidia-nvvm/nvidia-cuda-crt 是 **13.4**，与 nvidia-cuda-runtime 13.0 不匹配 → 必须三件套统一 **13.0.88**（`nvidia-cuda-crt==13.0.88 nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88`），否则 cccl 报 #error 或 PTX 9.4 vs ptxas 9.0
  - 另装 `ninja`；建软链 `cu13/lib/libcudart.so→libcudart.so.13`、`cu13/lib64→lib`（flashinfer 链接走 -L lib64 -lcudart）
  - triton/tokenspeed_triton 自带旧 ptxas（12.8/12.9）会被 worker 优先调用报 "Unsupported .version 9.4" → 把这两个 `backends/nvidia/bin/ptxas` 软链到 cu13/bin/ptxas
- **P7 FP8 备注**：3090（sm_86）无原生 FP8，vLLM 自动用 Marlin weight-only 内核（日志有 WARNING，正常）；首次 torch.compile+flashinfer JIT 约 5-8 分钟，之后缓存于 `~/.cache/flashinfer`
- **P7 脚本**：`extract_kg.py`（分窗 256 断点续跑、track a 用 structured output + no thinking，track b 用 prompt_only + thinking 开启）；`p7_run.sh` 启动封装；输出 `kg_out/triples_{a_text_table,b_image_chart}.jsonl`
  - **轨 B prompt 优化（踩坑后定稿）**：
    - 专用 `SYS_PROMPT_B`（带表单/流程图/示意图抽取示例）；track a 用原 `SYS_PROMPT`
    - 正确提取 `image_caption`/`image_footnote` 字段（dict 格式），不再用 `text`（只是 `(image)` 占位符）
    - `enable_thinking=True`（VL 视觉推理需要思考链）；track a 保持 `enable_thinking=False`
    - 无 structured output（prompt_only 模式），让模型自由思考后输出 JSON；`parse_json_loose` 已增加 `</think>` 标签剥离逻辑
  - 1,398 张 image/chart，953 张有 caption（仅图号如"图 2.1"，非描述性），445 张无 caption；图片多为小文件（15-25KB），含表单模板/示意图/流程图
  - **轨 B 全量结果（2026-10-09 完成）**：`kg_out/triples_b_image_chart.jsonl` 13,265 三元组，覆盖 1,213/1,397 chunk（10.93 条/chunk）；耗时约 47 min（0.5 chunk/s，GPU 0-3 TP=4）；184 chunk 产出 0 三元组（封面装饰图/空白图）；类型分布：s_type 以 结构(6583)/规范条款(2271)/参数(2171) 为主，o_type 以 结构(4995)/参数(4855) 为主；注意 prompt_only 模式下模型偶尔产出非标准类型（如"示意图""符号""公式"等），load_kg.py 的 majority_type 会自动处理
- **P7 轨 A 结果（2026-10-09 完成）**：`kg_out/triples_a_text_table.jsonl` 56,875 三元组，覆盖 7,997/10,678 chunk（5.33 条/chunk）；耗时约 3.6h（0.8 chunk/s，GPU 2,3 TP=2）；类型分布：参数 31,894 / 规范条款 28,563 / 结构 17,935 / 过程 10,642 / 其他 9,825 / 设备 6,365 / 检验 2,828 / 标准 2,489 / 组织 1,773 / 材料 1,436
- **P7 已完成**：① 轨 B 全量 ✅（13,265 三元组）→ ② 双轨统计 ✅ 已补写 → ③ P8 待执行（见下）

## 已知数据特性（不是 bug，勿再排查）

- 续表 fragment 的 `table_body` 常为空（表格识别为整页图片，仅首 fragment 有 HTML）→ col_mismatch 警告 227 组属正常 rowspan/空表解析差异
- `chunks.json` 中 text chunk min tokens=0：个别空标题 chunk，影响可忽略
- 跨 folder 相邻表共 12 处，bbox 启发式通过 4 处，其中 1 处列数 0 vs 14 已拒绝（空表）
- 4 个 table chunk 的 `table_html` 是字面量 `'None'` 字符串、2 个全空单元格表 → P5 按空表处理，共 119 个 `table_markdown=''`
- max_rows=578 / max_cols=33（合并大表），markdown 表头行为空（无 `<th>`）
