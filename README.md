# TraceRAG：可追溯 Agentic RAG 知识库

这是一个面向本地技术文档、可离线测试的个人 Vibe Coding 项目。它以 Streamlit 官方 Apache-2.0 示例项目 [`streamlit/example-app-langchain-rag`](https://github.com/streamlit/example-app-langchain-rag) 的“BM25 + 向量集成检索”思路为技术基线，重新拆分为文档加载、切片、检索工具、Agent 决策、RRF、可选重排、生成、引用校验、评测、CLI 和 API 模块。

## 开发方式与个人贡献

本项目采用 Vibe Coding：我负责需求拆解、上游与许可证核验、技术路线和模块接口设计、验收条件、关键代码审查、运行测试、模型实评、故障定位与结果判断；AI Coding 生成主要实现代码。项目保留上游 Apache-2.0 的 `LICENSE` 与 `NOTICE`，不把二次开发表述为从零手写。

这次改造的目标不是简单增加界面功能，而是把教学示例变成一条可复现、可观测、可回归的 RAG 工程链路。详细分工与证据见 `docs/vibe-coding-contribution.md`。

## 我做了哪些改造

上游是教学型示例，主要展示 LangChain 的 EnsembleRetriever。本项目增加并验证了以下内容：

- Markdown、TXT、JSONL 本地文档加载。
- 默认 `chunk_size=420`、`chunk_overlap=60` 的确定性切片。
- 基于来源、字符位置和内容生成稳定的 16 位 Chunk ID。
- 自实现 BM25 和离线字符 n-gram Hashing Embedding，CI 不需要下载模型。
- 可选 BGE-M3、FAISS 和 `BAAI/bge-reranker-v2-m3` 适配器；实际后端会写入返回结果。
- BM25 与稠密检索通过 RRF 融合，默认权重 `0.5/0.5`、常数 `k=60`。
- 回答只能引用本次检索返回的 Chunk ID；不存在的引用会被标为无效。
- 离线抽取式生成器，以及可选的 LangChain + Ollama/Qwen 生成链。
- Recall@K、MRR、nDCG@K、Required Term Coverage、Citation Validity 评测。
- CLI 和懒加载 FastAPI 边界。
- 针对 Markdown 空行导致正文丢失的回归测试。
- 受约束的离线 Agent 控制器：根据问题特征选择 BM25、Dense 或 Hybrid 检索工具。
- 为 `lexical_search`、`dense_search`、`hybrid_search` 定义结构化 Tool Call Schema，通过注册器执行真实工具；Trace 记录工具名、参数和耗时。
- 证据不足时执行查询清理、检索范围扩展和最多两轮的有界重试；仍不足则拒答。
- 返回每轮查询、路由、证据数量、证据 ID、判断原因、动作和最终停止原因。

## 四项岗位要求对应关系

1. **业务场景**：面向 AI 学习与面试准备的本地技术知识库；输入是技术问题与本地文档，输出是带 Chunk ID 证据的回答或明确拒答。
2. **工具调用**：Agent 根据问题与中间证据选择三类检索工具，工具参数经过 Schema 校验；首轮不足时执行 Hybrid 扩展，最多两轮。
3. **工程设计**：文档切分、检索、融合、重排、生成、引用校验、API 和评测分层；包含稳定 ID、后端披露、故障处理、Trace、停止条件、单元/集成/API 回归和离线 CI。
4. **量化结果**：保留 5 题正式模型烟测，同时新增 11 题固定 Pipeline 与 Agent 的 Hashing 行为对照；两者数据集、后端和结论分开披露。

## 数据流

```text
MD/TXT/JSONL
    -> 文档加载与稳定切片
    -> Agent 问题规划
       -> 精确标识符/路径/版本号：BM25
       -> 概念与原理问题：Dense
       -> 一般问题：Hybrid
    -> RRF 排名融合（Hybrid 路径）
    -> Identity/CrossEncoder 重排
    -> 证据充分性判断
       -> 不足：查询改写/扩展为 Hybrid，最多重试 1 次
       -> 仍不足：停止并拒答
    -> 证据上下文与 Chunk ID
    -> Extractive 或 LangChain Ollama 生成
    -> 引用合法性校验
```

## 快速开始

Python 3.11 及以上：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
pytest -q
```

离线问答：

```powershell
hybrid-rag ask "错误码 E101 应该如何处理？"
```

离线评测：

```powershell
hybrid-rag evaluate --output data/evaluation/results.json
```

使用 LangChain + Ollama：

```powershell
python -m pip install -e .[langchain]
hybrid-rag ask --generator ollama --model qwen2.5:7b "RRF 为什么不用原始分数相加？"
```

使用 BGE/FAISS/CrossEncoder 时安装模型依赖；第一次运行会下载相应模型：

```powershell
python -m pip install -e .[models]
hybrid-rag ask --embedding bge-m3 --rerank "为什么 CrossEncoder 只处理少量候选？"
```

模型已经缓存后，可以禁止启动时访问 Hugging Face：

```powershell
hybrid-rag evaluate --embedding bge-m3 --rerank --local-files-only
```

启动 API：

```powershell
python -m pip install -e .[api]
uvicorn hybrid_rag.api:create_app --factory --host 127.0.0.1 --port 8000
```

先调用 `POST /index`：

```json
{
  "path": "data/demo",
  "top_k": 3,
  "generator": "extractive",
  "embedding": "bge-m3",
  "embedding_model": "BAAI/bge-m3",
  "device": "cpu",
  "batch_size": 8,
  "min_dense_score": 0.0,
  "max_agent_iterations": 2,
  "rerank": true,
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "local_files_only": true
}
```

`embedding` 仅接受 `hashing` 或 `bge-m3`，`generator` 仅接受 `extractive` 或 `ollama`，非法配置返回 HTTP 400。`min_dense_score` 设置稠密召回的最低证据分，默认 `0.0`，即丢弃非正分候选；不同模型的合理阈值必须通过验证集校准，项目不声称存在通用 BGE 阈值。BM25 同样不再返回零分文档。`max_agent_iterations` 是正整数，默认 2；Agent 不允许无限循环。两轮仍没有足够证据时明确返回“未检索到足够证据，已停止回答”。索引响应披露检索、生成后端与 Agent 配置；问答响应增加 `agent_trace` 和 `stop_reason`，避免 Agent 决策成为黑盒。

运行固定 Pipeline 与 Agent 对照：

```powershell
hybrid-rag evaluate --compare-fixed --embedding hashing --output data/evaluation/results_agent_comparison.json
```

默认 Agent 是可复现的规则策略，不调用外部规划大模型。它已经具备 Agentic RAG 的核心运行特征——动态工具选择、读取中间证据、条件分支、有界重试和可审计停止——但不应表述成具备开放式任务规划能力的通用自主 Agent。

再调用 `POST /ask`：

```json
{"question": "错误码 E101 应该如何处理？"}
```

## 当前已验证结果

2026-08-15 已在隔离的 Python 3.12 CPU 环境完成两条 CLI 评测链路的实际运行：

- `data/evaluation/results.json`：BM25 + Hashing + FAISS + Identity Reranker。
- `data/evaluation/results_bge_m3_reranker.json`：BM25 + BGE-M3 + FAISS + `bge-reranker-v2-m3`。

正式模型链路的返回后端为 `bm25+BAAI/bge-m3+faiss+BAAI/bge-reranker-v2-m3`。BGE 输出为 1024 维 `float32` 归一化向量；模型缓存后的 5 题 CPU 运行耗时为 10.25 秒。

同日还通过 FastAPI TestClient 实际完成正式模型 `/index -> /ask`：索引 5 篇文档、5 个 Chunk，`/ask` 返回 3 条证据且引用合法，API 披露的检索后端与上述正式模型组合一致。这里验证的是服务链路可运行，不把这次本地调用描述为并发或延迟压测。

| 指标 | 结果 |
|---|---:|
| 问题数 | 5 |
| Recall@3 | 1.0000 |
| MRR | 1.0000 |
| nDCG@3 | 1.0000 |
| Required Term Coverage | 1.0000 |
| Citation Validity | 1.0000 |
| Agent 平均迭代轮数 | 1.0000 |
| Agent 拒答率 | 0.0000 |

本次 5 题 Agent 路由分布为 lexical 2 题、dense 2 题、hybrid 1 题；所有问题首轮均获得证据，因此这组烟测没有覆盖真实的二轮改写收益。查询改写、扩展检索与达到上限后拒答由自动化故障测试覆盖，仍需要补充真实无答案和首轮失败数据集。

两条链路在这组数据上的五项汇总指标均为 1.0。BGE-M3 和 CrossEncoder 没有在这个简单烟测集上带来可见指标提升；这 5 个问题和 5 篇短文档只用于验证数据流与回归行为，不代表真实用户效果。Ollama/Qwen 生成链路仍未运行，因为当前机器没有安装 Ollama；上述答案指标来自离线抽取式生成器。

### 11 题 Agent 行为对照

2026-08-23 在项目完整环境中使用 Hashing/FAISS/Identity 与离线抽取式生成器实跑 11 题，其中 8 题可回答、3 题为不存在的错误码/文件/版本。固定 Hybrid Pipeline 的任务成功率为 `0.7273`、无答案拒答 F1 为 `0.0000`；受约束 Agent 的任务成功率、工具选择准确率、响应引用安全率和拒答 F1 均为 `1.0000`，平均工具调用轮数为 `1.2727`。两者 Recall@3 和 nDCG@3 均为 `0.7273 / 0.6937`，说明本次收益来自精确标识符约束和安全拒答，而不是可回答题的排序提升。CI 未安装 FAISS 时会走 NumPy，但结果文件始终披露实际后端。

结果保存在 `data/evaluation/results_agent_comparison.json`。该数据集是合成回归基准，`human_gold=false`；不能与上面的 5 题 BGE-M3/CrossEncoder 正式模型烟测混报，也不能据此声称真实用户效果提升。

完整环境与耗时记录见 `data/evaluation/model_run_metadata.json`。Hugging Face 缓存约占 6.39 GB，其中 BGE-M3 约 4.25 GB、重排模型约 2.14 GB。

## 自动化测试与 CI

2026-08-23 在项目完整环境中本地回归为 `52 passed`，覆盖文档加载与切片、Embedding、三类结构化检索工具、Agent 路由与有界重试、精确标识符跨轮约束、文件名来源匹配、Agent 行为指标、零证据拒答、RRF 与重排、Pipeline 与引用、API 配置和完整服务流程。`.github/workflows/ci.yml` 在 push 和 pull request 时使用 Python 3.11 运行编译检查、完整离线测试和 11 题 Agent 对照评测；CI 固定走 Hashing/Identity 或假模型，不下载数 GB 的正式模型。

## 项目结构

```text
src/hybrid_rag/
  ingestion.py      文档加载、切片和稳定 ID
  embeddings.py     Hashing 与 BGE 适配器
  retrieval.py      BM25 与 NumPy/FAISS 稠密检索
  fusion.py         RRF 与混合检索编排
  agent.py          问题路由、证据判断与查询改写策略
  reranking.py      Identity 与 CrossEncoder 重排
  generation.py     抽取式与 LangChain Ollama 生成
  pipeline.py       固定 Pipeline 与有界 Agentic Pipeline
  evaluation.py     检索、答案和引用指标
  cli.py            命令行入口
  api.py            FastAPI 工厂与返回契约
tests/              离线单元与回归测试
data/demo/          5 篇合成技术文档
data/evaluation/    5 题烟测与实测结果
docs/               架构、实施计划和面试口径
```

## 许可证与归属

本项目保留上游来源说明，采用 Apache License 2.0。详见 `LICENSE` 和 `NOTICE`。上游提供了混合检索教学思路；本地模块化实现、证据协议、评测、API、离线后端和测试属于本次改造。
