# TraceRAG 的 Vibe Coding 开发说明

## 项目来源

项目以 Streamlit 官方 Apache-2.0 示例 `streamlit/example-app-langchain-rag` 为技术基线，保留 `LICENSE` 和 `NOTICE`。上游提供了 BM25 与向量集成检索的教学思路；本项目围绕可追溯、可测试、可实评进行了模块化二次开发。

## 我的工作

- 拆解需求并确定文档入库、双路召回、RRF、重排、证据生成、引用校验、评测、CLI 与 API 的模块边界。
- 选择 BM25 + BGE-M3/FAISS + RRF + CrossEncoder 技术路线，定义默认切片参数、融合参数、返回契约和验收条件。
- 审查 AI Coding 生成的主要实现，运行单元、回归和 API 测试，并判断结果是否满足预期。
- 安装正式模型依赖，在 Python 3.12 CPU 环境实际运行 BGE-M3、FAISS 和 bge-reranker-v2-m3，记录模型版本、向量维数、实际后端、缓存体积和耗时。
- 通过分层指标定位 Markdown 空行导致正文丢失的问题，修复证据分隔协议并补充回归测试。
- 维护上游归属、许可证和评测边界，不把合成烟测结果外推为真实业务收益。
- 将固定检索链升级为受约束 Agent：定义三类检索工具、路由规则、证据充分性条件、查询改写、有界重试和审计轨迹，并通过故障场景验收停止策略。

## AI Coding 的工作

AI Coding 根据需求和技术路线生成了主要实现代码、部分测试与文档草稿。我负责审查、运行、故障复现、修改验收和最终结果判断。

## 已验证证据

- 52 项自动化测试通过，覆盖结构化工具 Schema、Agent 路由、查询改写、有界重试、精确标识符跨轮约束、文件名来源匹配、Agent 决策指标、API 模型配置、实际后端披露、零证据拒答和重复来源评测去重，并加入可离线运行的 GitHub Actions。
- FastAPI `/health -> /index -> /ask` 已分别实跑离线默认链路和 BGE-M3 + FAISS + CrossEncoder 正式模型链路。
- `/health -> /index -> /ask` FastAPI 自动化调用通过。
- BGE-M3 输出 1024 维 `float32` 归一化向量。
- 正式后端记录为 `bm25+BAAI/bge-m3+faiss+BAAI/bge-reranker-v2-m3`。
- 模型缓存后的 5 题 CPU 运行耗时为 10.25 秒。
- 5 题合成烟测的五项汇总指标均为 1.0；Hashing 基线同样为 1.0，因此不声称正式模型带来指标提升。
- Ollama/Qwen 生成链已完成适配但未实际运行；当前答案指标来自离线抽取式生成器。
