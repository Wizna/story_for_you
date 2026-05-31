# Story For You

基于 LLM 的中文网络小说处理工具，解决长篇小说阅读中的常见痛点：内容冗长、角色繁杂、结局不满意。

## 功能特性

- **剧情压缩** - 智能提取核心剧情，去除水分内容，保留 90%+ 原文风格
- **角色筛选** - 只保留指定角色相关的剧情线，快速追踪 CP 或主角故事
- **角色删除** - 从文本中移除不喜欢的角色，自动修补叙事连贯性
- **结局续写** - 四阶段创作流程（构思大纲→初稿→修订润色→伏笔检查），生成符合原作风格的新结局

## 设计准则

- **LLM 负责语义**：人物、事件、关系、故事状态、读者 hint、结局审查、删除/桥接文本都交给 LLM 判断。
- **Python 负责流程**：本地代码只做分块、排序、缓存、索引、去重、prompt 渲染、JSON/schema 校验和错误上报。
- **不做语义兜底**：LLM 不可用、返回空内容、坏 JSON 或缺少必需字段时直接失败，不用关键词、正则、计数或默认值伪造结果。
- **大上下文不等于粗粒度**：即使模型支持 1M 上下文，分析仍按稳定语义单元执行，再聚合成全局上下文。

## 安装

```bash
# 克隆仓库
git clone git@github.com:Wizna/story_for_you.git
cd story_for_you

# 安装依赖（需要 Python 3.11+）
uv sync

# 验证安装
uv run story --help
```

### 前置要求

- Python 3.11+
- 可用的 LLM 后端。默认使用 DeepSeek API；也可以显式切换到其他 OpenAI-compatible API 或本地 Ollama，但不会在 LLM 不可用时走本地语义兜底。

```bash
# 方式一：云端 DeepSeek API（默认）
export DEEPSEEK_API_KEY=sk-xxx

# 方式二：本地 Ollama
export STORY_LLM__PROVIDER=ollama
export STORY_LLM__MODEL=qwen3.5:9b
export STORY_LLM__BASE_URL=http://localhost:11434
ollama pull qwen3.5:9b
```

## 快速开始

```bash
# 1. 分析小说（首次运行，会自动缓存结果）
uv run story analyze novel.txt -o analysis.json

# 2. 使用分析结果进行处理
uv run story compress novel.txt --level medium      # 压缩剧情
uv run story filter novel.txt -c "张三,李四"        # 筛选角色
uv run story remove novel.txt -c "王五"             # 删除角色
uv run story continue novel.txt --hint "希望是HE"   # 续写结局
```

## 命令详解

### `analyze` - 故事分析

分析小说结构，提取人物、事件、关系和写作风格，生成 `StoryContext`。

```bash
uv run story analyze novel.txt -o analysis.json
uv run story analyze novel.txt --no-resume    # 禁用断点续传
uv run story analyze novel.txt --format yaml  # 输出 YAML 格式
```

分析结果会自动缓存到 `.story_cache/`，后续命令直接复用。

### `compress` - 剧情压缩

根据压缩级别精简内容，保留核心剧情。

| 级别   | 保留率 | 说明                       |
| ------ | ------ | -------------------------- |
| light  | 80%    | 轻度压缩，去除明显水分     |
| medium | 50%    | 中度压缩（默认），保留主线 |
| heavy  | 30%    | 重度压缩，仅保留核心情节   |

```bash
uv run story compress novel.txt --level light   # 轻度压缩 (80%)
uv run story compress novel.txt --level medium  # 中度压缩 (50%，默认)
uv run story compress novel.txt --level heavy   # 重度压缩 (30%)
```

### `filter` - 角色筛选

只保留与指定角色相关的内容。

```bash
uv run story filter novel.txt -c "张三,李四" --mode soft  # 软筛选：保留相关场景
uv run story filter novel.txt -c "张三,李四" --mode hard  # 硬筛选：仅保留直接互动
```

### `remove` - 角色删除

从文本中移除指定角色，自动修补叙事。

```bash
uv run story remove novel.txt -c "王五" --mode hard  # 硬删除：完全移除（默认）
uv run story remove novel.txt -c "王五" --mode soft  # 软删除：弱化存在感
```

### `continue` - 结局续写

基于原作风格续写新结局，先由 LLM 解析读者 hint 为结构化指令，再采用四阶段创作流程，最终由 LLM reviewer 审查用户约束、剧情一致性和结局闭合度：

| 阶段     | 温度 | 说明                           |
| -------- | ---- | ------------------------------ |
| 构思大纲 | 0.55 | 分析主题，规划情节点和情感弧线 |
| 初稿写作 | 0.65 | 按风格指南撰写初稿             |
| 修订润色 | 0.35 | 检查风格一致性，强化意象表达   |
| 伏笔检查 | 0.35 | 校验伏笔收束，必要时补写桥段   |

```bash
uv run story continue novel.txt                      # 自动续写
uv run story continue novel.txt --hint "希望是HE"    # 带提示续写
uv run story continue novel.txt --hint "男主和女二在一起"
```

### `cache` - 缓存管理

```bash
uv run story cache status  # 查看缓存状态
uv run story cache clear   # 清空所有缓存
uv run story cache status --config config.yaml  # 查看配置指定的缓存目录
```

## 配置

### 环境变量

API Key 通过 `config.yaml` 中的 `api_key_env` 指定环境变量名，系统自动读取：

```bash
# DeepSeek（默认）— config.yaml 已配置 api_key_env: DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY="sk-xxx"

# 切换到 OpenAI — 修改 config.yaml 中 model/base_url/api_key_env 后：
export OPENAI_API_KEY="sk-xxx"

# 切换到本地 Ollama（无需 API Key）
export STORY_LLM__PROVIDER="ollama"
export STORY_LLM__MODEL="qwen3.5:9b"
export STORY_LLM__BASE_URL="http://localhost:11434"
```

所有配置项都可通过 `STORY_<SECTION>__<KEY>` 环境变量覆盖 `config.yaml` 中的值，例如
`STORY_LLM__TIMEOUT=600` 或 `STORY_CACHE__DIRECTORY=.story_cache_alt`。配置文件和环境变量加载后会重新校验，
非法的 chunk、overlap、temperature、context window 等值会在命令启动时直接报错。

### 配置文件

创建 `config.yaml`：

```yaml
llm:
  # DeepSeek（默认配置）
  provider: openai
  model: deepseek-v4-pro
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY   # 从此环境变量读取 API Key
  temperature: 0.7
  context_window: 1000000         # deepseek-v4-pro 上下文窗口
  max_tokens: 32768

  # 切换到 OpenAI 只需改这三行：
  # model: gpt-4o
  # base_url: https://api.openai.com
  # api_key_env: OPENAI_API_KEY

parser:
  chunk_size: 120000              # 长上下文模型下减少请求次数
  overlap: 2000

prompt:
  margin: 20000                   # 给指令、历史摘要和结构化输出预留空间

analysis:
  target_unit_chars: 8000          # 语义分析单元大小，不等于模型上下文窗口
  min_units: 8
  max_units_per_batch: 8
  batch_context_chars: 120000

cache:
  enabled: true
  directory: .story_cache
  auto_save: true
```

使用配置文件：

```bash
uv run story analyze novel.txt --config config.yaml
```

默认 provider 使用 DeepSeek 的 OpenAI-compatible Chat API。结构化抽取场景会自动把内部
`no_think` 选项转换为 DeepSeek 支持的 `thinking: {"type": "disabled"}`，以减少推理内容对
JSON 输出的干扰；其他 OpenAI 兼容服务不会收到这个 DeepSeek 专用参数。

CLI 会为 LLM 请求输出可见进度：命令开始时列出基线请求计划，每次请求会显示当前阶段、
prompt 摘要、attempt/retry、输入/输出 token、缓存命中 token、剩余请求估算和累计费用。
DeepSeek 官方模型会按内置价目估算费用；其他 OpenAI-compatible 服务若未配置价目则显示
`cost=n/a`，但仍会展示 token 用量。

为提高 DeepSeek KV Cache 命中率，分析阶段会把同一章节正文拆成稳定的 `cache-prefix`
前置消息，人物、关系、摘要、事件等任务说明放在后续 user 消息中。后续 Prompt 设计也应遵循
“长正文/稳定上下文在前，短任务/可变参数在后”的顺序，避免把任务说明放在正文前面破坏公共前缀。
续写阶段也会把 `StoryContext` 渲染出的固定上下文作为前缀复用，让 hint 解析、大纲、初稿、润色、
伏笔审查和最终审稿共享同一段上下文前缀。

`analyze` 的请求计划是上限估算：常规章节最多为“人物、关系、摘要、事件、故事状态”5 次请求，
再加最终风格提取；但只有 0-1 个角色的章节会跳过关系抽取，且没有新事件时会复用上一帧故事状态，
不再额外请求状态合并。schema repair、retry、final repair 仍会作为额外请求记录。

示例输出：

```text
LLM plan for analyze: baseline ~41 request(s). Repairs/retries are logged as extra requests.
  - for each of 8 chapter chunk(s): characters, relationships if useful, summary, events, story state if changed
  - final writing style extraction
[LLM 1/41] analyze chapter: extract characters
[LLM 1/41] prompt: cache-prefix: 第一章正文... / task: # Character Sheet Prompt...
[LLM 1/41] done in 8.2s | in=6420 out=810 total=7230 cache=2048 cache_rate=31.9% cost=$0.002613 cumulative=$0.002613
[LLM 1/41] remaining~40
```

字段说明：

- `in` / `out` / `total`：provider 返回的输入、输出和总 token。
- `cache` / `cache_rate`：provider 返回的缓存命中输入 token 和命中率，用来判断 prompt 是否缓存友好。
- `cost` / `cumulative`：基于当前内置价目表的近似估算，最终费用以服务商账单为准。
- `remaining~N`：基于基线流程估算；schema repair、retry、final repair 会作为额外请求继续记录。

`max_tokens` 控制单次生成的最大输出，不代表可用上下文长度；长篇分析的分块预算由
`context_window - prompt.margin`、`parser.chunk_size` 和 `analysis.target_unit_chars` 共同决定。
默认配置面向 1M 上下文的 `deepseek-v4-pro`，但语义分析仍按约 8000 字符的分析单元执行，
避免把整本书压成单个“章节”导致事件覆盖丢失。

语义任务不再提供 Python 兜底：人物、关系、事件、故事状态、读者指令、结局审查、删除/桥接文本都由 LLM 处理。Python 仅负责分块、缓存、索引、去重、JSON/schema 校验和流程编排；LLM 失败、空响应或结构错误会直接报错。

## 技术架构

```
CLI Layer (Typer)
    story analyze | compress | filter | remove | continue | cache
         │
Core Business Layer
    Compressor, CharacterFilter, CharacterRemover, EndingWriter (多阶段)
         │
Analysis Layer (三层记忆)
    ├─ 短期：ChapterSummaryWindow（最近 N 章摘要）
    ├─ 中期：EventLedger（不可逆剧情事件）
    ├─ 长期：StateStore（人物/世界状态快照）
    └─ 风格：StyleExtractor → WritingStyle
    → 汇聚为 StoryContext
         │
Indexer Layer
    SegmentIndexService, SegmentRetriever, CharacterTagger
         │
LLM Abstraction Layer
    LLMProvider (抽象) → OpenAICompatibleProvider (默认, DeepSeek API) / OllamaProvider (本地备选)
         │
Infrastructure
    TextSplitter, Config (Pydantic + YAML), ContextStore (缓存), ProgressStore (断点续传)
```

### 分析管道

每章按以下顺序处理：

```text
CharacterExtractor → RelationshipMapper → ChapterSummarizer → EventExtractor → StateSynthesizer
```

结果汇聚到三层记忆系统，最终组装为 `StoryContext`。

### 共享索引

`SegmentIndexService` 在分析阶段构建段落/角色索引。四个核心命令（compress/filter/remove/continue）复用同一份缓存的 `SegmentIndex`，确保处理一致性。

**关键设计**：
- **LLM 语义优先**：所有人物/剧情/关系/续写约束判断都交给 LLM，Python 不用关键词或正则猜测剧情含义
- **断点续传**：长篇分析支持中断后继续，进度保存在 `.story_cache/`
- **风格一致性**：自动提取原作风格，注入所有处理环节

## 开发

```bash
# 安装开发依赖
uv sync --dev

# 代码检查
uv run ruff check .

# 运行测试
uv run pytest -q
```

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).
