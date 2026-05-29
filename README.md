# Story For You

基于 LLM 的中文网络小说处理工具，解决长篇小说阅读中的常见痛点：内容冗长、角色繁杂、结局不满意。

## 功能特性

- **剧情压缩** - 智能提取核心剧情，去除水分内容，保留 90%+ 原文风格
- **角色筛选** - 只保留指定角色相关的剧情线，快速追踪 CP 或主角故事
- **角色删除** - 从文本中移除不喜欢的角色，自动修补叙事连贯性
- **结局续写** - 四阶段创作流程（构思大纲→初稿→修订润色→伏笔检查），生成符合原作风格的新结局

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/story_for_you.git
cd story_for_you

# 安装依赖（需要 Python 3.11+）
uv sync

# 验证安装
uv run story --help
```

### 前置要求

- Python 3.11+
- LLM 后端（二选一）：
  - **云端**：DeepSeek API（默认，需要 API Key）
  - **本地**：[Ollama](https://ollama.ai/) 运行于 `localhost:11434`

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

基于原作风格续写新结局，采用四阶段创作流程：

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

其他 LLM 参数也可通过 `STORY_LLM__*` 环境变量覆盖 `config.yaml` 中的值。

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
  max_tokens: 32768

  # 切换到 OpenAI 只需改这三行：
  # model: gpt-4o
  # base_url: https://api.openai.com
  # api_key_env: OPENAI_API_KEY

parser:
  chunk_size: 4000
  overlap: 200

cache:
  enabled: true
  directory: .story_cache
```

使用配置文件：

```bash
uv run story analyze novel.txt --config config.yaml
```

默认 provider 使用 DeepSeek 的 OpenAI-compatible Chat API。结构化抽取场景会自动把内部
`no_think` 选项转换为 DeepSeek 支持的 `thinking: {"type": "disabled"}`，以减少推理内容对
JSON 输出的干扰；其他 OpenAI 兼容服务不会收到这个 DeepSeek 专用参数。

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
- **检索优先策略**：最大化保留原文（90%+），LLM 仅用于生成桥接文本
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

MIT
