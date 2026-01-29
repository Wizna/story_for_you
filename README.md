# Story For You

基于 LLM 的中文网络小说处理工具，解决长篇小说阅读中的常见痛点：内容冗长、角色繁杂、结局不满意。

## 功能特性

- **剧情压缩** - 智能提取核心剧情，去除水分内容，保留 90%+ 原文风格
- **角色筛选** - 只保留指定角色相关的剧情线，快速追踪 CP 或主角故事
- **角色删除** - 从文本中移除不喜欢的角色，自动修补叙事连贯性
- **结局续写** - 五阶段创作流程（灵感→大纲→初稿→修订→润色），生成符合原作风格的新结局

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
- [Ollama](https://ollama.ai/) 运行于 `localhost:11434`
- 推荐模型：`qwen2.5:7b-instruct`

```bash
# 安装并启动 Ollama 后，拉取模型
ollama pull qwen2.5:7b-instruct
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

```bash
uv run story compress novel.txt --level light   # 轻度压缩
uv run story compress novel.txt --level medium  # 中度压缩（默认）
uv run story compress novel.txt --level heavy   # 重度压缩
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

基于原作风格续写新结局。

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

```bash
export STORY_LLM__MODEL="qwen2.5:7b-instruct"
export STORY_LLM__BASE_URL="http://localhost:11434"
export STORY_LLM__TEMPERATURE="0.7"
```

### 配置文件

创建 `config.yaml`：

```yaml
llm:
  model: qwen2.5:7b-instruct
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 8192

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

## 技术架构

```
CLI Layer (Typer)
    story analyze | compress | filter | remove | continue | cache
         │
Core Business Layer
    Compressor, CharacterFilter, CharacterRemover, EndingWriter
         │
Analysis Layer (三层记忆)
    ├─ 短期：ChapterSummaryWindow（最近 N 章摘要）
    ├─ 中期：EventLedger（不可逆剧情事件）
    ├─ 长期：StateStore（人物/世界状态快照）
    └─ 风格：StyleExtractor → WritingStyle
         │
LLM Abstraction Layer
    LLMProvider → OllamaProvider
```

**关键设计**：
- **检索优先策略**：最大化保留原文（90%+），LLM 仅用于生成桥接文本
- **断点续传**：长篇分析支持中断后继续，进度保存在 `.story_cache/`
- **风格一致性**：自动提取原作风格，注入所有处理环节

## 开发

```bash
# 安装开发依赖
uv sync --dev

# 代码格式化
uv run black story_for_you tests

# 代码检查
uv run ruff check .

# 运行测试
uv run pytest -q
```

## License

MIT
