# Story For You

基于 LLM 的中文网络小说处理工具，解决长篇小说阅读中的常见痛点：内容冗长、角色繁杂、结局不满意。

## 功能特性

- **剧情压缩** - 智能提取核心剧情，去除水分内容，保留 90%+ 原文风格
- **角色筛选** - 只保留指定角色相关的剧情线，快速追踪 CP 或主角故事
- **角色删除** - 从文本中移除不喜欢的角色，自动修补叙事连贯性
- **自适应续写** - 先由模型判断叙事阶段、范围和收束程度，再按单元生成符合原作风格的承接、展开或结局

## 设计准则

- **LLM 负责语义**：人物、事件、关系、故事状态、读者 hint、续写范围、叙事阶段、承接审查、删除/桥接文本都交给 LLM 判断。
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

# 2. 需要单独更新风格分析时（可选）
uv run story style novel.txt -o novel_style.json

# 3. 使用分析结果进行处理
uv run story compress novel.txt --level medium      # 压缩剧情
uv run story filter novel.txt -c "张三,李四"        # 筛选角色
uv run story remove novel.txt -c "王五"             # 删除角色
uv run story continue novel.txt --hint "续写后四十回，保持原作余韵"   # 模型判断范围与叙事阶段
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

### `style` - 独立风格分析

从原文首、中、尾样本提取写作风格，不需要重新执行完整故事分析。默认输出到
`{input}_style.json`；使用 `--inject` 时会把结果写入已有的 context JSON。

```bash
uv run story style novel.txt -o novel_style.json
uv run story style novel.txt --context analysis.json --inject
```

`--inject` 必须同时提供 `--context`，且会原地更新该 JSON 文件。

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

### `continue` - 自适应续写

先由模型判断本次是承接场景、展开阶段、转折、高潮、余波还是自然终止，再决定需要多少个叙事单元。单元可以对应一回、一个场景、一个大章节或多个回目的结构段；调用 `continue` 不会强迫故事进入结局。续写上下文同时包含原文尾部、索引场景尾部、不可逆事实、人物位置/目标/知情状态和关系状态。长篇分析会限制每次抽取携带的已知人物 roster（默认 `analysis.max_known_characters: 96`），避免早期配角挤掉当前场景人物。人物关系保存按单元标注的历史转折，人物和世界伏笔保存证据轨迹；提示词明确区分当前状态、历史原文和人物实际知情范围。末章出现反常对白、药物/死亡、宗教或动物意象等候选深意时，会先做一次带逐字证据的伏笔与隐喻预分析，同时用末章动作和词面生成高显著物件/文书检查清单，再从全书索引回溯同一物件、称谓、意象或关系的早期片段，把高显著物件、双关解释、候选对应人物和可执行情节建议交给结构规划；低置信度解释只作为假设，不会直接写成事实。

| 阶段 | 说明 |
| --- | --- |
| 末章伏笔与隐喻预分析（条件触发） | 从反常对白、重复意象和异常反应中提取逐字证据，列出候选解释与后果 |
| 续写范围与结构规划 | 判断叙事阶段、时间/回目范围、单元数量和本轮收束程度 |
| 单元场景规划 | 明确目标、阻力、行动、转折、代价与人物知情边界，保留后续单元职责 |
| 单元初稿与润色 | 段落、对白和修辞服从原作，不设统一段长与特征词配额 |
| 单元审查与记忆更新 | 审查通过后记录实际状态变化与原文证据，供后续单元承接 |
| 跨单元承接检查 | 检查关系、人物状态、不可逆事实、视角和信息差是否前后一致 |
| 最终审查 | 定位替换问题片段，保留其他正文，修改后再次审查 |

```bash
uv run story continue novel.txt                              # 自动判断续写范围与单元数
uv run story continue novel.txt --hint "续写后四十回"         # 明确指定回目范围
uv run story continue novel.txt --hint "续写后四十回" --max-units 40  # 允许回目级计划覆盖 40 回
uv run story continue novel.txt --hint "补足家族衰败和人物归宿，保留悲剧余韵"
uv run story continue novel.txt --max-units 6                  # 单次运行的安全上限（--max-chapters 是兼容别名）
```

人物死亡只依据逐字死亡证据或事件中的明确死亡主体；死亡状态不会被后续章节的含糊抽取复活。人物与世界伏笔按稳定 ID 更新，保留章节/物理单元证据历史，只有明确解决证据才能关闭。
重点人物的完整知情状态、相关历史事件与原文位置会补入上下文，不可逆事实不会按最近 20 条截断。
物品、地点、势力和场景线索也参与历史原文召回；最终审查涵盖全部计划人物及各单元开始时的状态。
规划阶段会为奏章、书信、药瓶、玉玺、火枪和象征图像等高显著锚点指定 `advance`、`hold` 或 `background` 处理，并要求每个锚点进入至少一个单元的具体动作。模型可同时列出 1–3 个基于证据的情节方向并标记一个优先项；“钳龙”对应谁、药是否有毒等未证实解释会保留候选分支。模型误写旧伏笔 ID 时，记忆审查只把有原文证据的未知更新降级为新的开放线索，不会静默关闭旧线索。
明确完成的目标、离开的地点和被纠正的知识可通过带原文证据的状态操作更新，并保留变更历史。
分析缓存格式为 7，旧分析会自动失效。模型输出被截断或调用预算不足时会报错，不保存半截结果。

默认 `ending.revision_mode: full` 执行整单元润色；`local` 只替换模型明确引用的片段；`none` 跳过润色，
三种模式都保留承接审查。对白、复沓与重复段落不会被程序自动删除。

续写支持阶段断点及下一轮状态导出。开启缓存时自动保存断点，也可明确指定：

```bash
uv run story continue novel.txt --hint "继续当前场景" --checkpoint .story_cache/scene.json
# 保持原文、hint、配置、单元上限和代码一致，恢复已完成阶段
uv run story continue novel.txt --hint "继续当前场景" --checkpoint .story_cache/scene.json --resume
# 成功后，使用导出的合并正文和状态继续下一轮
uv run story continue .story_cache/scene.context.source.txt --context .story_cache/scene.context.json --hint "推进下一场"
```

`--state-output PATH` 可指定状态输出位置。成稿若经过最终修订，会先重建与成稿一致的状态；
下轮输入会核对正文哈希，避免将旧状态用到不同的稿件。断点和生成材料不应提交到 Git。

### `eval` - 续写盲评

支持截断前正文隔离、冻结分析、润色消融、匿名 A/B 打包，以及人工评分汇总。准备与评分不调用模型，
`eval analyze/generate` 会调用已配置的模型。完整示例与指标解释见 [续写质量与评测](docs/continuation_quality.md)。

```bash
uv run story eval prepare cases.json --repetitions 3 --output-dir outputs/eval-inputs
uv run story eval analyze outputs/eval-inputs/cases.json --output-dir outputs/eval-analysis
uv run story eval generate outputs/eval-inputs/cases.json --analysis outputs/eval-analysis/analysis.json --revision-mode full --output-dir outputs/eval-baseline
uv run story eval generate outputs/eval-inputs/cases.json --analysis outputs/eval-analysis/analysis.json --revision-mode local --output-dir outputs/eval-candidate
uv run story eval blind outputs/eval-inputs/cases.json outputs/eval-baseline/results.json outputs/eval-candidate/results.json --output-dir outputs/eval-comparison
# 人工填写 blind/ratings.json 后汇总
uv run story eval report outputs/eval-comparison/blind/ratings.json outputs/eval-comparison/private_key.json -o outputs/eval-report.json
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
  model: deepseek-flash
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY   # 从此环境变量读取 API Key
  temperature: 0.7
  context_window: 1000000         # 本地上下文预算，需与服务实际限制一致
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
  max_known_characters: 96          # 每个分析单元送入人物/关系/事件抽取的 roster 上限

ending:
  max_chapters: 40                # 计划单元的配置上限；CLI --max-units 默认 6
  revision_mode: full             # full / local / none

rendering:
  max_history_events: 24          # 按实体、人物与场景相关性召回
  max_evidence_passages: 8
  evidence_chars: 12000           # 检索证据与滚动摘要各自的字符预算；必需状态另行保留
  max_relationship_history: 4      # 重点人物关系转折在提示词中的最近条数
  max_thread_history: 4            # 重点人物/世界伏笔证据在提示词中的最近条数
  max_foreshadowing_passages: 10  # 末章信号回溯全书时的片段数
  foreshadowing_chars: 16000      # 全书伏笔回溯证据字符预算

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

请求数速查：

| 命令 | 已有分析缓存时 | 需要现场分析时 | 省请求 / 省 token 策略 |
| --- | --- | --- | --- |
| `analyze` | 不适用 | 最多 `chapter_chunks * 5 + 1` | 单角色跳过关系抽取；无新事件复用上一帧状态；同章正文作为 cache prefix |
| `compress` | 按输入与输出预算分批改写 | 先跑 `analyze`，再分批改写 | 复用缓存的 `StoryContext + SegmentIndex`，按原文顺序处理选中片段 |
| `filter` | 每个断点 1 次桥接请求 | 先跑 `analyze`，再按断点桥接 | 原文段落直接拼接，仅断点调用 LLM |
| `remove` | 每个受影响段落最多 1 次处理请求 | 先跑 `analyze`，再处理受影响段落 | 未受影响段落原样保留 |
| `continue` | 默认模式 `3 + 4N + R + S`；`none` 为 `3 + 3N + R + S` | 先跑 `analyze`，再续写 | N 为实际单元数；R 为伏笔审查 0/1；S 为候选深意预分析 0/1（检测到候选片段时）；修复、重审和最终修订后的状态重建另计；可恢复已完成阶段 |

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
默认模型为 `deepseek-flash`，本地上下文预算配置为 1M，但语义分析仍按约 8000 字符的分析单元执行，
避免把整本书压成单个“章节”导致事件覆盖丢失。

语义任务不再提供 Python 兜底：人物、关系、事件、故事状态、读者指令、结局审查、删除/桥接文本都由 LLM 处理。Python 仅负责分块、缓存、索引、去重、JSON/schema 校验和流程编排；LLM 失败、空响应或结构错误会直接报错。

## 技术架构

```
CLI Layer (Typer)
    story analyze | style | compress | filter | remove | continue | cache
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
- **断点续传**：长篇分析与续写均支持中断后继续，进度保存在 `.story_cache/`；续写可导出下一轮正文与状态
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
