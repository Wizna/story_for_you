# Continue Failure Analysis and Remediation Plan

## 背景

用 `/Users/wizna/Downloads/边城.txt` 测试 `story continue` 时，链路可以完成分析、缓存和续写，但生成结果暴露出框架级问题：

- 第一版仍是开放式结局，出现“也许”“明天回来”“船影未近”等表达。
- 第二版更接近非开放式，但内部自相矛盾：先写傩送遗体被打捞，后又写傩送来信说不回来了。

结论：问题不应通过反复重试或单次 prompt 加强解决。需要修复分析粒度、上下文选择、用户意图约束和结局一致性审查。

## 已确认问题

### P0: “非开放式”被解析成开放式

`HintInterpreter` 的 `OE` 关键词包含“开放”。用户输入“非开放式结局”时会匹配到“开放”，从而把结局方向设为 `OE`。这是确定性 bug。

影响：
- 大纲阶段被错误引导为开放式。
- 后续 draft/polish/resolution 阶段都携带错误方向。

### P0: 大上下文配置与分析粒度耦合错误

`context_window=1000000` 后，`边城.txt` 被切成 1 个 chunk。分析器仍把 chunk 当“单章”处理，但现有抽取 prompt 是小章节级设计：

- chapter summary 只要求 3-6 个 beats。
- character sheet 最多 8 人。
- event extraction 没有全文覆盖要求或分段事件覆盖要求。

结果：DeepSeek 缓存只抽出 3 个开篇事件，`StoryState` 停在开篇，未覆盖天保死亡、老船夫死亡、傩送离开、翠翠等待等终局事实。

### P0: 续写“最近片段”在单 segment 下取到了全文开头

`EndingWriter._recent_segment_digest()` 对 segment 只取 `content[:SNIPPET_EXCERPT_LEN]`。当全文只有 1 个 segment 时，所谓“最近片段”变成版权信息/题记/开头，而不是小说结尾。

影响：
- 续写不承接原作末尾。
- 输出开头模仿题记腔。

### P0: 风格提取采样在单 chunk 下只取前 2000 字

`StyleExtractor` 在 `chapters == 1` 时只采样第 1 个 chunk 的前 2000 字。`边城.txt` 前 2000 字包含版权信息和题记，导致风格被提取成议论性前言风格，而不是正文叙事风格。

影响：
- characteristic words 变成“作品/读者/民族/生活”等前言词。
- style samples 来自题记。
- 续写开头出现“就我所接触的世界一面”等元叙述。

### P0: 结局审查不检查用户约束或事实矛盾

`ending_resolution.txt` 只检查“伏笔是否回应或有意留白”，并允许 `status: ok`。它不检查：

- 是否满足用户硬性要求。
- 是否仍是开放式。
- 人物生死/去向是否自洽。
- 同一人物是否同时死亡又写信/归来。

### P1: 人物别名合并污染

DeepSeek 缓存中 `傩送` 的 aliases 包含 `大老`、`天保`。这会混淆兄弟角色，降低人物状态可信度。

### P1: 失败路径不应生成伪结果

`EndingWriter` 原先在多阶段失败后会走简化草稿甚至本地 heuristic 结局，且曾默认偏向 `OE`。这类失败路径会掩盖真实错误，必须删除；LLM 不可用或返回坏结构时应直接报错。

### P1: 本地语义硬匹配会误判任务

`HintInterpreter`、`EndingValidator`、事件/人物/关系抽取如果靠 Python 关键词、黑名单、计数或正则判断语义，会把“非开放式”这类否定表达误判为相反含义，也会制造不存在的剧情状态。语义解释必须交给 LLM，Python 只校验 JSON/schema 和执行流程。

## 总体设计目标

1. **模型上下文窗口与分析粒度解耦**：大上下文模型可以减少网络开销，但不能牺牲事件覆盖和时间线结构。
2. **分析粒度自适配**：不同模型、不同上下文窗口、不同文本长度，应自动选择合适的“场景/章节/批次/全局”多层分析方案。
3. **续写前有可信状态**：继续故事必须依赖结尾附近片段、全局人物状态、核心冲突和未解决伏笔，而不是只依赖风格和用户提示。
4. **用户约束可验证**：将 hint 交给 LLM 转成结构化硬约束，并在最终输出后由 LLM reviewer 审查，失败则报错或进入显式修复流程。
5. **Python 不做语义兜底**：允许的本地逻辑仅包括分块、排序、缓存、JSON/schema 校验、去重等机械处理；人物关系、剧情矛盾、结局闭合、文风质量等语义判断由 LLM 完成。

## 分析粒度方案

### 1. 增加 `AnalysisGranularitySettings`

建议新增配置：

```yaml
analysis:
  target_unit_chars: 6000
  min_units: 8
  max_units_per_batch: 8
  batch_context_chars: 120000
  preserve_chapter_boundaries: true
```

含义：

- `target_unit_chars`：语义分析单元大小。默认 4k-8k 字符，适合人物、事件、摘要抽取。
- `min_units`：即使模型有 1M 上下文，也至少切出若干分析单元，避免全文压成一个“章节”。
- `max_units_per_batch`：大模型可以一次请求处理多个 unit，但输出必须按 unit 分组。
- `batch_context_chars`：单次 LLM 请求的最大正文预算，不能等同于 `context_window`。

### 2. 分层处理：Unit -> Batch -> Global

#### Unit 层

将原文切成稳定的分析单元：

- 优先按真实章节标题切。
- 没有章节标题时按场景/段落边界切到 `target_unit_chars`。
- 保留 `unit_id`、`source_span`、`chapter_label`、`is_front_matter`。

每个 unit 输出：

- `ChapterSummary`
- `PlotEvent[]`
- `CharacterState[]`
- `Relationship[]`

#### Batch 层

大上下文模型可把多个 unit 放进一个 batch 请求，但 prompt 必须要求：

- 输出对象以 `unit_id` 分组。
- 每个 unit 至少产出 summary。
- 事件抽取必须覆盖 batch 的前/中/后位置。

这样能减少请求次数，同时保留分析粒度。

#### Global 层

所有 unit 分析完成后，再做全局聚合：

- 合并人物和别名。
- 生成全局 timeline。
- 归纳主线冲突和未解决伏笔。
- 提取最终状态快照。
- 生成 `ending_preconditions`，专供续写使用。

### 3. 模型自适配策略

根据 `context_window` 和文本长度选择执行计划：

| 模型上下文 | Unit 大小 | Batch 策略 | 说明 |
| --- | --- | --- | --- |
| <= 8k | 2k-4k | 1 unit/request | 小模型稳定优先 |
| 8k-64k | 4k-8k | 2-4 units/request | 中等模型减少请求 |
| 64k-256k | 6k-12k | 4-8 units/request | 长篇默认 |
| >= 256k | 8k-16k | 8-16 units/request + global pass | 大上下文用于批处理和全局聚合，不直接吞全文 |

## 续写方案

### 1. 新增 `EndingDirective`

从用户 hint 解析结构化约束：

```python
@dataclass
class EndingDirective:
    closure: Literal["closed", "open", "unspecified"]
    ending_direction: Literal["HE", "BE", "OE"] | None
    required_outcomes: list[str]
    forbidden_outcomes: list[str]
    required_characters: list[str]
    forbidden_phrases: list[str]
```

规则：

- “非开放式/不开放/不要留白/不留悬念”必须解析为 `closure="closed"`。
- “开放式/留白/OE”才解析为 `closure="open"`。
- 否定词优先级高于关键词匹配。

### 2. 新增 `EndingPreconditions`

从 `StoryContext` 和尾部原文生成续写前置状态：

```python
@dataclass
class EndingPreconditions:
    last_scene: str
    unresolved_threads: list[str]
    character_statuses: dict[str, CharacterFinalState]
    must_resolve: list[str]
    timeline_facts: list[str]
```

来源：

- 最后 2-3 个 analysis units 的摘要和原文片段。
- 全局 timeline 的最后若干不可逆事件。
- 用户关注人物的状态快照。

### 3. 修改最近片段选择

不能用 `segment.content[:280]`。应支持：

- `tail_excerpt(segment, chars=800)` 用于续写承接。
- 若按人物找 segment，取该人物最后出现位置附近窗口，而不是 segment 开头。
- 始终注入原文最后 `tail_chars`，作为“必须承接”的文本锚点。

### 4. 输出后校验

新增 `EndingValidator`，但它只负责调用 LLM reviewer 并校验返回 schema：

- 检查 structured directives 是否被满足。
- 检查 required_outcomes / forbidden_outcomes / required_resolutions。
- 检查主要人物最终状态是否自洽。
- 检查是否存在人物生死、关系、时间线或结局闭合要求的矛盾。

Python 不做短语黑名单、正则矛盾检测或开放式关键词判断。

## 修复顺序

### Phase 1: 立即修确定性 bug（已开始）

1. [x] 将 `HintInterpreter` 改为 LLM 指令抽取器，不再做本地字符串匹配。
2. [x] 修最近片段取尾部，不再取 segment 开头。
3. [x] 修 `StyleExtractor` 单 chunk 采样：用结构位置取中部样本，避免固定取 segment 开头。
4. [x] 删除 `EndingWriter` 失败兜底和本地 heuristic 结局。
5. [x] 将 `EndingValidator` 改为 LLM reviewer，不再用本地短语/正则检查开放式表达和人物矛盾。

### Phase 2: 分析粒度重构（已完成第一步）

0. [x] 新增 `analysis.target_unit_chars/min_units/max_units_per_batch/batch_context_chars` 配置，并让分析阶段使用 `target_unit_chars`，不再直接按 `parser.chunk_size` 或 `context_window` 吞全文。
1. 新增 `AnalysisUnit` 和 `AnalysisPlanner`。
2. `TextSplitter` 支持真实章节识别和 front matter 标记。
3. 分析 pipeline 从 `chapters: list[str]` 改为 `units: list[AnalysisUnit]`。
4. 大上下文走 batch，但输出按 unit 分组。
5. `StoryContext.metadata` 记录 analysis plan、unit_count、batch_count。

### Phase 3: 全局聚合

1. 新增 `GlobalStorySynthesizer`。
2. 聚合 timeline、人物最终状态、主线冲突、未解决伏笔。
3. 为 continue 生成 `EndingPreconditions`。

### Phase 4: 质量门禁

1. 续写 pipeline 增加 validator -> repair -> validator。
2. 失败时输出明确错误，而不是保存不合格结果。
3. 对 `边城.txt` 增加离线 fixture 或小样本回归测试，覆盖非开放式约束。

## 验收标准

以 `边城.txt` 为用例：

1. 分析结果至少覆盖故事开端、中段、终局三类事件。
2. `StoryContext.events` 包含天保死亡、老船夫死亡、傩送离开、翠翠留守渡口等关键事件。
3. 写作风格样本来自正文叙事段落，不来自版权信息/题记。
4. `continue --hint 非开放式结局` 输出不得包含开放式表达。
5. 输出中主要人物最终状态不自相矛盾。
6. 若校验失败，CLI 不应静默保存为成功结果。
