# 续写质量优化与评测

## 当前实现

1. `PlotEvent.deceased_characters` 标明死亡主体，死亡关键词不再传播到同事件的其他参与者。
2. 人物与世界伏笔按稳定 ID 维护。新增、更新、解决必须有证据，未提及不等于解决；旧数据中的
   `unresolved` 字符串会迁移成稳定记录。分析缓存格式版本为 5，自动重建旧格式缓存。
3. 每单元通过承接审查后才提交实际状态变化。滚动记忆保留所有历史摘要、关键事件与引文，
   人物的地点、目标、关系、知情范围和未决事项供后续单元使用；原作 `StoryContext` 保持独立。
4. 上下文优先本单元重点人物，补齐其历史事件与最近出现位置的原文。重点人物状态不会因姓名排序
   排在第七位而缺席，全部不可逆事实保留；世界未决线索独立呈现。
5. 每次调用以 UTF-8 字节数保守估计输入 token，并预留 `llm.max_tokens` 输出空间和消息开销。
   超出 `llm.context_window` 时明确失败，不悄悄删除证据。OpenAI-compatible 的 `finish_reason=length`
   及 Ollama 的 `done_reason=length` 会作为错误处理；不会把截断文本保存为成功结果。
6. 段长、对白比例、叙事距离、解释密度和修辞强度服从原作与当前场景，没有统一 120 字段落、
   两个锚点词或诗意结尾要求。
7. 每单元写作前细化场景目标、阻力、行动、转折、代价、结果与知情边界；生成阶段能看到跨单元安排，
   同时明确区分未来计划和已发生事实。
8. 伏笔审查和最终修复返回唯一原文引用的局部替换。引用失配、歧义或重叠时不应用修改。
   修改后再校验，章节标题必须完整且顺序一致。未被修改的正文不经整篇重写。
9. `story eval` 提供截断点隔离、独立生成、盲评打包与评分汇总。

正常无修复时，实际调用数为 `3 + 4N + R`，其中 N 是模型选择的单元数，R 为是否执行伏笔审查（0/1）。
相较旧流程，每单元新增场景规划、承接审查与记忆提取两次调用。CLI 展示的是按 max_units 计算的估计，
局部修复及重审会增加调用。当前实现使用同步生成；遇到失败会报错，尚无续写过程的持久化断点恢复。

## 准备截断输入

创建 `cases.json`，不同小说、不同叙事阶段分别设置截断点。`cutoff_chars` 以解码后的 Python 字符数计算，
必须大于 0 且小于全文长度。源文件相对路径按 manifest 所在目录解析。

```json
[
  {
    "id": "family-mid",
    "source": "/path/to/family.txt",
    "cutoff_chars": 12000,
    "hint": "承接当前场景，继续展开人物关系，不提前收束",
    "max_units": 3
  },
  {
    "id": "mystery-late",
    "source": "/path/to/mystery.txt",
    "cutoff_chars": 18000,
    "hint": "继续调查，保留角色尚未获知的信息差",
    "max_units": 3
  }
]
```

```bash
uv run story eval prepare cases.json --output-dir outputs/eval-inputs
```

此步骤不调用模型，不修改原文件。输出仅包含截断前正文、hint、单元上限与正文哈希，
不复制被排除的后文，也不把源文件路径带入准备后的 manifest。所有输出目录必须是新目录，避免混入旧结果。

## 生成两个待比较版本

```bash
uv run story eval generate outputs/eval-inputs/cases.json --config baseline.yaml --output-dir outputs/eval-baseline
uv run story eval generate outputs/eval-inputs/cases.json --config candidate.yaml --output-dir outputs/eval-candidate
```

`generate` 会实际调用模型，每个 case 从截断前正文重新分析，不读取 `.story_cache`、外部 context 或 segments。
相同准备输入可以供两个模型配置或代码版本使用。也可以通过 `evaluation.continuation.generate_cases`
注入旧版本的生成函数；回调只能接收 prefix、hint 和 max_units，应只分析该 prefix。
两个版本必须使用相同的输入哈希、hint 与单元上限。结果记录模型名与输入/输出哈希，不记录凭证。

生成阶段只提供前缀，不保证模型训练数据中从未见过原作。评测应包含不同小说和多个截断点，
避免仅凭熟悉原作的某个样本判断泛化效果。

## 盲评与汇总

```bash
uv run story eval blind outputs/eval-inputs/cases.json outputs/eval-baseline/results.json outputs/eval-candidate/results.json --output-dir outputs/eval-comparison
```

对每个 case 随机分配 A/B。只把 `outputs/eval-comparison/blind/` 交给评审，其中包含前缀、读者要求、
A/B 正文、说明和 `ratings.json`。`private_key.json` 留在 blind 目录之外；评审完成前不揭示映射。
`--seed` 可重现分配，种子也不向评审披露。

填写每个 case 的 `winner`（A/B/tie），以及 A、B 的以下字段：

| 字段 | 评分含义 |
| --- | --- |
| contradictions | 确认的事实矛盾数量，非负整数；存在矛盾时必须提供引文证据 |
| motivation | 人物选择与既有动机是否连贯，1–5 分 |
| knowledge_boundary | 视角和人物知情范围是否正确，1–5 分 |
| plot_progress | 是否通过行动推进情节，1–5 分 |
| style | 句式、对白、叙事姿态与原作是否一致，1–5 分 |
| evidence | 原文/续写引文及问题说明组成的字符串数组 |

```bash
uv run story eval report outputs/eval-comparison/blind/ratings.json outputs/eval-comparison/private_key.json -o outputs/eval-report.json
```

报告包含胜出数量、平局数、各维度均分、事实矛盾数及每万字符矛盾数。未填写的评分、缺失 case、
重复 ID、无证据的矛盾计数都不会被当作有效评分汇总。输出是描述性结果，不自动宣称统计显著。

## 已验证与待验证

离线回归覆盖死亡主体、人物与世界伏笔关闭、跨 8000 字尾部的历史保留、重点人物召回、输出截断、
局部修稿、章节完整性，以及多小说前缀隔离、CLI 生成入口、盲评数据一致性和评分汇总。
这些检查验证流程行为；真实小说续写的质量、盲评胜率和成本收益需要通过上述流程单独测量。
