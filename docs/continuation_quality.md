# 续写质量优化与评测

## 当前实现

1. 事件按物理分析单元、源章节和 `event_id` 区分发生次数，不用相同摘要合并不同时间的行为。
   物品转交、归还、再次转交会保留完整过程；同一发生记录的重放不会重复入账。
2. 机械后处理只清理全文首尾空白，保留对白、复沓和重复段落。是否重复讲述由带上下文的审查决定。
3. 最终审查和修复使用全部计划人物的原作证据及每单元起始状态，区分原作事实、已生成事实和未来计划。
4. `StateChange` 支持目标/地点/阵营的 set/clear 和知识的 add/remove/replace。每项必须有原句证据，
   clear/remove/replace 必须匹配旧值；未提及不等于清空。状态变更历史可序列化供下一轮使用。
5. 历史检索同时考虑人物、物品、地点、势力和场景词项。原文切成小段，按中文 n-gram 和词项相关性排序；
   召回原句供模型核验，不在本地推断情节。默认最多 24 条相关事件、8 段原文、12000 字符检索证据。
   全部历史仍保存在状态中，滚动摘要按近期性和相关性另用 12000 字符预算呈现；必需人物状态和不可逆事实不截断。
6. 规划提示明确传入本次真实单元上限。模型可选择少于上限的单元数，仍需覆盖读者要求的叙事范围。
7. 风格抽样覆盖开头、中段、倒数第二及最后分析单元。最多保留 6 段可回溯的原文范例，按当前说话人、
   场景词项及近期性挑选；范例的来源和引文都需通过校验。段长、对白比例、修辞强度服从原作和场景。
8. `story eval` 支持正文区间隔离、冻结分析、重复生成、分阶段响应与指标、匿名对照和人工评分汇总。
   润色模式为 full（整单元润色，默认）、local（唯一引用的局部替换）、none（跳过润色）。三者都保留事实审查。
9. 续写通过断点保存各阶段响应，可恢复已完成工作；成功后导出与成稿一致的状态和合并正文，支持下一轮续写。
10. 末章候选片段包含反常对白、药物/死亡、宗教或动物意象、重复词和异常反应时，条件触发 `ending_subtext` 预分析。
    `NarrativeSignal` 必须携带原文连续证据、置信度、候选对应和候选后果；`SourceAnchor`、`AnchorAction` 与
    `PlotRecommendation` 再把奏章、药瓶、玉玺、火枪和隐喻等线索分配到可执行的推进、保留或背景处理。
    长引文只允许回指高度相似的原文连续片段；跨段拼接会回退到最长连续证据，短的或实质改写的引文仍被拒绝。

人物与世界伏笔仍按稳定 ID 管理，死亡只作用于明确的 `deceased_characters`。分析缓存格式为 6，
旧格式自动重建。人物抽取不再按 8 人固定截断；人物/事件的引文或格式错误可带原文和错误位置修复一次，
关系重试会收到具体拒绝原因。修复后仍需通过原有校验，失败不会保存为成功结果。

每次续写调用按 UTF-8 字节数保守估算 token，并预留输出空间和消息开销。总预算超限或模型返回
`finish_reason=length` 时明确失败，不截断必要证据或接受半截输出。

## 断点与下一轮续写

```bash
uv run story continue novel.txt --hint "承接当前场景" --max-units 2 --checkpoint .story_cache/run.json
uv run story continue novel.txt --hint "承接当前场景" --max-units 2 --checkpoint .story_cache/run.json --resume
uv run story continue .story_cache/run.context.source.txt --context .story_cache/run.context.json --hint "继续下一场"
```

恢复时须使用相同原文、分析、hint、模型选项、单元上限与实现。响应通过原子写入和哈希校验保存，
文件锁防止两个进程同时写同一断点。已接受响应直接回放，重建同一份计划、正文和状态；解析或校验失败的
最后响应会丢弃，网络失败不丢弃此前接受的阶段。改变原文或实现时应创建新断点。

开启缓存且没有 `--no-cache` 时，CLI 自动在 `.story_cache/continuations/` 创建断点并显示路径。
`--resume` 必须明确指定 `--checkpoint`；`--state-output PATH` 可单独指定状态输出。
默认状态文件为断点旁的 `.context.json`，合并原作与续写的下轮输入为 `.context.source.txt`。
最终局部修订若改变正文，先从修订后的各单元重建记忆，再导出状态；下一轮核对正文哈希，防止误配。
当前文件锁使用 POSIX `fcntl`，适用于 macOS/Linux。

正常无修复时，full/local 的请求数为 `3 + 4N + R + S`，none 为 `3 + 3N + R + S`：
N 为实际单元数，R 为伏笔审查（0/1），S 为候选深意预分析（检测到候选片段时为 1，否则为 0）。每次单元修复与重审增加 2 次调用，终稿修复与重审也增加 2 次；
最终修订改变正文后再增加 N 次状态重建审查。CLI 展示按 max_units 计算的基线估算，实际修复另计。
记忆输出的字段、引文或旧值不匹配时，可带具体反馈修复一次，增加 1 次调用；修复通过前不提交状态。
记忆审查额外收到本单元提及人物的精确旧状态；场景规划字段遗漏也可反馈修复一次。
local 模式允许空修改列表或经过唯一引文校验的原样保留；要求修复事实错误时，空修改仍会被拒绝。

## 准备评测输入

`cutoff_chars` 和可选 `start_chars` 使用解码后的 Python 字符数，源文件路径相对 manifest 解析。
默认使用从头到截断点的正文；指定 start_chars 可进行局部试跑，但结论只适用于提供的窗口。
对未完稿的当前结尾，必须明确 `at_end: true`，cutoff_chars 等于解码后总长度。

```json
[
  {
    "id": "unfinished-tail",
    "source": "/path/to/novel.txt",
    "start_chars": 100000,
    "cutoff_chars": 120000,
    "at_end": true,
    "hint": "续写当前场景，不提前收束全书；仅依据提供的原文",
    "max_units": 1
  }
]
```

该例假设源文总长恰为 120000 字符。为已完稿小说设置中途截断点时省略 at_end。

```bash
uv run story eval prepare cases.json --repetitions 3 --output-dir outputs/eval-inputs
```

准备不调用模型、不修改原文件。输出只含选定的正文、hint、单元上限、源区间和正文哈希，
不会复制被排除的后文或携带源文件路径。验证哈希时保留原有换行符。所有输出目录必须是新目录。

## 冻结分析与润色对照

```bash
uv run story eval analyze outputs/eval-inputs/cases.json --config model.yaml --output-dir outputs/eval-analysis
uv run story eval generate outputs/eval-inputs/cases.json --config model.yaml --analysis outputs/eval-analysis/analysis.json --revision-mode none --output-dir outputs/eval-none
uv run story eval generate outputs/eval-inputs/cases.json --config model.yaml --analysis outputs/eval-analysis/analysis.json --revision-mode full --output-dir outputs/eval-full
uv run story eval generate outputs/eval-inputs/cases.json --config model.yaml --analysis outputs/eval-analysis/analysis.json --revision-mode local --output-dir outputs/eval-local
```

`analyze` 只分析准备后的正文，不读外部缓存；相同正文哈希只分析一次，复用到所有重复样本。
冻结包包含 context、segments、index、分析配置白名单和实现摘要，每次加载检查正文及包哈希并反序列化独立状态。
比较写作策略时必须共用这份包。省略 `--analysis` 则每个 case 重新分析，适合评估包含分析差异的完整方案。

每个生成 case 使用配置 seed 加 case 序号；不同策略按相同顺序运行可匹配种子，但服务端未必保证确定性。
`results.json` 保存状态、模型、正文/分析哈希、实现摘要和非敏感配置；逐请求记录阶段、耗时、token、缓存 token
及异常类型。`responses-*` 保存模型响应，`*.trace.json` 保存计划、初稿、润色稿和单元审查后的正文。
这些材料含小说内容，留在本地实验目录，不进入 blind 目录，也不记录 API 凭证和服务 URL。

失败 case 会立即写入结果，CLI 返回非零状态；已完成 case 保留，失败输出不能进入盲评。
API 请求成功不等于后续解析/审查成功，须同时检查 pipeline status、异常类型和响应记录。
当前评测生成本身不提供 `--resume`；续写阶段断点适用于 `story continue`。

固定分析与种子只能减少差异；三种模式各自生成的初稿仍可能不同。应结合 trace 检查计划/初稿差异，
跨多个截断点、小说和重复样本比较。调用所提供的正文不含后文，也不能保证模型训练中从未见过原作。

## 匿名打包与评分

```bash
uv run story eval blind outputs/eval-inputs/cases.json outputs/eval-full/results.json outputs/eval-local/results.json --output-dir outputs/eval-comparison
```

两组必须包含相同 case、正文哈希、hint、单元上限；使用冻结分析时分析哈希也必须相同。
对每个 case 随机分配 A/B，只交付 `outputs/eval-comparison/blind/` 给评审，其中含输入正文、要求、
A/B 续写、说明和空白评分。`private_key.json` 留在 blind 目录之外，评审结束前不揭示映射。
`--seed` 可重现分配，种子也不向评审披露。

| 字段 | 评分含义 |
| --- | --- |
| winner | A/B/tie |
| contradictions | 有原文和续写引文支持的事实矛盾数量，非负整数 |
| motivation | 人物选择与既有动机是否连贯，1–5 分 |
| knowledge_boundary | 视角和人物知情范围是否正确，1–5 分 |
| plot_progress | 是否通过行动推进情节，1–5 分 |
| style | 句式、对白、叙事姿态与原作是否一致，1–5 分 |
| evidence | 原文/续写引文及问题说明组成的字符串数组 |

```bash
uv run story eval report outputs/eval-comparison/blind/ratings.json outputs/eval-comparison/private_key.json -o outputs/eval-report.json
```

报告包含胜出数量、平局数、各维度均分、事实矛盾数及每万字符矛盾数。未填写评分、缺失 case、
重复 ID、无证据的矛盾计数不能汇总。输出是描述性结果，不自动宣称统计显著。

## 验证边界

离线测试覆盖事件发生身份、重复对白保留、全部计划人物证据、状态更正、线索检索和预算、风格引文、
局部润色、断点恢复与完整性、成稿状态导出、换行一致性、冻结分析隔离、失败留存及匿名评分流程。
这些测试验证程序行为。真实小说的风格、叙事推进和盲评胜率需要独立人工评分，不能由测试通过推定。
