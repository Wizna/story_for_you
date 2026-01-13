# 完整风格分析功能实现计划（扩展版）

## 问题描述

1. 当前 `story continue` 生成的结局像故事大纲，缺乏原作者文学风格
2. 风格分析仅用于续写，未应用于压缩、筛选、移除等其他功能
3. 续写过程过于简单，缺乏人类作者的创作流程

## 解决方案概述

1. **风格分析全局化**：`WritingStyle` 应用于所有四个核心功能
2. **多阶段续写流程**：模拟人类作者的创作过程（构思→大纲→草稿→修订→优化）

---

## Part A: 风格分析基础设施

### A1. 数据结构 - [context.py](../story_for_you/analysis/context.py)

```python
@dataclass
class StyleSample:
    source_chapter: int
    content: str          # 20-50字原文片段
    style_notes: str      # 为何典型

@dataclass
class WritingStyle:
    # 句式结构
    avg_sentence_length: int
    sentence_variety: str  # uniform | varied | mixed
    paragraph_density: str # sparse | medium | dense

    # 用词风格
    register: str          # literary | colloquial | classical | mixed
    characteristic_words: list[str]  # 特征词汇（最多8个）
    idiom_frequency: str   # none | sparse | moderate | heavy

    # 修辞手法
    metaphor_style: str
    description_focus: list[str]  # landscape, psychological, action
    parallelism_use: str   # rare | occasional | frequent

    # 叙事语气
    tone_markers: list[str]  # 常用语气词
    narrator_style: str      # detached | intimate | intrusive

    # 示例与摘要
    representative_samples: list[StyleSample]
    style_summary: str     # 100-150字风格总结，用于提示词
```

新增 `_render_style_section()` 方法，更新 `for_prompt()` 输出。

### A2. 风格提取器 - 新建 [extractors/style.py](../story_for_you/analysis/extractors/style.py)

```python
class StyleExtractor:
    SAMPLE_CHAPTERS = 3   # 首、中、尾章节
    SAMPLE_SIZE = 2000    # 每样本字符数

    def extract(self, chapters: list[str], summaries: list[ChapterSummary]) -> WritingStyle:
        samples = self._select_samples(chapters)
        pov_summary = self._summarize_pov(summaries)
        mood_summary = self._summarize_mood(summaries)
        prompt = self._build_prompt(samples, pov_summary, mood_summary)
        response = self.llm.generate(prompt=prompt)
        return self._parse_response(response.content)
```

### A3. 风格提取提示词 - 新建 [analysis/prompt_templates/style_extraction.txt](../story_for_you/analysis/prompt_templates/style_extraction.txt)

详见附录 A。

### A4. 独立风格分析命令（解耦设计）

**设计原则**：风格分析与主分析流程完全解耦，可独立运行。

#### 新增 CLI 命令 - [cli/main.py](../story_for_you/cli/main.py)

```bash
# 独立运行风格分析（不重新分析）
uv run story style novel.txt -o style.json

# 将风格注入已有的 context
uv run story style novel.txt --context existing_context.json --inject

# 分析时可选启用风格提取
uv run story analyze novel.txt --with-style
```

```python
@app.command(name="style")
def analyze_style(
    input_file: Path,
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    context_path: Optional[Path] = typer.Option(None, "--context"),
    inject: bool = typer.Option(False, "--inject", help="将风格注入已有context"),
) -> None:
    """独立提取写作风格，不依赖完整分析流程。"""
    text = input_file.read_text(encoding="utf-8")
    chapters = split_chapters(text)  # 简单章节分割

    extractor = StyleExtractor(llm)
    style = extractor.extract_from_raw(chapters)  # 直接从原文提取

    if inject and context_path:
        # 注入已有 context
        context = load_context(context_path)
        context.writing_style = style
        save_context(context, context_path)
    else:
        # 单独保存风格
        save_style(style, output or input_file.with_suffix("_style.json"))
```

#### StyleExtractor 双模式 - [extractors/style.py](../story_for_you/analysis/extractors/style.py)

```python
class StyleExtractor:
    def extract(self, chapters: list[str], summaries: list[ChapterSummary]) -> WritingStyle:
        """从已分析的章节摘要中提取风格（集成模式）。"""
        ...

    def extract_from_raw(self, chapters: list[str]) -> WritingStyle:
        """直接从原始章节文本提取风格（独立模式）。"""
        # 不依赖 ChapterSummary，自行推断 POV 和 mood
        samples = self._select_samples(chapters)
        prompt = self._build_standalone_prompt(samples)
        response = self.llm.generate(prompt=prompt)
        return self._parse_response(response.content)
```

#### 核心功能自动加载风格

修改四个核心命令（compress/filter/remove/continue），支持自动查找风格文件：

```python
# 在 CLI 命令中
def _load_style(input_file: Path, context: StoryContext) -> WritingStyle | None:
    """按优先级加载风格：context内嵌 > 同名style文件 > None"""
    if context.writing_style:
        return context.writing_style
    style_file = input_file.with_suffix("_style.json")
    if style_file.exists():
        return load_style(style_file)
    return None
```

---

## Part B: 风格应用于所有核心功能

### B1. 统一风格注入工具 - [core/prompting.py](../story_for_you/core/prompting.py)

新增函数：

```python
def format_style_guide(style: WritingStyle | None) -> str:
    """格式化风格指南，用于所有提示词模板。"""
    if not style:
        return "(无风格信息，请保持中性文学风格)"
    return style.style_summary

def format_style_samples(style: WritingStyle | None, max_samples: int = 2) -> str:
    """格式化风格示例片段。"""
    if not style or not style.representative_samples:
        return "(无示例片段)"
    lines = [f"「{s.content}」" for s in style.representative_samples[:max_samples]]
    return "\n".join(lines)
```

### B2. 修改压缩模板 - [core/prompt_templates/compress.txt](../story_for_you/core/prompt_templates/compress.txt)

增加风格保持要求：

```markdown
## 风格指南
{{style_guide}}

## 风格示例
{{style_samples}}

## 输出要求（更新）
- 维持时间顺序与 POV 连贯
- 明确保留主线冲突、人物抉择、不可逆事件
- **严格保持原作文风**：句式节奏、用词习惯、描写手法需与示例一致
- 压缩后的文字应像原作者写的精简版，而非编辑的概要
```

### B3. 修改筛选桥接模板 - [core/prompt_templates/filter_bridge.txt](../story_for_you/core/prompt_templates/filter_bridge.txt)

```markdown
## 风格指南
{{style_guide}}

## 输出要求（更新）
- 1~2 句中文描述，聚焦人物与冲突承接
- **模仿原作风格**：使用原作的句式和语气词
- 过渡文字应自然融入上下文，而非编辑注释
```

### B4. 修改移除重写模板 - [core/prompt_templates/remove_rewrite.txt](../story_for_you/core/prompt_templates/remove_rewrite.txt)

```markdown
## 风格指南
{{style_guide}}

## 输出要求（更新）
- 维持原段落关键信息
- 不得再出现需移除人物的名字
- **保持原作文风**：重写后的段落应与原文风格一致
```

### B5. 更新核心类以传递风格

修改以下文件，在调用 `fill_template` 时传入 `style_guide` 和 `style_samples`：

| 文件 | 方法 |
|------|------|
| [compressor.py](../story_for_you/core/compressor.py) | `compress()` |
| [character_filter.py](../story_for_you/core/character_filter.py) | `_generate_bridge()` |
| [character_remover.py](../story_for_you/core/character_remover.py) | `_minimal_rewrite()` |

示例修改（compressor.py）：

```python
def compress(self, text: str, context: StoryContext) -> str:
    context_block = format_context_sections(context.for_prompt())
    style_guide = format_style_guide(context.writing_style)
    style_samples = format_style_samples(context.writing_style)
    prompt = fill_template(
        self.template,
        level=self.level,
        context_block=context_block,
        segments=segments_payload,
        style_guide=style_guide,      # 新增
        style_samples=style_samples,  # 新增
    )
```

---

## Part C: 多阶段续写流程

### C1. 创作流程设计

模拟人类作者的五阶段创作过程：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. 构思灵感  │ →  │  2. 规划大纲  │ →  │  3. 草稿写作  │ →  │  4. 修订编辑  │ →  │  5. 反馈优化  │
│  Inspiration │    │   Outline   │    │    Draft    │    │   Revision  │    │   Polish    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                   │                 │                  │                  │
      ▼                   ▼                 ▼                  ▼                  ▼
   主题分析           结构规划          初稿生成           风格校正           最终润色
   情感基调           关键节点          场景描写           连贯性检查          细节打磨
   结局方向           转折设计          人物刻画           语气词调整          意象升华
```

### C2. 数据结构 - [core/ending_writer.py](../story_for_you/core/ending_writer.py)

```python
@dataclass
class EndingOutline:
    """续写大纲结构"""
    theme: str                    # 主题/情感基调
    ending_direction: str         # 结局走向（HE/BE/OE）
    key_beats: list[str]          # 关键情节点（3-5个）
    emotional_arc: str            # 情感曲线描述
    final_image: str              # 结尾意象

@dataclass
class EndingDraft:
    """初稿结构"""
    content: str                  # 初稿正文
    paragraph_count: int          # 段落数
    word_count: int               # 字数

@dataclass
class EndingRevision:
    """修订记录"""
    original: str                 # 修订前
    revised: str                  # 修订后
    changes: list[str]            # 修改说明

@dataclass
class MultiStageEndingResult:
    """多阶段续写结果"""
    outline: EndingOutline
    draft: EndingDraft
    final_content: str
    revision_notes: list[str]
```

### C3. 重构 EndingWriter - [core/ending_writer.py](../story_for_you/core/ending_writer.py)

```python
class EndingWriter:
    """多阶段续写器，模拟人类作者创作流程。"""

    def __init__(self, llm: LLMProvider, segment_index: SegmentIndex):
        self.llm = llm
        self.segment_index = segment_index
        # 加载各阶段模板
        self.inspiration_template = load_template("ending_inspiration")
        self.outline_template = load_template("ending_outline")
        self.draft_template = load_template("ending_draft")
        self.revision_template = load_template("ending_revision")
        self.polish_template = load_template("ending_polish")

    def continue_story(self, text: str, context: StoryContext, hint: str = "") -> str:
        """执行完整的多阶段续写流程。"""
        style = context.writing_style

        # 阶段 1: 构思灵感
        inspiration = self._phase_inspiration(context, hint)

        # 阶段 2: 规划大纲
        outline = self._phase_outline(context, inspiration, hint)

        # 阶段 3: 草稿写作
        draft = self._phase_draft(context, outline, style)

        # 阶段 4: 修订编辑
        revised = self._phase_revision(draft, style, context)

        # 阶段 5: 反馈优化
        final = self._phase_polish(revised, style, outline)

        return final

    def _phase_inspiration(self, context: StoryContext, hint: str) -> dict:
        """阶段1: 分析故事主题、情感基调、可能的结局方向。"""
        prompt = fill_template(
            self.inspiration_template,
            context_block=format_context_sections(context.for_prompt()),
            recent_events=self._format_recent_events(context),
            unresolved_threads=self._format_unresolved(context),
            hint=hint or "无特别要求",
        )
        response = self.llm.generate(prompt=prompt)
        return self._parse_inspiration(response.content)

    def _phase_outline(self, context: StoryContext, inspiration: dict, hint: str) -> EndingOutline:
        """阶段2: 基于灵感构思规划具体大纲。"""
        prompt = fill_template(
            self.outline_template,
            inspiration=json.dumps(inspiration, ensure_ascii=False),
            characters=self._format_main_characters(context),
            conflicts=self._format_conflicts(context),
            hint=hint,
        )
        response = self.llm.generate(prompt=prompt)
        return self._parse_outline(response.content)

    def _phase_draft(self, context: StoryContext, outline: EndingOutline, style: WritingStyle | None) -> str:
        """阶段3: 按大纲写作初稿，应用风格指南。"""
        prompt = fill_template(
            self.draft_template,
            outline=self._format_outline(outline),
            style_guide=format_style_guide(style),
            style_samples=format_style_samples(style),
            recent_segments=self._recent_segment_digest(),
        )
        response = self.llm.generate(prompt=prompt)
        return response.content.strip()

    def _phase_revision(self, draft: str, style: WritingStyle | None, context: StoryContext) -> str:
        """阶段4: 检查并修订初稿，确保风格一致性。"""
        prompt = fill_template(
            self.revision_template,
            draft=draft,
            style_guide=format_style_guide(style),
            characteristic_words=", ".join(style.characteristic_words) if style else "",
            tone_markers=", ".join(style.tone_markers) if style else "",
            checklist=self._revision_checklist(style),
        )
        response = self.llm.generate(prompt=prompt)
        return response.content.strip()

    def _phase_polish(self, revised: str, style: WritingStyle | None, outline: EndingOutline) -> str:
        """阶段5: 最终润色，强化意象和情感。"""
        prompt = fill_template(
            self.polish_template,
            revised_content=revised,
            final_image=outline.final_image,
            emotional_arc=outline.emotional_arc,
            style_summary=style.style_summary if style else "",
        )
        response = self.llm.generate(prompt=prompt)
        return response.content.strip()
```

### C4. 新增提示词模板

#### [core/prompt_templates/ending_inspiration.txt](../story_for_you/core/prompt_templates/ending_inspiration.txt)

```markdown
# 续写灵感构思

## 角色
你是一位资深文学编辑，正在为一部小说构思结局方向。

## 输入
- 故事上下文: {{context_block}}
- 近期重要事件: {{recent_events}}
- 未解决的伏笔: {{unresolved_threads}}
- 读者期望: {{hint}}

## 任务
分析故事的主题内核和情感走向，提出结局构思。

## 输出（JSON）
{
  "core_theme": "故事核心主题（10字内）",
  "emotional_tone": "整体情感基调",
  "possible_endings": ["可能的结局方向1", "方向2", "方向3"],
  "recommended_direction": "推荐的结局类型（HE/BE/OE）及理由",
  "key_resolution": "最需要解决的核心矛盾"
}
```

#### [core/prompt_templates/ending_outline.txt](../story_for_you/core/prompt_templates/ending_outline.txt)

```markdown
# 续写大纲规划

## 角色
你是小说作者，正在规划结局的具体结构。

## 输入
- 灵感构思: {{inspiration}}
- 主要人物: {{characters}}
- 核心冲突: {{conflicts}}
- 读者期望: {{hint}}

## 任务
规划 3-5 个关键情节点，设计情感曲线和结尾意象。

## 输出（JSON）
{
  "theme": "本次续写的主题（15字内）",
  "ending_direction": "HE/BE/OE",
  "key_beats": [
    "情节点1: ...",
    "情节点2: ...",
    "情节点3: ..."
  ],
  "emotional_arc": "情感曲线描述（如：低落→挣扎→释然→希望）",
  "final_image": "结尾意象（如：渡船在晨雾中缓缓驶来）"
}
```

#### [core/prompt_templates/ending_draft.txt](../story_for_you/core/prompt_templates/ending_draft.txt)

```markdown
# 初稿写作

## 角色
你是小说作者，正在按大纲写作结局初稿。

## 输入
- 大纲: {{outline}}
- 最近片段: {{recent_segments}}

## 风格指南
{{style_guide}}

## 风格示例
{{style_samples}}

## 任务
按大纲写作 3-4 段结局正文。

## 要求
- 严格模仿风格示例的句式和用词
- 每段 3-5 句，有景物/心理/动作描写
- 承接最近剧情，不要跳跃
- 情感自然过渡，不要突兀
- 此为初稿，允许粗糙
```

#### [core/prompt_templates/ending_revision.txt](../story_for_you/core/prompt_templates/ending_revision.txt)

```markdown
# 修订编辑

## 角色
你是文学编辑，正在修订初稿以确保风格一致性。

## 输入
- 初稿: {{draft}}

## 风格检查清单
{{checklist}}

## 特征词汇（应使用）
{{characteristic_words}}

## 语气词（应使用）
{{tone_markers}}

## 任务
修订初稿，确保：
1. 句式节奏与原作一致
2. 适当加入特征词汇和语气词
3. 描写手法符合原作风格
4. 段落间过渡自然

## 输出
直接输出修订后的正文，不要解释。
```

#### [core/prompt_templates/ending_polish.txt](../story_for_you/core/prompt_templates/ending_polish.txt)

```markdown
# 最终润色

## 角色
你是作者本人，正在对作品做最后润色。

## 输入
- 修订稿: {{revised_content}}
- 预设结尾意象: {{final_image}}
- 情感曲线: {{emotional_arc}}
- 风格摘要: {{style_summary}}

## 任务
最终润色，重点：
1. 强化结尾意象的诗意表达
2. 确保情感曲线完整
3. 检查语言的音韵美感
4. 添加"（读者定制版本）"标注

## 输出
输出最终版本正文。
```

---

## 关键文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| [context.py](../story_for_you/analysis/context.py) | 修改 | 新增 StyleSample, WritingStyle |
| [extractors/style.py](../story_for_you/analysis/extractors/style.py) | 新建 | StyleExtractor 类（双模式） |
| [extractors/__init__.py](../story_for_you/analysis/extractors/__init__.py) | 修改 | 导出 StyleExtractor |
| [analysis/prompt_templates/style_extraction.txt](../story_for_you/analysis/prompt_templates/style_extraction.txt) | 新建 | 风格提取提示词 |
| [cli/main.py](../story_for_you/cli/main.py) | 修改 | 新增 `story style` 命令 |
| [core/prompting.py](../story_for_you/core/prompting.py) | 修改 | 新增 format_style_guide/samples |
| [compressor.py](../story_for_you/core/compressor.py) | 修改 | 传递风格参数 |
| [character_filter.py](../story_for_you/core/character_filter.py) | 修改 | 传递风格参数 |
| [character_remover.py](../story_for_you/core/character_remover.py) | 修改 | 传递风格参数 |
| [ending_writer.py](../story_for_you/core/ending_writer.py) | 重构 | 多阶段续写流程 |
| [core/prompt_templates/compress.txt](../story_for_you/core/prompt_templates/compress.txt) | 修改 | 增加风格占位符 |
| [core/prompt_templates/filter_bridge.txt](../story_for_you/core/prompt_templates/filter_bridge.txt) | 修改 | 增加风格占位符 |
| [core/prompt_templates/remove_rewrite.txt](../story_for_you/core/prompt_templates/remove_rewrite.txt) | 修改 | 增加风格占位符 |
| [core/prompt_templates/ending_*.txt](../story_for_you/core/prompt_templates/) | 新建 | 5个续写阶段模板 |

---

## Token 预算分析

| 阶段 | 输入 tokens | 输出 tokens | 总计 |
|------|-------------|-------------|------|
| 风格提取 | ~2800 | ~500 | ~3300 |
| 续写-灵感 | ~1500 | ~200 | ~1700 |
| 续写-大纲 | ~1000 | ~300 | ~1300 |
| 续写-初稿 | ~1500 | ~800 | ~2300 |
| 续写-修订 | ~1200 | ~800 | ~2000 |
| 续写-润色 | ~1000 | ~800 | ~1800 |

续写总计：~9100 tokens（5次 LLM 调用）

---

## 验证方式

### 1. 独立风格提取（不重新分析）

```bash
# 对已分析过的小说单独提取风格
uv run story style /path/to/novel.txt -o novel_style.json

# 或注入已有 context
uv run story style /path/to/novel.txt --context .story_cache/novel_context.json --inject
```

检查 `novel_style.json` 内容：

- `style_summary` 是否准确描述作者风格
- `characteristic_words` 是否包含特征词汇
- `representative_samples` 是否选取了典型片段

### 2. 压缩风格验证

```bash
# 会自动加载 novel_style.json（如存在）
uv run story compress /path/to/novel.txt --level medium -o novel_compressed.txt
```

对比压缩结果是否保持原作文风。

### 3. 多阶段续写验证

```bash
uv run story continue /path/to/novel.txt --hint "希望是HE" -o novel_ending.txt
```

检验结果：

- 是否有诗意的景物描写
- 是否使用了原作的特征词汇
- 句式是否与原作一致
- 情感是否自然过渡

---

## 附录 A: 风格提取完整提示词

```markdown
# 写作风格提取

## 角色
你是一名文学风格分析师，负责从给定的章节样本中提取作者的写作风格特征。分析需基于原文证据，不要猜测或编造。

## 输入
1. **样本章节**：
{{chapter_samples}}

2. **已知作品信息**：
- 叙事视角: {{pov_summary}}
- 主要情绪基调: {{mood_summary}}

## 分析维度
1. **句式结构**：句子平均长度、长短句变化、是否使用排比/对偶
2. **用词风格**：文学腔/口语/文言混用程度、特征词汇（反复出现的词）
3. **修辞手法**：比喻类型、景物/心理/动作描写偏好
4. **叙事语气**：语气词使用、叙述者介入程度
5. **代表性片段**：选取2-3个最能体现风格的短句（每个20-40字）

## 输出格式
返回 UTF-8 JSON 对象：
{
  "avg_sentence_length": <int>,
  "sentence_variety": "uniform | varied | mixed",
  "paragraph_density": "sparse | medium | dense",
  "register": "literary | colloquial | classical | mixed",
  "characteristic_words": ["词1", "词2", ...],
  "idiom_frequency": "none | sparse | moderate | heavy",
  "metaphor_style": "<20字内描述>",
  "description_focus": ["landscape", "psychological", "action"],
  "parallelism_use": "rare | occasional | frequent",
  "tone_markers": ["呢", "吧", ...],
  "narrator_style": "detached | intimate | intrusive",
  "representative_samples": [
    {"source_chapter": <int>, "content": "<原文片段>", "style_notes": "<为何典型>"}
  ],
  "style_summary": "<100-150字中文风格总结>"
}
```
