# 技术架构开发文档（v0.1）

## 1. 系统架构概述

### 1.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI 层                              │
│  story analyze | compress | filter | remove | continue      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                       Core 业务层                            │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│  │Compressor │ │CharFilter │ │CharRemover│ │EndingWriter│   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘   │
└────────┼─────────────┼─────────────┼─────────────┼──────────┘
         │             │             │             │
         └─────────────┴──────┬──────┴─────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     内容理解层 (Analysis)                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │
│  │ Character   │ │   Event     │ │    Relationship     │    │
│  │ Extractor   │ │  Extractor  │ │      Mapper         │    │
│  └─────────────┘ └─────────────┘ └─────────────────────┘    │
│                    ┌─────────────┐                           │
│                    │StoryContext │  ← 统一上下文对象         │
│                    └─────────────┘                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                       LLM 抽象层                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              LLMProvider (Abstract)                 │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │                                 │
│  ┌─────────────┐  ┌────────▼────────┐  ┌─────────────┐      │
│  │ OpenAI      │  │ Ollama (默认)   │  │ 其他实现... │      │
│  └─────────────┘  └─────────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────┐
│                      基础设施层                              │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                  │
│  │TextParser │ │  Config   │ │   Utils   │                  │
│  └───────────┘ └───────────┘ └───────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 设计原则

- **模块化**：每个功能独立成模块，便于维护和测试
- **可扩展**：LLM 层抽象，支持多种模型后端
- **内容理解优先**：所有处理基于对小说的深度理解（人物、事件、关系）
- **简单优先**：CLI 为主，避免过度设计
- **可复现**：相同参数产出相同结果（设置固定 seed）

---

## 2. 技术选型

| 组件       | 选择                    | 理由                       |
| ---------- | ----------------------- | -------------------------- |
| 语言       | Python 3.11+            | 生态丰富，LLM 集成方便     |
| LLM 运行时 | Ollama                  | 本地部署简单，Mac 友好     |
| 默认模型   | Qwen2.5-7B-Instruct     | 中文优秀，内存 < 16G       |
| CLI 框架   | Typer                   | 类型提示友好，自动生成帮助 |
| LLM 调用   | httpx + Ollama REST API | 轻量，无额外依赖           |
| 配置管理   | Pydantic + YAML         | 类型安全，易于验证         |
| 测试       | pytest                  | Python 标准选择            |

---

## 3. 目录结构

```
story_for_you/
├── story_for_you/                 # Python package（直接作为 PyPI 包发布）
│   ├── __init__.py
│   ├── cli/                      # CLI 入口层
│   │   ├── __init__.py
│   │   └── main.py               # Typer 命令定义
│   │
│   ├── analysis/                 # 内容理解层（核心）
│   │   ├── __init__.py
│   │   ├── context.py            # StoryContext + dataclasses
│   │   ├── layers/               # 三层记忆结构
│   │   │   ├── __init__.py
│   │   │   ├── chapter_window.py # 短期：章节滑窗
│   │   │   ├── event_ledger.py   # 中期：事件账本
│   │   │   └── state_store.py    # 长期：状态快照
│   │   ├── extractors/           # LLM 驱动的语义抽取
│   │   │   ├── __init__.py
│   │   │   ├── chapters.py       # ChapterSummarizer
│   │   │   ├── characters.py     # CharacterExtractor + PersonalityAnalyzer
│   │   │   ├── events.py         # EventExtractor（含 impact）
│   │   │   ├── relationships.py  # RelationshipMapper
│   │   │   └── state.py          # StateSynthesizer
│   │   └── prompt_templates/     # Prompt 统一维护
│   │       ├── chapter_summary.txt
│   │       ├── event_extraction.txt
│   │       └── state_update.txt
│   │
│   ├── indexer/                  # 文本索引与检索
│   │   ├── __init__.py
│   │   ├── segment.py            # 段落/场景分割
│   │   ├── tagger.py             # 人物标注器
│   │   └── retriever.py          # 段落检索
│   │
│   ├── cache/                    # 分析结果缓存
│   │   ├── __init__.py
│   │   └── store.py              # 缓存存储管理
│   │
│   ├── core/                     # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── compressor.py         # 情节压缩
│   │   ├── character_filter.py   # 只看指定人物
│   │   ├── character_remover.py  # 删除人物
│   │   └── ending_writer.py      # 结局续写
│   │
│   ├── llm/                      # LLM 抽象层
│   │   ├── __init__.py
│   │   ├── base.py               # 抽象基类
│   │   └── ollama.py             # Ollama 实现
│   │
│   ├── parser/                   # 文本处理
│   │   ├── __init__.py
│   │   └── text_splitter.py      # 文本分块
│   │
│   ├── config/                   # 配置管理
│   │   ├── __init__.py
│   │   └── settings.py           # 配置定义
│   │
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       └── file_io.py            # 文件读写
│
├── tests/                        # 测试目录（与包结构一一对应）
│   ├── test_analysis.py
│   ├── test_compressor.py
│   ├── test_character_filter.py
│   └── ...
│
├── .story_cache/                 # 默认缓存目录（gitignore）
│   └── {file_hash}/              # 按文件哈希组织
│       └── context.json          # 缓存的分析结果
│
├── docs/                         # 文档
│   ├── specs01.md
│   └── dev-architecture.md
│
├── .story_cache/ (runtime)       # 命令执行后生成的缓存目录（gitignore）
├── pyproject.toml                # PEP 621 配置（包含入口脚本）
└── README.md

> 注：原设计中的 `tests/` 与 `config.example.yaml` 仍在规划阶段，当前代码库尚未提供，可在需要时再补齐。
```

---

## 4. 模块设计

### 4.1 内容理解层 (`analysis/`) - 核心模块

**目的**：遵循 `docs/analysis_memory_design.md` 中的“三层记忆”方案，产出可验证的 `StoryContext`，让四大核心业务在最少 LLM 调用下保持剧情一致。详细设计补充在该文档中。

#### 4.1.1 StoryContext 与三层记忆 (`context.py` + `layers/`)

```python
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

@dataclass
class ChapterSummary:
    """短期记忆：最近章节摘要"""
    chapter: int
    title: str
    pov: str
    beats: list[str]
    mood: str
    synopsis: str
    irreversible_flags: list[str] = field(default_factory=list)

@dataclass
class EventImpact:
    power_shifts: dict[str, str] = field(default_factory=dict)
    relation_changes: dict[str, str] = field(default_factory=dict)
    world_flags: list[str] = field(default_factory=list)

@dataclass
class PlotEvent:
    """中期记忆：长期影响事件"""
    event_id: str
    chapter: int
    type: Literal["conflict", "reveal", "progress", "setback"]
    participants: list[str]
    summary: str
    impact: EventImpact
    is_irreversible: bool = False

@dataclass
class Relationship:
    targets: list[str]
    relation_type: str
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    description: str = ""
    source: str | None = None

@dataclass
class CharacterState:
    """长期记忆：人物状态"""
    name: str
    aliases: list[str] = field(default_factory=list)
    realm: str | None = None
    role: Literal["main", "support", "minor"] = "minor"
    personality: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

@dataclass
class StoryState:
    current_arc: str
    world_tension: Literal["low", "medium", "high"]
    major_conflicts: list[str]
    time_constraints: list[str]
    unresolved_events: list[str]

@dataclass
class StoryContext:
    """统一上下文容器"""
    metadata: dict[str, Any] = field(default_factory=dict)
    chapter_window: list[ChapterSummary] = field(default_factory=list)
    events: list[PlotEvent] = field(default_factory=list)
    characters: dict[str, CharacterState] = field(default_factory=dict)
    story_state: StoryState | None = None

    def for_prompt(self) -> dict[str, Any]:
        """拆分为 prompt 需要的几个段落"""
        ...

> 关系去重策略：`Relationship.targets` 记录与 `source` 同句出现的所有人物，因而每段 `description` 只会写入一次，就算原文涉及多人也不会在缓存中反复出现。
```

- **短期记忆**：`ChapterSummaryWindow`（`layers/chapter_window.py`）维护最近 N 章（默认 12），提供 `append`, `dump`, `to_prompt_lines()`。
- **中期记忆**：`EventLedger`（`layers/event_ledger.py`）记录不可逆事件，自动生成 `event_id`，并按人物/章节建立索引。
- **长期记忆**：`StateStore`（`layers/state_store.py`）聚合人物/世界状态，暴露 `snapshot()` 供 `StoryContext` 使用。

三层记忆共同保证「不存全文、只存结构化状态」，同时所有条目都可在 prompt 中引用。

#### 4.1.2 抽取器与 Prompt（`extractors/` + `prompt_templates/`)

```python
class CharacterExtractor:
    def __init__(self, llm: LLMProvider): ...
    def extract(self, text: str) -> list[CharacterState]: ...
    def merge_aliases(self, characters: list[CharacterState]) -> list[CharacterState]: ...

class ChapterSummarizer:
    def summarize(self, chapter_text: str, chapter_no: int) -> ChapterSummary: ...

class RelationshipMapper:
    def map(self, chapter_text: str, characters: list[CharacterState]) -> list[Relationship]: ...

class EventExtractor:
    def extract(self, chapter_text: str, participants: list[str]) -> list[PlotEvent]: ...

class StateSynthesizer:
    def update(self, story_state: StoryState | None, events: list[PlotEvent]) -> StoryState: ...
```

- Prompt 模板集中在 `analysis/prompt_templates/`，便于测试快照稳定。
- `CharacterExtractor` 与 `PersonalityAnalyzer` 合并输出 `CharacterState`，性格锚点直接写入 `personality`，避免后续“性格漂移”。
- `RelationshipMapper` 以句子为粒度提取关系，把同一句中的所有其他人物一次性写在 `targets` 内，并把去除多余空白后的完整句子写入 `description`，既不截断也不重复。
- `EventExtractor` 使用 `prompt_templates/event_extraction.txt` 中的【事件标准】，过滤掉无长期影响的桥段。
- `StateSynthesizer` 接收旧状态 + 新事件，输出 diff，再由 `StateStore` 合并。

**Prompt 模板细则**

| 模板 | 输入片段 | 关键字段 | 约束 |
| --- | --- | --- | --- |
| `character_sheet.txt` | 章节原文 | `name`、`aliases`、`role`、`realm`、`personality`、`unresolved` | 最多 8 人，禁止杜撰，仅记录原文证实的信息 |
| `chapter_summary.txt` | 章节元信息、最近上下文、章节原文 | `pov`（first/third/omniscient/multi）、`beats` 3~6 条、`mood`、`synopsis`、`irreversible_flags` | 仅产出 1 个 JSON 对象；不可逆标记格式 `flag: reason`，类型限定为 `death/betrayal/identity/world_shift/binding/catastrophe` |
| `event_extraction.txt` | 章节号、人物清单、上下文、章节原文 | `event_id`(`CH{chapter}-E01`)、`type`、`participants`、`impact`、`is_irreversible` | 只保留对剧情长期有意义的事件；`participants` 需映射回主名；`impact` 字段允许空结构 |
| `state_update.txt` | 旧 `StoryState`、事件数组、上下文 | `current_arc`、`world_tension`、`major_conflicts`、`time_constraints`、`unresolved_events` | Arc 取值 `setup/journey/twist/climax/dark-night/resolution`；数组最长 5 条；无新增字段 |

模板中统一使用 `{{chapter_text}}`、`{{recent_context}}`、`{{events}}` 等占位符，后续在 `StoryAnalyzer` 中按需替换。文本采用 Markdown 说明 + JSON schema 例子，便于快照测试与手动审阅。

#### 4.1.3 分析协调器 (`StoryAnalyzer`)

```python
class StoryAnalyzer:
    def __init__(self, llm: LLMProvider, window_size: int = 12):
        self.chapter_window = ChapterSummaryWindow(window_size)
        self.event_ledger = EventLedger()
        self.state_store = StateStore()
        # 注入各类 extractors...

    def analyze(self, chapters: list[str]) -> StoryContext:
        for chapter_no, chapter_text in enumerate(chapters, start=1):
            characters = self.character_extractor.extract(chapter_text)
            relationships = self.relationship_mapper.map(chapter_text, characters)
            summary = self.chapter_summarizer.summarize(chapter_text, chapter_no)
            events = self.event_extractor.extract(chapter_text, [c.name for c in characters])

            self.chapter_window.append(summary)
            self.event_ledger.record(events)
            self.state_store.update(characters, relationships, events)

        return StoryContext(
            metadata=self._build_metadata(),
            chapter_window=self.chapter_window.dump(),
            events=self.event_ledger.timeline(),
            characters=self.state_store.characters_snapshot(),
            story_state=self.state_store.story_snapshot(),
        )
```

**流程**（简化）：

1. `parser/TextSplitter` 先按章节/场景切分，减少重复提示；
2. 每章执行 “人物 → 摘要 → 事件 → 状态” 的顺序，避免无用调用；
3. 所有输出即时写入三层记忆结构；
4. 最后组装 `StoryContext` 并交由缓存层持久化。

#### 4.1.4 StoryContext 数据约定

| 字段                                      | 说明                                                                | 作用范围                                      |
| ----------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------- |
| `chapter_window: list[ChapterSummary]`  | 最近 N 章摘要，用于短期记忆                                         | `continue`/`compress`                     |
| `events: list[PlotEvent]`               | 中期记忆，含 impact/irreversible                                    | `compress`/`filter`/`remove`/`ending` |
| `characters: dict[str, CharacterState]` | 人物画像 + 性格锚点 + 未解要素                                      | 全部                                          |
| `story_state: StoryState`               | 当前大局状态、冲突、悬念                                            | prompt 拼装 & 决策                            |
| `metadata: dict[str, Any]`              | `_version`、`config_fingerprint`、`model`、`window_size` 等 | 缓存校验                                      |

`StoryContext.for_prompt()` 输出「世界观 → 人物状态 → 当前剧情线 → 最近章节摘要」四段文本，直接对应记忆文档中的黄金顺序。

### 4.2 缓存层 (`cache/`)

**目的**：持久化分析结果，避免重复分析相同文本

#### 4.2.1 缓存存储 (`store.py`)

```python
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

class ContextStore:
    """分析结果缓存管理"""

    def __init__(self, cache_dir: Path = Path(".story_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def _build_fingerprint(self, *, file_hash: str, config_hash: str) -> str:
        """组合文本哈希 + 配置指纹，避免跨模型/参数误用"""
        raw = f"{file_hash}:{config_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _get_config_hash(self, settings: dict) -> str:
        """将 llm/parser/cache 相关配置序列化后哈希"""
        payload = json.dumps(settings, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_file_hash(self, file_path: Path) -> str:
        """计算文件内容哈希"""
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    def _get_cache_dir(self, file_path: Path, settings: dict) -> Path:
        """获取缓存目录（包含 context/segments/index）"""
        key = self._build_fingerprint(
            file_hash=self._get_file_hash(file_path),
            config_hash=self._get_config_hash(settings),
        )
        return self.cache_dir / key

    def get(self, file_path: Path, settings: dict) -> CachedArtifacts | None:
        """获取缓存（StoryContext + segments + index）

        Returns:
            命中返回 `CachedArtifacts`（context, segments, char_index），否则 None
        """
        cache_dir = self._get_cache_dir(file_path, settings)
        ctx_path = cache_dir / "context.json"
        seg_path = cache_dir / "segments.json"
        idx_path = cache_dir / "index.json"
        if ctx_path.exists() and seg_path.exists() and idx_path.exists():
            return CachedArtifacts(
                context=StoryContext.from_dict(json.loads(ctx_path.read_text())),
                segments=json.loads(seg_path.read_text()),
                index=json.loads(idx_path.read_text()),
                metadata=json.loads((cache_dir / "meta.json").read_text()),
            )
        return None

    def save(self, file_path: Path, settings: dict, artifacts: CachedArtifacts) -> Path:
        """保存分析 + 索引结果"""
        cache_dir = self._get_cache_dir(file_path, settings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "context.json").write_text(
            json.dumps(artifacts.context.to_dict(), ensure_ascii=False, indent=2)
        )
        (cache_dir / "segments.json").write_text(
            json.dumps(artifacts.segments, ensure_ascii=False, indent=2)
        )
        (cache_dir / "index.json").write_text(
            json.dumps(artifacts.index, ensure_ascii=False, indent=2)
        )
        (cache_dir / "meta.json").write_text(
            json.dumps(
                {"config": settings, "created_at": datetime.now().isoformat()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return cache_dir

@dataclass
class CachedArtifacts:
    """一次分析结果的完整缓存包"""
    context: StoryContext
    segments: list[dict]
    index: dict[str, list[int]]
    metadata: dict[str, Any]

    def invalidate(self, file_path: Path, settings: dict) -> bool:
        """使缓存失效（文件修改或配置变化后调用）"""
        cache_dir = self._get_cache_dir(file_path, settings)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            return True
        return False

    def clear_all(self) -> int:
        """清空所有缓存，返回清理数量"""
        count = 0
        for sub in self.cache_dir.iterdir():
            shutil.rmtree(sub)
            count += 1
        return count
```

#### 4.2.2 缓存策略

| 场景                                         | 行为                                                 |
| -------------------------------------------- | ---------------------------------------------------- |
| 首次分析                                     | 分析后自动缓存（`context + segments + index`）     |
| 再次执行命令                                 | 根据 `config_hash` 命中缓存，直接复用上下文 + 索引 |
| 文件内容变更                                 | `file_hash` 变化 → 新 fingerprint → 重新分析     |
| CLI/配置变更 (`--model`, `--chunk-size`) | `config_hash` 改变 → 自动失效                     |
| 手动指定 `--context`                       | 加载提供的 `context.json` + 可选 `segments.json` |
| 指定 `--no-cache`                          | 不读取/写入缓存                                      |
| 指定 `--reanalyze`                         | 重新分析并覆盖缓存                                   |

#### 4.2.3 StoryContext 序列化

```python
@dataclass
class StoryContext:
    # ... 其他字段 ...

    def to_dict(self) -> dict:
        """序列化为字典（用于 JSON 存储）"""
        return {
            "metadata": self.metadata,
            "chapter_window": [asdict(summary) for summary in self.chapter_window],
            "events": [asdict(event) for event in self.events],
            "characters": {
                name: asdict(state)
                for name, state in self.characters.items()
            },
            "story_state": asdict(self.story_state) if self.story_state else None,
            "_version": "1.1",  # 缓存格式版本
            "_created_at": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StoryContext":
        """从字典反序列化"""
        return cls(
            metadata=data.get("metadata", {}),
            chapter_window=[
                ChapterSummary(**summary)
                for summary in data.get("chapter_window", [])
            ],
            events=[PlotEvent(**event) for event in data.get("events", [])],
            characters={
                name: CharacterState(**state)
                for name, state in data.get("characters", {}).items()
            },
            story_state=StoryState(**data["story_state"])
            if data.get("story_state")
            else None,
        )
```

---

### 4.3 索引检索层 (`indexer/`) - 检索优先策略

**核心理念**：最大程度保留原文，仅在必要时进行最小改写

#### 4.3.1 设计原则

| 原则               | 说明                                |
| ------------------ | ----------------------------------- |
| **检索优先** | 先通过文本搜索定位，再决定保留/删除 |
| **原文优先** | 尽可能保留原始文本，不做无谓改写    |
| **最小改写** | 仅修复逻辑断裂和语言不通顺之处      |
| **标记透明** | 清晰标注哪些是原文、哪些是桥接文本  |

#### 4.3.2 段落分割 (`segment.py`)

```python
@dataclass
class Segment:
    """文本段落单元"""
    id: str
    content: str                    # 原始文本
    chapter: int
    start_pos: int
    end_pos: int
    characters: set[str]            # 出现的人物
    is_scene_break: bool = False    # 是否为场景分界

class SegmentSplitter:
    """将小说切分为语义段落"""

    def split(self, text: str) -> list[Segment]:
        """智能分割文本

        分割依据：
        - 章节标记
        - 场景分界（空行、分隔符）
        - 对话边界
        - 视角切换
        """
        ...
```

#### 4.3.3 人物标注 (`tagger.py`)

```python
class CharacterTagger:
    """为每个段落标注涉及的人物"""

    def __init__(self, context: StoryContext):
        self.context = context
        # 构建人物名+别名的匹配词典
        self.name_patterns = self._build_patterns()

    def tag(self, segments: list[Segment]) -> list[Segment]:
        """为段落标注人物

        使用规则匹配（非 LLM）：
        - 精确匹配人物名和别名
        - 代词消解（简单规则：最近提及）
        """
        for seg in segments:
            seg.characters = self._find_characters(seg.content)
        return segments

    def _build_patterns(self) -> dict[str, str]:
        """构建 别名 -> 主名 的映射"""
        patterns = {}
        for name, char in self.context.characters.items():
            patterns[name] = name
            for alias in char.aliases:
                patterns[alias] = name
        return patterns
```

#### 4.3.4 段落检索 (`retriever.py`)

```python
class SegmentRetriever:
    """基于人物的段落检索"""

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        # 构建人物 -> 段落索引
        self.char_index: dict[str, list[int]] = self._build_index()

    def retrieve_by_characters(
        self,
        include: list[str],
        mode: str = "soft"
    ) -> list[Segment]:
        """检索包含指定人物的段落

        Args:
            include: 要包含的人物列表
            mode:
                - strict: 仅包含目标人物的段落
                - soft: 包含目标人物 + 相关情节（如场景开头）
        """
        ...

    def retrieve_excluding(
        self,
        exclude: list[str],
        mode: str = "hard"
    ) -> list[Segment]:
        """检索排除指定人物的段落

        Args:
            exclude: 要排除的人物列表
            mode:
                - hard: 完全不含该人物的段落
                - soft: 该人物非主要参与者的段落
        """
        ...
```

#### 4.3.5 处理策略对比

| 功能       | 传统方式（LLM 重写） | 检索优先方式                   |
| ---------- | -------------------- | ------------------------------ |
| 人物筛选   | LLM 逐段改写         | 检索 + 原文拼接 + 最小桥接     |
| 人物删除   | LLM 逐段改写         | 检索排除 + 原文保留 + 断点修复 |
| 原文保留度 | 低（大量改写）       | 高（90%+ 原文）                |
| LLM 调用量 | 高                   | 低（仅桥接处）                 |
| 幻觉风险   | 高                   | 低                             |

---

#### 4.3.6 SegmentIndexService（共享索引）

四个核心业务都通过统一的 `SegmentIndexService` 访问分段、人物标注和预计算指标。构建流程：

1. `SegmentSplitter.split` → 生成 `segments`
2. `CharacterTagger.tag` → 补充 `segment.characters`
3. `_precompute_gaps` → 计算场景断点、章节范围
4. `_build_index` → 输出 `SegmentIndex` 数据结构：

```python
@dataclass
class SegmentIndex:
    segments: list[Segment]
    char_index: dict[str, list[int]]   # 人物 -> 段落 id 集合
    chapter_index: dict[int, list[int]]
    gap_map: dict[str, Gap]            # 邻接段落之间的断裂
```

`SegmentIndexService` 在完成一次分析后立即运行，并与 `StoryContext` 一起写入缓存（参见 §4.2）。业务模块在初始化时接收 `(context, segment_index)`，禁止重复切分/标注，确保四个命令在同一份原文切片上工作。

### 4.4 LLM 抽象层 (`llm/`)

**目的**：解耦业务逻辑与具体 LLM 实现

```python
# llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    tokens_used: int

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        """生成文本"""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system: str = ""):
        """流式生成"""
        pass
```

```python
# llm/ollama.py
class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "qwen2.5:7b-instruct",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        # 调用 Ollama /api/generate 接口
        ...
```

### 4.5 文本解析层 (`parser/`)

**目的**：将长文本分块处理，避免超出 LLM 上下文限制

```python
# parser/text_splitter.py
@dataclass
class TextChunk:
    content: str
    start_pos: int
    end_pos: int
    chapter: str | None = None

class TextSplitter:
    def __init__(self, chunk_size: int = 4000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[TextChunk]:
        """按章节或字数分块"""
        ...

    def merge(self, chunks: list[str]) -> str:
        """合并处理后的文本块"""
        ...
```

### 4.6 核心业务层 (`core/`)

**说明**：所有业务处理模块都依赖 `StoryContext` 进行更智能的处理

#### 4.6.1 情节压缩 (`compressor.py`)

```python
class StoryCompressor:
    def __init__(self, llm: LLMProvider, segment_index: SegmentIndex, level: str = "medium"):
        self.llm = llm
        self.segment_index = segment_index
        self.level = level  # light | medium | heavy

    def compress(self, text: str, context: StoryContext) -> str:
        """压缩小说文本

        利用 context 信息：
        - 保护主要人物相关情节
        - 保留关键事件
        - 根据人物关系判断情节重要性
        - 结合 SegmentIndex + EventLedger 的重要度标签决定删除/保留比例
        """
        ...
```

**压缩 Prompt 策略**：

- 保留：主线剧情、关键对话、转折点、主要人物互动
- 删除：重复心理描写、无推进日常、信息回顾

#### 4.6.2 人物筛选 (`character_filter.py`) - 检索优先

```python
class CharacterFilter:
    """基于检索的人物筛选（最大保留原文）"""

    def __init__(self, llm: LLMProvider, retriever: SegmentRetriever):
        self.llm = llm
        self.retriever = retriever

    def filter(self, text: str, characters: list[str],
               context: StoryContext, mode: str = "soft") -> FilterResult:
        """筛选指定人物相关内容

        处理流程：
        1. 检索包含目标人物的段落（无 LLM）
        2. 原文拼接
        3. 仅在断点处生成桥接文本（LLM）
        """
        # Step 1: 检索相关段落
        segments = self.retriever.retrieve_by_characters(
            include=characters, mode=mode
        )

        # Step 2: 检测需要桥接的断点
        gaps = self._find_gaps(segments)

        # Step 3: 仅对断点生成桥接（最小 LLM 调用）
        bridges = [self._generate_bridge(gap, context) for gap in gaps]

        # Step 4: 拼接原文 + 桥接
        return self._assemble(segments, bridges)

    def _find_gaps(self, segments: list[Segment]) -> list[Gap]:
        """检测段落间的逻辑断裂"""
        ...

    def _generate_bridge(self, gap: Gap, context: StoryContext) -> str:
        """生成最小桥接文本（1-2句过渡）"""
        ...

@dataclass
class FilterResult:
    content: str                    # 最终文本
    original_ratio: float           # 原文保留比例
    bridges: list[BridgeInfo]       # 桥接位置信息（便于标注）
```

`SegmentRetriever` 基于缓存中的 `SegmentIndex` 初始化，只构建一次，即可被筛选与删除模块复用。

**筛选策略**：

| 模式       | 行为                                   |
| ---------- | -------------------------------------- |
| `strict` | 仅保留人物直接出现的段落               |
| `soft`   | 包含：直接出现 + 场景上下文 + 相关事件 |

---

#### 4.6.3 人物删除 (`character_remover.py`) - 检索优先

```python
class CharacterRemover:
    """基于检索的人物删除（最大保留原文）"""

    def __init__(self, llm: LLMProvider, retriever: SegmentRetriever):
        self.llm = llm
        self.retriever = retriever

    def remove(self, text: str, characters: list[str],
               context: StoryContext, mode: str = "hard") -> RemoveResult:
        """删除指定人物

        处理流程：
        1. 检索不包含目标人物的段落（保留）
        2. 检索包含目标人物的段落（待处理）
        3. 评估待处理段落：
           - 人物仅被提及 → 文本替换删除名字
           - 人物为主要参与者 → 整段删除
           - 人物影响主线逻辑 → 最小改写
        4. 拼接结果
        """
        # Step 1: 检索排除目标人物的段落
        kept_segments = self.retriever.retrieve_excluding(
            exclude=characters, mode=mode
        )

        # Step 2: 检索包含目标人物的段落
        affected_segments = self.retriever.retrieve_by_characters(
            include=characters, mode="strict"
        )

        # Step 3: 评估每个受影响段落
        processed = []
        for seg in affected_segments:
            action = self._evaluate_segment(seg, characters, context)
            if action.type == "keep_with_replace":
                # 简单文本替换，删除人物名
                processed.append(self._replace_names(seg, characters))
            elif action.type == "minimal_rewrite":
                # 最小改写（仅此处调用 LLM）
                processed.append(self._minimal_rewrite(seg, characters, context))
            # action.type == "delete" → 不加入

        # Step 4: 合并并修复断点
        return self._assemble_and_fix(kept_segments, processed, context)

    def _evaluate_segment(self, seg: Segment, chars: list[str],
                          ctx: StoryContext) -> Action:
        """评估段落处理方式

        判断依据：
        - 人物出现频次
        - 是否为对话主体
        - 是否涉及关键事件
        """
        ...

    def _replace_names(self, seg: Segment, chars: list[str]) -> Segment:
        """简单文本替换（无 LLM）"""
        ...

    def _minimal_rewrite(self, seg: Segment, chars: list[str],
                         ctx: StoryContext) -> Segment:
        """最小改写：仅修复逻辑，保留其他原文"""
        ...

@dataclass
class RemoveResult:
    content: str
    original_ratio: float
    deleted_segments: int           # 删除的段落数
    rewritten_segments: int         # 改写的段落数
    replaced_segments: int          # 仅替换名字的段落数
```

**删除策略**：

| 模式     | 行为                                    |
| -------- | --------------------------------------- |
| `hard` | 彻底删除：整段移除 + 断点桥接           |
| `soft` | 弱化存在：名字替换为代词/删除，保留情节 |

---

#### 4.6.4 结局续写 (`ending_writer.py`)

```python
class EndingWriter:
    def __init__(self, llm: LLMProvider, segment_index: SegmentIndex):
        self.llm = llm
        self.segment_index = segment_index

    def continue_story(self, text: str, context: StoryContext,
                       hint: str = "") -> str:
        """续写结局

        利用 context 信息：
        - 保持人物性格一致性
        - 延续已建立的人物关系
        - 呼应前文关键事件
        - 借助 SegmentIndex 识别最后几个场景及其节奏
        """
        ...
```

---

## 5. CLI 设计

### 5.1 命令结构

```bash
story <command> [options] <input_file>
```

### 5.2 命令列表

#### 内容分析（推荐首先执行）

```bash
story analyze novel.txt -o analysis.json
```

| 参数             | 说明               | 默认值                    |
| ---------------- | ------------------ | ------------------------- |
| `-o, --output` | 分析结果输出文件   | `{input}_analysis.json` |
| `--format`     | 输出格式 json/yaml | json                      |

**输出内容**：人物列表、人物关系、关键事件、性格分析

#### 情节压缩

```bash
story compress novel.txt -o output.txt --level medium
```

| 参数             | 说明                        | 默认值                     |
| ---------------- | --------------------------- | -------------------------- |
| `-o, --output` | 输出文件                    | `{input}_compressed.txt` |
| `--level`      | 压缩级别 light/medium/heavy | medium                     |

#### 人物筛选

```bash
story filter novel.txt --characters "张三,李四" --mode soft -o output.txt
```

| 参数                 | 说明                     | 默认值 |
| -------------------- | ------------------------ | ------ |
| `-c, --characters` | 要保留的人物（逗号分隔） | 必填   |
| `--mode`           | strict/soft              | soft   |

#### 人物删除

```bash
story remove novel.txt --characters "王五" --mode hard -o output.txt
```

| 参数                 | 说明         | 默认值 |
| -------------------- | ------------ | ------ |
| `-c, --characters` | 要删除的人物 | 必填   |
| `--mode`           | hard/soft    | hard   |

#### 结局续写

```bash
story continue novel.txt --hint "希望是HE" -o output.txt
```

| 参数       | 说明         | 默认值 |
| ---------- | ------------ | ------ |
| `--hint` | 结局期望提示 | 空     |

### 5.3 全局选项

```bash
story --config config.yaml <command> ...  # 指定配置文件
story --model qwen2.5:7b-instruct ...     # 指定模型
story --verbose ...                        # 详细输出
```

### 5.4 缓存相关选项

所有处理命令（compress/filter/remove/continue）都支持以下缓存选项，并会自动读取同一缓存目录下的 `context.json + segments.json + index.json`：

```bash
# 自动使用缓存（默认行为）
story compress novel.txt -o output.txt

# 指定已有的分析文件（可选地附带 segments）
story compress novel.txt --context analysis.json --segments segments.json -o output.txt

# 强制重新分析，不使用缓存
story compress novel.txt --no-cache -o output.txt

# 重新分析并更新缓存
story compress novel.txt --reanalyze -o output.txt

# 清空所有缓存
story cache clear

# 查看缓存状态
story cache status
```

| 参数            | 说明                                                | 默认值       |
| --------------- | --------------------------------------------------- | ------------ |
| `--context`   | 指定分析结果文件路径                                | 自动查找缓存 |
| `--segments`  | 指定 SegmentIndex JSON（若与 context 不在同一目录） | 自动查找缓存 |
| `--no-cache`  | 不使用也不保存缓存                                  | false        |
| `--reanalyze` | 强制重新分析并更新缓存                              | false        |

---

## 6. 配置设计

### 6.1 配置文件格式 (`config.yaml`)

```yaml
# LLM 配置
llm:
  provider: ollama              # ollama | openai
  model: qwen2.5:7b-instruct
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 4096
  seed: 42                      # 固定种子保证可复现

# 文本处理配置
parser:
  chunk_size: 4000              # 每块最大字符数
  overlap: 200                  # 块间重叠字符数

# 压缩配置
compress:
  default_level: medium
  levels:
    light: 0.8                  # 保留 80%
    medium: 0.5                 # 保留 50%
    heavy: 0.3                  # 保留 30%

# 输出配置
output:
  add_ai_marker: true           # 是否添加 AI 处理标记
  marker_text: "【本文经 AI 处理】"

# 缓存配置
cache:
  enabled: true                 # 是否启用自动缓存
  dir: .story_cache             # 缓存目录
  auto_save: true               # 分析后自动保存
```

### 6.2 配置加载优先级

1. 命令行参数（最高）
2. 环境变量 `STORY_*`
3. 配置文件 `config.yaml`
4. 默认值

---

## 7. 数据流

### 7.1 内容分析流程（首次处理）

```
输入文件 (txt/md)
      │
      ▼
┌─────────────┐
│  计算文件哈希 │  ← 用于缓存键
└─────┬───────┘
      │
      ▼
┌─────────────────────┐
│  检查缓存是否存在     │
│  ├─ 命中 → 直接返回   │
│  └─ 未命中 → 继续     │
└─────────┬───────────┘
          │
          ▼
┌─────────────┐
│  文本分块    │
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────┐
│           内容分析                   │
│  ┌───────────┐  ┌───────────────┐   │
│  │ 人物提取   │  │ 关系映射      │   │
│  └───────────┘  └───────────────┘   │
│  ┌───────────┐  ┌───────────────┐   │
│  │ 事件提取   │  │ 性格分析      │   │
│  └───────────┘  └───────────────┘   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────┐
│  StoryContext 对象   │
└─────────┬───────────┘
          │
          ├──────────────────┐
          │                  │
          ▼                  ▼
┌─────────────────┐  ┌──────────────────┐
│ 保存到缓存目录   │  │ 输出 JSON 文件   │
│ .story_cache/   │  │ (若指定 -o)      │
└─────────────────┘  └──────────────────┘
```

### 7.2 业务处理流程（压缩/筛选/删除/续写）

```
输入文件 (txt/md)
      │
      ▼
┌─────────────────────────────────────────┐
│   获取 StoryContext + SegmentIndex      │
│  ┌─────────────────────────────────┐    │
│  │ 1. --context/--segments 指定？   │    │
│  │ 2. 缓存命中？ → 直接加载         │    │
│  │ 3. 都没有？ → 执行分析+索引      │    │
│  └─────────────────────────────────┘    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────┐
│  文本分块    │
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────┐
│  逐块处理（带上下文）                 │
│  - 利用人物信息判断重要性             │
│  - 利用关系信息处理关联内容           │
│  - 利用事件信息保护关键情节           │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────┐
│  结果合并    │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  添加标记    │  可选：AI 处理声明
└─────┬───────┘
      │
      ▼
输出文件
```

### 7.3 错误处理

- LLM 调用失败：重试 3 次，记录日志
- 文件读取失败：明确错误提示
- 内存不足：提示减小 chunk_size

---

## 8. 依赖清单

```toml
# pyproject.toml
[project]
name = "story-for-you"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.9.0",
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "rich>=13.0.0",       # 美化 CLI 输出
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
]

[project.scripts]
story = "story_for_you.cli.main:app"
```

---

## 9. 开发环境准备

### 9.1 前置条件

1. Python 3.12+
2. Ollama 已安装并运行
3. Qwen2.5:7b-instruct 模型已下载

### 9.2 安装步骤

```bash
# 1. 安装 Ollama (macOS)
brew install ollama

# 2. 启动 Ollama 服务
ollama serve

# 3. 下载模型
ollama pull qwen2.5:7b-instruct

# 4. 克隆项目
git clone <repo>
cd story_for_you

# 5. 安装依赖
uv sync --dev

# 6. 验证安装
uv run story --help
```

---

## 10. 后续扩展点

- [ ] 支持 OpenAI API 兼容接口
- [ ] Web UI（v0.2）
- [ ] 进度条显示
- [ ] 分析结果可视化（人物关系图）
- [ ] 支持更多输入格式（epub、mobi）

---

## 11. LLM 推理策略（Ollama + Qwen2.5）

### 11.1 Provider 生命周期

- CLI 入口通过 `_build_llm(settings)` 构造单例 `OllamaProvider`，再把同一个实例注入 `StoryAnalyzer` 与四个核心业务，确保一次命令只建立一个 HTTP client，避免本地 Ollama 端口被短时间打爆。
- `Settings.llm` 提供 `provider/model/base_url/temperature/max_tokens/seed`，默认指向 `http://localhost:11434` 与 `qwen2.5:7b-instruct`。所有命令都尊重同一套配置，因此切换模型只需修改 `config.yaml` 或 `STORY_LLM__MODEL` 环境变量。
- `LLMProvider` 抽象层允许在测试中注入 Fake，实现如下最小协议即可：

```python
class FakeLLM(LLMProvider):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        return LLMResponse(content=self.mapping.get(prompt, ""), tokens_used=0)

    def generate_stream(self, prompt: str, system: str = ""):
        yield from []
```

### 11.2 Prompt 组装套路

所有抽取与业务 Prompt 遵循统一骨架，便于替换底模：

1. **System 层**：描述抽取/处理目标（位于 `analysis/prompt_templates/` 或各业务模块旁）。
2. **上下文层**：`StoryContext.for_prompt()` 输出的四段（世界观 → 人物 → 当前剧情 → 最近章节）。
3. **任务层**：描述当前步骤（如“列出事件”“压缩以下段落”）。
4. **输出约束**：强制 JSON/YAML schema，避免模型自由发挥。

示意：

```python
from story_for_you.utils.json_utils import load_json_response

system = templates.load("event_extraction")
context_block = "\n".join(context.for_prompt().values())
task = EVENT_TASK_TEMPLATE.format(chapter_no=chapter_no, text=chapter_text)
prompt = f"{context_block}\n\n{task}"
response = llm.generate(prompt=prompt, system=system)
payload = load_json_response(response.content)
```

### 11.3 资源与容错策略

- `OllamaProvider` 默认超时 120 秒，若 Qwen 在本地推理较慢可通过构造函数调大该值（后续会暴露到配置）。
- `settings.llm.temperature`、`seed` 直接透传到 Prompt 模板，用于固定输出（分析阶段设 0.2～0.3，续写阶段可升到 0.7）。
- 对于结构化输出（analysis、filter、remove），使用同步 `generate`，确保拿到完整 JSON；续写等长文本可切换到 `generate_stream` 以逐步落盘（v0.2 计划）。
- 所有命令在捕获 `RuntimeError("Ollama request failed")` 时会立即退出，让用户检查本地服务；未来计划在 `LLMProvider` 层加入指数退避重试。

### 11.4 扩展其他 Provider

1. 新建 `story_for_you/llm/<provider>.py`，继承 `LLMProvider` 并实现 `generate/generate_stream`。
2. 在 `_build_llm` 中按 `settings.llm.provider` 分支返回对应实现。
3. 若需要额外配置（API key、proxy 等），把字段加入 `LLMSettings` 并同步 `config.example.yaml`、文档。
4. 测试：使用 Pytest fixture 注入新的 provider fake，覆盖 CLI 命令的 happy path。

---

## 12. 调试与可观测性

### 12.1 CLI 诊断开关

- `story <cmd> --config path/to/config.yaml`：快速切换不同模型或 chunk 策略。
- `--context` / `--segments`：复用既有分析输出，定位问题时不用重复跑 LLM。
- `--no-cache`：排查缓存污染；`--reanalyze`：强制刷新缓存后再执行业务。
- `story cache clear` / `story cache status`：对应 `.story_cache/`，可立即清除异常条目或读取缓存规模。

### 12.2 Prompt & 缓存审计

- `analysis.json` 中的 `metadata`（`model/_fingerprint/window_size`）是复现关键，提交 issue 时需附带。
- `segments.json` / `index.json` 搭配 `SegmentIndexService` 可以复盘一次筛选/删除为何拿到特定段落。
- 若怀疑上下文不一致，可使用 `StoryContext.to_dict()` + `jq` 检查 `characters[*].unresolved` 是否按预期更新。

### 12.3 常见故障排查

| 症状 | 可能原因 | 排查步骤 |
| ---- | -------- | -------- |
| `RuntimeError: Ollama request failed` | 本地 `ollama serve` 未启动或模型未下载 | `curl http://localhost:11434/api/tags` 验证服务；`ollama pull qwen2.5:7b-instruct` |
| 结果不复现 | 使用旧缓存或变更了配置 | 加 `--reanalyze`，确认 `StoryContext.metadata.config_fingerprint` 更新 |
| 筛选结果空白 | 角色别名未登记 | 检查 `StoryContext.characters[name].aliases`，必要时在分析阶段补充 |

---

## 13. 测试策略

### 13.1 单元测试

- `analysis/layers`：验证窗口、事件账本、状态存储的增删查；利用轻量 dataclass 构造输入，确保窗口大小、事件索引正确。
- `cache/store`：使用 `tmp_path` 建立临时缓存目录，覆盖 `save/get/clear_all` 分支。
- `indexer`：为 `SegmentSplitter/SegmentRetriever` 编写 deterministic 文本片段，断言检索结果与 gap 计算。

### 13.2 LLM 相关测试

- 通过前述 `FakeLLMProvider` 固定输出，覆盖 `CharacterExtractor` 等需要 JSON 解析的逻辑，避免在 CI 中调用真实模型。
- 对核心命令（`compress/filter/remove/continue`）使用 Typer 的 `CliRunner`，注入 fake context/segments，验证 CLI wiring 与文件输出。

### 13.3 基准数据与夹具

- `tests/fixtures/`（待建立）存放 2～3 段短篇文本，涵盖人物冲突、关系变化、反转，以驱动所有模块的回归测试。
- 针对缓存逻辑，添加 `sample_context.json` 与 `segments.json`，便于在测试中快速构造 `CachedArtifacts`。

### 13.4 覆盖率目标

- `pytest --cov=story_for_you --maxfail=1`：分析、核心、LLM 三个子包需保持 ≥85%。
- PR 前必须在本地运行 `pytest -q` + `ruff check .`，CLi 输出附于 PR 描述，确保 reviewer 能够复现。
