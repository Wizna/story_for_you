# Story For You

Story For You 提供一组围绕长篇故事文本的 CLI/库工具，目前聚焦于：

- 内容分析：构建 `StoryContext`，提取人物、事件与长期状态；
- 核心处理：剧情压缩、人物筛选 / 删除以及结局续写；
- 统一的 Typer CLI：`story analyze|compress|filter|remove|continue`。

## 快速开始

```bash
git clone <repo>
cd story_for_you
uv sync --dev
uv run story --help
```

配置、架构和模块说明详见 `docs/dev-architecture.md` 与 `docs/specs01.md`。

3
