"""Chinese name processing utilities.

Provides functions for parsing and splitting compound Chinese names,
particularly those with ranking suffixes, family terms, or role titles.
"""

from __future__ import annotations

__all__ = [
    "RANKING_SUFFIXES",
    "FAMILY_TERMS",
    "ROLE_SUFFIXES",
    "split_compound_chinese_name",
]

# Common ranking/title suffixes for splitting compound names
RANKING_SUFFIXES = ("大老", "二老", "三老", "四老", "大佬", "二佬", "三佬")

# Family relationship terms
FAMILY_TERMS = ("祖父", "爷爷", "外公", "外婆", "奶奶", "姥姥", "姥爷", "父亲", "母亲", "爹", "娘")

# Occupation/role suffixes
ROLE_SUFFIXES = ("船夫", "马兵", "乡绅", "老爷", "夫人", "小姐", "公子", "先生")


def split_compound_chinese_name(name: str) -> list[str]:
    """拆分复合中文名为组成部分。

    例如：
    - 傩送二老 → [傩送, 二老]
    - 天保大老 → [天保, 大老]
    - 老船夫 → [船夫]
    - 祖父（老船夫）→ [祖父, 老船夫]

    Args:
        name: The compound Chinese name to split.

    Returns:
        A list of component parts extracted from the name, or empty list if
        the name is too short or doesn't contain recognizable patterns.
    """
    if not name or len(name) < 3:
        return []

    parts: list[str] = []

    # 检查是否以排行后缀结尾 (如 傩送二老)
    for suffix in RANKING_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            prefix = name[: -len(suffix)]
            if prefix:
                parts.append(prefix)
                parts.append(suffix)
            break

    # 检查是否以"老"字开头 + 职业/身份 (如 老船夫)
    if name.startswith("老") and len(name) >= 3:
        suffix_part = name[1:]  # 去掉"老"
        if suffix_part:
            parts.append(suffix_part)

    # 检查是否包含家庭称呼 (如 祖父、爷爷)
    for term in FAMILY_TERMS:
        if term in name and name != term:
            parts.append(term)
            # 如果不是以该称呼结尾，也提取前缀
            idx = name.find(term)
            if idx > 0:
                parts.append(name[:idx])

    # 检查是否以职业后缀结尾 (如 杨马兵)
    for suffix in ROLE_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            prefix = name[: -len(suffix)]
            # 要求前缀至少2个字符，避免太短的通用词如"老"
            if prefix and len(prefix) >= 2:
                parts.append(prefix)
            break

    return parts
