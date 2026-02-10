"""Chinese name processing utilities.

Provides functions for parsing and splitting compound Chinese names,
particularly those with honorific prefixes/suffixes, family terms, or titles.
"""

from __future__ import annotations

import re

__all__ = [
    "HONORIFIC_PREFIXES",
    "HONORIFIC_SUFFIXES",
    "ROLE_PRIORITY",
    "names_have_overlap",
    "split_compound_chinese_name",
]

ROLE_PRIORITY: dict[str, int] = {"main": 3, "support": 2, "minor": 1}


def _normalize_name(name: str) -> str:
    """统一名字格式：去除空白并转小写。"""
    return re.sub(r"\s+", "", name).lower()


def names_have_overlap(names1: list[str], names2: list[str]) -> bool:
    """检测两组名字是否有实质性重叠（子串匹配）。

    对于中文名，较短名字是较长名字的子串且长度 >= 2，视为匹配。
    例如：傩送 ⊂ 傩送二老 → 匹配
    """
    for n1 in names1:
        n1_clean = _normalize_name(n1)
        if len(n1_clean) < 2:
            continue
        for n2 in names2:
            n2_clean = _normalize_name(n2)
            if len(n2_clean) < 2:
                continue
            shorter, longer = (
                (n1_clean, n2_clean)
                if len(n1_clean) <= len(n2_clean)
                else (n2_clean, n1_clean)
            )
            if shorter in longer:
                if len(shorter) >= len(longer) * 0.5 or len(longer) <= 4:
                    return True
    return False

# 通用称谓前缀（适用于各类网文）
HONORIFIC_PREFIXES = ("老", "小", "阿", "大", "二", "三")

# 通用称谓后缀（适用于各类网文）
# 按长度降序排列，确保先匹配长后缀
HONORIFIC_SUFFIXES = (
    # 三字称谓（必须先匹配）
    "大少爷",
    "二少爷",
    "三少爷",
    "大小姐",
    "二小姐",
    "三小姐",
    # 双字尊称/家族称呼
    "爷爷",
    "奶奶",
    "姥爷",
    "姥姥",
    "外公",
    "外婆",
    "祖父",
    "祖母",
    "父亲",
    "母亲",
    "爹爹",
    "娘亲",
    "大哥",
    "二哥",
    "三哥",
    "大姐",
    "二姐",
    "三姐",
    # 双字社交称谓
    "先生",
    "女士",
    "小姐",
    "夫人",
    "老爷",
    "公子",
    "少爷",
    "少主",
    "大人",
    "阁下",
    "前辈",
    "师傅",
    "师父",
    "徒弟",
    "师兄",
    "师姐",
    "师弟",
    "师妹",
    # 修仙/玄幻类
    "道长",
    "真人",
    "仙子",
    "仙尊",
    "魔君",
    "魔尊",
    "圣女",
    "圣子",
    "长老",
    "太上",
    "祖师",
    "宗主",
    "掌教",
    "护法",
    "使者",
    "执事",
    # 武侠类
    "大侠",
    "女侠",
    "少侠",
    "掌门",
    "帮主",
    "堂主",
    "盟主",
    "庄主",
    "教主",
    "门主",
    "谷主",
    "岛主",
    "城主",
    # 现代职场类
    "总裁",
    "董事",
    "经理",
    "主任",
    "主管",
    "组长",
    "队长",
    "教授",
    "博士",
    "医生",
    "律师",
    "警官",
    "局长",
    "处长",
    "科长",
    # 单字家族称呼
    "爷",
    "奶",
    "爸",
    "妈",
    "叔",
    "婶",
    "姑",
    "姨",
    "舅",
    "哥",
    "姐",
    "弟",
    "妹",
    "爹",
    "娘",
)


def split_compound_chinese_name(name: str) -> list[str]:
    """拆分复合中文名为组成部分（通用版本）。

    核心规则：
    1. 前缀模式：老王 → [王], 小李 → [李], 阿强 → [强]
    2. 后缀模式：王先生 → [王], 李大人 → [李]
    3. 称呼提取：张爷爷 → [张, 爷爷]

    例如：
    - 老王 → [王]
    - 小李 → [李]
    - 阿强 → [强]
    - 张爷爷 → [张, 爷爷]
    - 王先生 → [王]
    - 李大人 → [李]
    - 赵大少爷 → [赵, 大少爷]
    - 林师兄 → [林, 师兄]
    - 清虚道长 → [清虚, 道长]
    - 紫霞仙子 → [紫霞, 仙子]

    Args:
        name: The compound Chinese name to split.

    Returns:
        A list of component parts extracted from the name, or empty list if
        the name is too short or doesn't contain recognizable patterns.
    """
    if not name or len(name) < 2:
        return []

    parts: list[str] = []

    # 规则1：检查前缀模式（老王、小李、阿强等）
    for prefix in HONORIFIC_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            suffix_part = name[len(prefix) :]
            # 确保剩余部分有意义（至少1个字符）
            if suffix_part and len(suffix_part) >= 1:
                parts.append(suffix_part)
            break

    # 规则2&3：检查后缀模式（王先生、张爷爷、赵大少爷等）
    # HONORIFIC_SUFFIXES 已按长度降序排列，先匹配长后缀
    for suffix in HONORIFIC_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            prefix_part = name[: -len(suffix)]
            if prefix_part:
                parts.append(prefix_part)
                # 对于家族称呼和职位称谓，也保留后缀以便建立别名关联
                parts.append(suffix)
            break

    return parts
