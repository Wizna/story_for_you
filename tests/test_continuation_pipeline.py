import json

from story_for_you.analysis.context import CharacterState, StoryContext
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.indexer.segment import SegmentIndex
from story_for_you.llm.base import LLMProvider, LLMResponse
from tests.test_cacheable_prompting import _RecordingLLM


class PipelineLLM(LLMProvider):
    def __init__(self):
        self.draft_prompts = []
        self.unit = 0
        self.texts = [
            "甲把宝剑交给乙，独自进入客栈。",
            "天色渐暗。" * 1800,
            "乙拔剑守在门外。",
        ]

    def generate(self, prompt, system="", options=None):
        step = options.get("_sfy_step")
        if step == ": interpret user hint":
            payload = {
                "normalized_text": "保持展开",
                "closure": "open",
                "focus_characters": ["甲"],
            }
        elif step == ": continuation plan":
            payload = {
                "unit_count": 3,
                "rationale": "依次交剑、等待、守门",
                "continuation_intent": "推进冲突",
                "narrative_horizon": "当夜",
                "closure_level": "ongoing",
                "unit_type": "章",
                "chapters": [
                    {
                        "title": f"单元{i}",
                        "purpose": "推进",
                        "viewpoint": "甲",
                        "setting": "城中",
                        "beats": ["行动", "后果"],
                        "focus_characters": ["甲", "乙"],
                        "end_state": "守门的结果留待后续",
                    }
                    for i in range(1, 4)
                ],
            }
        elif step == ": plan unit scenes":
            payload = {
                "scenes": [
                    {
                        key: "承接当前场景"
                        for key in (
                            "viewpoint",
                            "setting",
                            "goal",
                            "obstacle",
                            "action",
                            "turn",
                            "cost",
                            "end_state",
                            "knowledge_boundary",
                        )
                    }
                ]
            }
        elif step == ": draft":
            self.draft_prompts.append(prompt.render())
            self.unit += 1
            return LLMResponse(self.texts[self.unit - 1], 0)
        elif step == ": polish":
            return LLMResponse(self.texts[self.unit - 1], 0)
        elif step == ": review and remember unit":
            payload = {
                "passed": True,
                "issues": [],
                "repair_instructions": [],
                "summary": "宝剑交给乙，甲留在客栈"
                if self.unit == 1
                else "当前场景继续发展",
                "characters": [],
                "events": [],
                "story_state": None,
            }
            if self.unit == 1:
                payload["characters"] = [
                    {
                        "name": "甲",
                        "aliases": [],
                        "role": "main",
                        "realm": None,
                        "personality": [],
                        "unresolved": [],
                        "status": "alive",
                        "location": "客栈",
                        "goal": None,
                        "knowledge": ["宝剑归乙持有"],
                        "evidence": self.texts[0],
                    }
                ]
                payload["events"] = [
                    {
                        "event_id": "new",
                        "chapter": 0,
                        "type": "progress",
                        "participants": ["甲", "乙"],
                        "summary": "宝剑现归乙持有",
                        "impact": {
                            "power_shifts": {},
                            "relation_changes": {},
                            "world_flags": [],
                        },
                        "is_irreversible": False,
                        "evidence": self.texts[0],
                    }
                ]
        elif step == ": validate continuation":
            payload = {"passed": True, "issues": [], "repair_instructions": []}
        else:
            raise AssertionError(f"Unexpected pipeline step: {step}")
        return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)

    def generate_stream(self, *args, **kwargs):
        yield from ()


def test_three_unit_pipeline_retains_facts_beyond_raw_tail_and_shares_future_plan():
    llm = PipelineLLM()
    context = StoryContext(
        characters={"甲": CharacterState(name="甲", role="main", location="城门")}
    )
    writer = EndingWriter(llm, SegmentIndex([], {}, {}, {}))
    result = writer.continue_story("甲在城门外握着宝剑。", context)
    third_prompt = llm.draft_prompts[2]
    raw_tail = third_prompt.split("## 已生成的前文（只用于承接，不要重复）", 1)[
        1
    ].split("# 续写初稿", 1)[0]
    assert "甲把宝剑交给乙，独自进入客栈。" not in raw_tail
    assert "原文证据: 甲把宝剑交给乙，独自进入客栈。" in third_prompt
    assert "宝剑现归乙持有" in third_prompt
    assert "location=客栈" in third_prompt
    assert "守门的结果留待后续" in llm.draft_prompts[0]
    assert "本单元场景计划" in third_prompt
    assert "第3章" in result
    assert context.characters["甲"].location == "城门"


def test_failed_unit_is_repaired_and_reviewed_before_next_unit_sees_its_state():
    class RepairPipeline(PipelineLLM):
        failed_once = False

        def generate(self, prompt, system="", options=None):
            step = options.get("_sfy_step")
            if (
                step == ": review and remember unit"
                and self.unit == 1
                and not self.failed_once
            ):
                self.failed_once = True
                return LLMResponse(
                    json.dumps(
                        {
                            "passed": False,
                            "issues": ["场景过渡不清楚"],
                            "repair_instructions": ["补足走进客栈的动作"],
                        }
                    ),
                    0,
                )
            if step == ": final repair":
                previous = self.texts[0]
                self.texts[0] = "甲将宝剑交给乙，随后走进客栈。"
                return LLMResponse(
                    json.dumps(
                        {
                            "edits": [
                                {
                                    "old_text": previous,
                                    "new_text": self.texts[0],
                                    "reason": "补足动作过渡",
                                }
                            ]
                        }
                    ),
                    0,
                )
            if step == ": review and remember unit" and self.unit == 1:
                assert "location=城门" in prompt.prefix
            return super().generate(prompt, system, options)

    llm = RepairPipeline()
    context = StoryContext(
        characters={"甲": CharacterState(name="甲", role="main", location="城门")}
    )
    result = EndingWriter(llm, SegmentIndex([], {}, {}, {})).continue_story(
        "甲在城门。", context
    )
    assert "随后走进客栈" in result
    assert "location=客栈" in llm.draft_prompts[1]


def test_final_validation_and_repair_include_all_planned_cast_source_evidence():
    class EvidencePipeline(PipelineLLM):
        def __init__(self):
            super().__init__()
            self.validations = []
            self.repairs = []

        def generate(self, prompt, system="", options=None):
            step = options["_sfy_step"]
            if step == ": validate continuation":
                self.validations.append(prompt.prefix)
                failed = len(self.validations) == 1
                return LLMResponse(json.dumps({
                    "passed": not failed,
                    "issues": ["动作需要澄清"] if failed else [],
                    "repair_instructions": ["调整持剑动作"] if failed else [],
                }), 0)
            if step == ": final repair":
                self.repairs.append(prompt.prefix)
                return LLMResponse(json.dumps({"edits": [{
                    "old_text": "乙拔剑守在门外。", "new_text": "乙持剑守在门外。", "reason": "澄清动作",
                }]}), 0)
            response = super().generate(prompt, system, options)
            if step == ": continuation plan":
                payload = json.loads(response.content)
                for chapter in payload["chapters"]:
                    chapter["focus_characters"].append("g")
                return LLMResponse(json.dumps(payload), 0)
            return response

    chars = {name: CharacterState(name, role="main") for name in "abcdefg"}
    chars["g"].knowledge = ["只有 g 知道的密钥 G_SECRET"]
    chars["甲"] = CharacterState("甲", location="城门")
    llm = EvidencePipeline()
    EndingWriter(llm, SegmentIndex([], {}, {}, {})).continue_story(
        "故事仍在继续。", StoryContext(characters=chars),
    )
    assert all("G_SECRET" in prompt for prompt in [*llm.validations, *llm.repairs])
    assert '"location": "城门"' in llm.validations[0]
    assert '"location": "客栈"' in llm.validations[0]


def test_scene_plan_repairs_a_missing_field_in_a_later_scene_once():
    fields = ("viewpoint", "setting", "goal", "obstacle", "action", "turn", "cost", "end_state", "knowledge_boundary")
    first = dict.fromkeys(fields, "有原文依据的安排")
    second = {key: value for key, value in first.items() if key != "obstacle"}
    llm = _RecordingLLM([json.dumps({"scenes": [first, second]}), json.dumps({"scenes": [first, first]})])
    result = EndingWriter(llm, SegmentIndex([], {}, {}, {}))._phase_scene_plan("原作证据", "当前单元", "继续")
    assert "场景2" in result
    assert "scenes[1].obstacle" in llm.prompts[1].task
    assert llm.prompts[1].prefix == "原作证据"
