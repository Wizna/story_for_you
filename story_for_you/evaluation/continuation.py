from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from story_for_you.utils.file_io import read_text_file

_SCORES = ("motivation", "knowledge_boundary", "plot_progress", "style")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _new_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def _case_list(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation cases must be a non-empty JSON array.")
    seen = set()
    for case in payload:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("Each evaluation case requires a string ID.")
        name = case["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) or name in seen:
            raise ValueError(
                "Case IDs must be unique and contain only letters, digits, hyphens and underscores."
            )
        seen.add(name)
    return payload


def prepare_cases(manifest: Path, output_dir: Path) -> Path:
    cases = _case_list(read_json(manifest))
    prepared = []
    # Validate every cutoff before creating any output files.
    prefixes = []
    for case in cases:
        if not isinstance(case.get("source"), str) or not isinstance(
            case.get("hint", ""), str
        ):
            raise ValueError(
                "Each case needs a source path and an optional string hint."
            )
        source = (manifest.parent / case["source"]).resolve()
        text = read_text_file(source)
        cutoff = case.get("cutoff_chars")
        if (
            not isinstance(cutoff, int)
            or isinstance(cutoff, bool)
            or not 0 < cutoff < len(text)
        ):
            raise ValueError(
                "cutoff_chars must leave a non-empty prefix and an excluded future suffix."
            )
        units = case.get("max_units", 3)
        if not isinstance(units, int) or isinstance(units, bool) or units < 1:
            raise ValueError("max_units must be positive.")
        prefixes.append(text[:cutoff])
    root = _new_directory(output_dir)
    for case, prefix in zip(cases, prefixes):
        target = root / f"{case['id']}.prefix.txt"
        target.write_text(prefix, encoding="utf-8")
        prepared.append(
            {
                "id": case["id"],
                "prefix": target.name,
                "prefix_sha256": digest(prefix),
                "hint": case.get("hint", ""),
                "max_units": case.get("max_units", 3),
            }
        )
    path = root / "cases.json"
    write_json(path, prepared)
    return path


def load_prepared(path: Path) -> list[dict[str, Any]]:
    cases = _case_list(read_json(path))
    for case in cases:
        if not isinstance(case.get("prefix"), str):
            raise ValueError("Prepared case requires a prefix path.")
        prefix = (path.parent / case["prefix"]).read_text(encoding="utf-8")
        if digest(prefix) != case.get("prefix_sha256"):
            raise ValueError(f"Prefix hash mismatch: {case['id']}")
        if not isinstance(case.get("hint"), str):
            raise ValueError("Prepared hint must be a string.")
        count = case.get("max_units")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Prepared max_units must be positive.")
    return cases


def generate_cases(
    prepared: Path,
    output_dir: Path,
    generate: Callable[[str, str, int], str],
    *,
    model: str = "unknown",
) -> Path:
    cases = load_prepared(prepared)
    root = _new_directory(output_dir)
    results = []
    for case in cases:
        prefix = (prepared.parent / case["prefix"]).read_text(encoding="utf-8")
        if digest(prefix) != case["prefix_sha256"]:
            raise ValueError("Prefix changed after validation.")
        text = generate(prefix, case["hint"], case["max_units"])
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Evaluation generation returned an empty continuation.")
        target = root / f"{case['id']}.txt"
        target.write_text(text, encoding="utf-8")
        results.append(
            {
                "id": case["id"],
                "prefix_sha256": case["prefix_sha256"],
                "output": target.name,
                "output_sha256": digest(text),
                "model": model,
                "hint": case["hint"],
                "max_units": case["max_units"],
            }
        )
    path = root / "results.json"
    write_json(path, results)
    return path


def prepare_blind_comparison(
    prepared: Path,
    baseline: Path,
    candidate: Path,
    output_dir: Path,
    *,
    seed: int | None = None,
) -> Path:
    cases = load_prepared(prepared)
    baseline_cases = {case["id"]: case for case in _case_list(read_json(baseline))}
    candidate_cases = {case["id"]: case for case in _case_list(read_json(candidate))}
    expected = {case["id"] for case in cases}
    if set(baseline_cases) != expected or set(candidate_cases) != expected:
        raise ValueError("Both runs must contain exactly the prepared cases.")
    texts = {}
    for case in cases:
        pair = {}
        for label, manifest, records in (
            ("baseline", baseline, baseline_cases),
            ("candidate", candidate, candidate_cases),
        ):
            record = records[case["id"]]
            if any(
                record.get(key) != case.get(key)
                for key in ("prefix_sha256", "hint", "max_units")
            ):
                raise ValueError(
                    "Runs must use matching prefixes, hints and continuation limits."
                )
            text = (manifest.parent / record["output"]).read_text(encoding="utf-8")
            if digest(text) != record.get("output_sha256"):
                raise ValueError("Run output hash mismatch.")
            pair[label] = text
        texts[case["id"]] = pair
    root = _new_directory(output_dir)
    blind = root / "blind"
    blind.mkdir()
    rng = random.Random(seed)
    key = {}
    ratings = []
    for case in cases:
        pair = ["baseline", "candidate"]
        rng.shuffle(pair)
        name = case["id"]
        prefix = (prepared.parent / case["prefix"]).read_text(encoding="utf-8")
        folder = blind / name
        folder.mkdir()
        (folder / "prefix.txt").write_text(prefix, encoding="utf-8")
        (folder / "hint.txt").write_text(case["hint"], encoding="utf-8")
        key[name] = {}
        for letter, label in zip(("A", "B"), pair):
            text = texts[name][label]
            (folder / f"{letter}.txt").write_text(text, encoding="utf-8")
            key[name][letter] = {
                "label": label,
                "chars": len(text),
                "sha256": digest(text),
            }
        ratings.append(
            {
                "id": name,
                "winner": None,
                "A": {
                    "contradictions": None,
                    **{key: None for key in _SCORES},
                    "evidence": [],
                },
                "B": {
                    "contradictions": None,
                    **{key: None for key in _SCORES},
                    "evidence": [],
                },
            }
        )
    write_json(root / "private_key.json", key)
    write_json(blind / "ratings.json", ratings)
    (blind / "instructions.txt").write_text(
        "仅根据 prefix.txt 与 hint.txt 比较 A/B；不得读取原作后文或 private_key.json。\n"
        "winner 填 A/B/tie；contradictions 为确认的事实矛盾数量；其他四项按 1-5 分评分，越高越好。\n"
        "motivation=人物动机；knowledge_boundary=知情范围；plot_progress=情节推进；style=原作文风。\n"
        "evidence 列出对应原文/续写引文及问题说明；任何确认矛盾必须有引文证据。\n",
        encoding="utf-8",
    )
    return blind


def summarize_ratings(ratings_path: Path, key_path: Path) -> dict[str, Any]:
    ratings = _case_list(read_json(ratings_path))
    key = read_json(key_path)
    if not isinstance(key, dict) or {case["id"] for case in ratings} != set(key):
        raise ValueError("Ratings must cover every blinded case exactly once.")
    totals = {
        label: {
            "wins": 0,
            "contradictions": 0,
            "chars": 0,
            **{score: 0 for score in _SCORES},
        }
        for label in ("baseline", "candidate")
    }
    ties = 0
    for case in ratings:
        mapping = key[case["id"]]
        if not isinstance(mapping, dict) or set(mapping) != {"A", "B"}:
            raise ValueError("Each private key entry requires A and B mappings.")
        if {mapping[letter].get("label") for letter in ("A", "B")} != {
            "baseline",
            "candidate",
        }:
            raise ValueError("Private key must map to one baseline and one candidate.")
        if any(
            not isinstance(mapping[letter].get("chars"), int)
            or isinstance(mapping[letter]["chars"], bool)
            or mapping[letter]["chars"] <= 0
            for letter in ("A", "B")
        ):
            raise ValueError("Private key requires positive character counts.")
        if not isinstance(case.get("winner"), str) or case["winner"] not in {
            "A",
            "B",
            "tie",
        }:
            raise ValueError("Every case needs winner A, B or tie.")
        if case["winner"] == "tie":
            ties += 1
        else:
            totals[mapping[case["winner"]]["label"]]["wins"] += 1
        for letter in ("A", "B"):
            data = case.get(letter)
            if not isinstance(data, dict):
                raise ValueError("Both A and B require scores.")
            count = data.get("contradictions")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Contradictions must be a non-negative integer.")
            if not isinstance(data.get("evidence"), list) or any(
                not isinstance(item, str) or not item.strip()
                for item in data["evidence"]
            ):
                raise ValueError("Evidence must be a list of non-empty strings.")
            if count and not data["evidence"]:
                raise ValueError("Contradictions require quoted evidence.")
            stats = totals[mapping[letter]["label"]]
            stats["contradictions"] += count
            stats["chars"] += mapping[letter]["chars"]
            for score in _SCORES:
                value = data.get(score)
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 1 <= value <= 5
                ):
                    raise ValueError(f"{score} must be an integer from 1 to 5.")
                stats[score] += value
    for stats in totals.values():
        stats["contradictions_per_10k_chars"] = (
            10000 * stats["contradictions"] / max(1, stats["chars"])
        )
        stats["mean_scores"] = {
            score: stats.pop(score) / len(ratings) for score in _SCORES
        }
    return {
        "cases": len(ratings),
        "ties": ties,
        "results": totals,
        "interpretation": "Descriptive blind ratings; no statistical significance or general quality gain is established.",
    }
