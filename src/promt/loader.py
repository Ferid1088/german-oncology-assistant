from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


PROMT_ROOT = Path(__file__).resolve().parent
AMBIGUITY_ROOT = PROMT_ROOT / "ambiguity"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_ambiguity_prompt_assets() -> dict:
    return {
        "system": _read_text(AMBIGUITY_ROOT / "system.txt"),
        "instructions": _read_text(AMBIGUITY_ROOT / "instructions.txt"),
        "output_schema": _read_json(AMBIGUITY_ROOT / "output_schema.json"),
        "few_shot_examples": _read_json(AMBIGUITY_ROOT / "few_shot_examples.json"),
    }


def _format_examples(examples: list[dict]) -> str:
    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Beispiel {index}",
                    "Eingabe:",
                    str(example.get("input", "")).strip(),
                    "Ausgabe:",
                    json.dumps(example.get("output", {}), ensure_ascii=False, indent=2),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_ambiguity_prompt_messages(history_block: str, query: str) -> list[dict[str, str]]:
    assets = load_ambiguity_prompt_assets()
    user_prompt = "\n\n".join(
        [
            assets["instructions"],
            "Erwartetes JSON-Schema:",
            json.dumps(assets["output_schema"], ensure_ascii=False, indent=2),
            "Few-shot-Beispiele:",
            _format_examples(assets["few_shot_examples"]),
            "Aktueller Gesprächsverlauf:",
            history_block.strip() or "(leer)",
            "Aktuelle Anfrage:",
            query.strip(),
            "Gib AUSSCHLIESSLICH JSON zurück.",
        ]
    )
    return [
        {"role": "system", "content": assets["system"]},
        {"role": "user", "content": user_prompt},
    ]