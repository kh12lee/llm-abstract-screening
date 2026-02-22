from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_DIR = _REPO_ROOT / "prompts"

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

@dataclass(frozen=True)
class PromptSpec:
    dataset: str
    system_version: str = "v1"
    ontology_version: str = "v1"

    def system_path(self) -> Path:
        return _PROMPT_DIR / f"system_{self.system_version}.txt"

    def ontology_path(self) -> Path:
        key = self.dataset.strip().lower()
        if key in {"cholangio", "cholangiocarcinoma"}:
            return _PROMPT_DIR / f"cholangio_ontology_{self.ontology_version}.txt"
        if key in {"nsclc"}:
            return _PROMPT_DIR / f"nsclc_ontology_{self.ontology_version}.txt"
        raise ValueError(f"Unknown dataset: {self.dataset}")

def load_prompts(spec: PromptSpec) -> tuple[str, str]:
    system_message = _read_text(spec.system_path())
    ontology_prompt = _read_text(spec.ontology_path())
    return system_message, ontology_prompt
