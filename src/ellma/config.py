from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class Pricing:
    input_per_1m: float
    output_per_1m: float

@dataclass(frozen=True)
class RunConfig:
    dataset: str
    model: str
    seed: int = 12345
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    timeout_sec: int = 60
    use_json_mode: bool = True
    pricing: Optional[Pricing] = None

@dataclass(frozen=True)
class IOConfig:
    input_xlsx: Path
    outdir: Path
    run_name: str
    store_raw_text: bool = False
