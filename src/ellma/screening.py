from __future__ import annotations
import json, re, time, hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from tqdm import tqdm

from .config import IOConfig, RunConfig
from .llm_client import chat_json
from .prompts import PromptSpec, load_prompts

def canonicalize(text: Any) -> str:
    if isinstance(text, str):
        s = text
    else:
        s = "" if pd.isna(text) else str(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(ln.rstrip() for ln in s.split("\n"))

def uid_from_text(title: str, abstract: str) -> str:
    s = canonicalize(title) + "\n" + canonicalize(abstract)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_json_parse(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"answer": "no", "reason": "Failed to parse JSON from model output."}

def build_user_prompt(ontology_prompt: str, title: str, abstract: str) -> str:
    return f"""{ontology_prompt}

Now, screen the following study and return ONLY a compact JSON object with keys "answer" and "reason".
- "answer" must be either "yes" or "no".
- "reason" should briefly justify the decision (1–2 sentences). If "no", state which criterion fails when possible.

Title: {title}
Abstract: {abstract}

Return JSON only, no extra text.
"""


def _strip_raw_text(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec.pop("title", None)
    rec.pop("abstract", None)
    rec.pop("raw_output", None)
    return rec

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def run_screening(
    *,
    df: pd.DataFrame,
    client: Any,
    run_cfg: RunConfig,
    io_cfg: IOConfig,
    system_version: str = "v1",
    ontology_version: str = "v1",
) -> Tuple[Path, Path, Path]:
    io_cfg.outdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = io_cfg.outdir / f"{io_cfg.run_name}.jsonl"
    out_xlsx  = io_cfg.outdir / f"{io_cfg.run_name}.xlsx"
    out_meta  = io_cfg.outdir / f"{io_cfg.run_name}.run_meta.json"

    spec = PromptSpec(dataset=run_cfg.dataset, system_version=system_version, ontology_version=ontology_version)
    system_message, ontology_prompt = load_prompts(spec)

    done = set()
    if out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    u = r.get("uid")
                    if u:
                        done.add(u)
                except json.JSONDecodeError:
                    continue

    if "title" not in df.columns or "abstract" not in df.columns:
        raise ValueError("Input spreadsheet must include columns: 'title' and 'abstract'.")

    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")

    total_prompt = 0
    total_completion = 0
    seen_models = set()
    created_vals = []

    out_fh = out_jsonl.open("a", encoding="utf-8")
    try:
        for row_index, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
            title = canonicalize(getattr(row, "title", ""))
            abstract = canonicalize(getattr(row, "abstract", ""))
            uid = uid_from_text(title, abstract)

            if uid in done:
                continue

            if abstract == "":
                rec = {
                    "row_index": row_index,
                    "uid": uid,
                    "ai_answer": "no",
                    "ai_reason": "Empty abstract.",
                    "response_time_sec": 0.0,
                    "ai_prompt_tokens": 0,
                    "ai_completion_tokens": 0,
                    "model_returned": run_cfg.model,
                    "created": None,
                }
            else:
                user_prompt = build_user_prompt(ontology_prompt, title, abstract)
                start = time.time()
                try:
                    resp = chat_json(
                        client,
                        model=run_cfg.model,
                        system_message=system_message,
                        user_message=user_prompt,
                        seed=run_cfg.seed,
                        temperature=run_cfg.temperature,
                        top_p=run_cfg.top_p,
                        use_json_mode=run_cfg.use_json_mode,
                    )
                except Exception:
                    resp = chat_json(
                        client,
                        model=run_cfg.model,
                        system_message=system_message,
                        user_message=user_prompt,
                        seed=run_cfg.seed,
                        temperature=run_cfg.temperature,
                        top_p=run_cfg.top_p,
                        use_json_mode=False,
                    )

                elapsed = round(time.time() - start, 3)
                content = resp.choices[0].message.content.strip()
                j = safe_json_parse(content)

                ans = str(j.get("answer", "")).lower().strip()
                if ans not in {"yes", "no"}:
                    ans = "no"

                rec = {
                    "row_index": row_index,
                    "uid": uid,
                    "ai_answer": ans,
                    "ai_reason": str(j.get("reason", "")).strip(),
                    "ai_score": j.get("score", None),
                    "criteria_flags": j.get("criteria_flags", None),
                    "evidence_phrases": j.get("evidence_phrases", None),
                    "response_time_sec": elapsed,
                    "ai_prompt_tokens": getattr(getattr(resp, "usage", None), "prompt_tokens", None),
                    "ai_completion_tokens": getattr(getattr(resp, "usage", None), "completion_tokens", None),
                    "model_returned": getattr(resp, "model", run_cfg.model),
                    "created": getattr(resp, "created", None),
                }

                pt = rec.get("ai_prompt_tokens") or 0
                ct = rec.get("ai_completion_tokens") or 0
                total_prompt += int(pt)
                total_completion += int(ct)
                seen_models.add(rec["model_returned"])
                if rec["created"] is not None:
                    created_vals.append(rec["created"])

            if not io_cfg.store_raw_text:
                rec = _strip_raw_text(rec)

            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            done.add(uid)
    finally:
        out_fh.close()

    # Build results table aligned to input order
    records = []
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    valid_uids = {uid_from_text(canonicalize(t), canonicalize(a)) for t, a in zip(df["title"], df["abstract"])}
    records = [r for r in records if r.get("uid") in valid_uids]
    records.sort(key=lambda r: r.get("row_index", 10**12))

    res_df = pd.DataFrame.from_records(records)
    keep_cols = [c for c in [
        "row_index","uid","ai_answer","ai_reason","ai_score",
        "response_time_sec","ai_prompt_tokens","ai_completion_tokens",
        "model_returned","created"
    ] if c in res_df.columns]
    res_df = res_df[keep_cols]
    res_df.to_excel(out_xlsx, index=False)

    est_cost = None
    if run_cfg.pricing is not None:
        est_cost = (total_prompt/1_000_000)*run_cfg.pricing.input_per_1m + (total_completion/1_000_000)*run_cfg.pricing.output_per_1m

    run_meta = {
        "dataset": run_cfg.dataset,
        "requested_model": run_cfg.model,
        "all_models_seen": sorted(seen_models) if seen_models else [run_cfg.model],
        "mixed_models_detected": (len(seen_models) > 1),
        "seed": run_cfg.seed,
        "temperature": run_cfg.temperature,
        "top_p": run_cfg.top_p,
        "use_json_mode": run_cfg.use_json_mode,
        "prompt_versions": {"system": system_version, "ontology": ontology_version},
        "time_window": {
            "first_created": min(created_vals) if created_vals else None,
            "last_created": max(created_vals) if created_vals else None,
        },
        "tokens": {
            "prompt": int(total_prompt),
            "completion": int(total_completion),
            "total": int(total_prompt + total_completion),
        },
        "estimated_total_cost_usd": round(est_cost, 6) if est_cost is not None else None,
        "prompt_hashes": {
            "system_message_sha256": _sha256(system_message),
            "ontology_prompt_sha256": _sha256(ontology_prompt),
        },
        "input_file": str(io_cfg.input_xlsx),
    }
    out_meta.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return out_jsonl, out_xlsx, out_meta
