from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ellma.config import IOConfig, Pricing, RunConfig
from ellma.llm_client import make_client
from ellma.screening import run_screening

def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="Run LLM-assisted abstract screening (raw text not stored by default).")
    p.add_argument("--dataset", required=True, help="Dataset preset: cholangio | nsclc")
    p.add_argument("--input", required=True, help="Input .xlsx with columns: title, abstract")
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("--run-name", default=None, help="Run name (default: dataset_model)")
    p.add_argument("--model", required=True, help="Model name or snapshot")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--no-json-mode", action="store_true", help="Disable response_format json_object")
    p.add_argument("--timeout-sec", type=int, default=60)
    p.add_argument("--input-per-1m", type=float, default=None, help="Pricing for run_meta.json (optional)")
    p.add_argument("--output-per-1m", type=float, default=None, help="Pricing for run_meta.json (optional)")
    p.add_argument("--store-raw-text", action="store_true", help="Store title/abstract in outputs (NOT recommended for Embase).")
    p.add_argument("--system-version", default="v1", help="System prompt version, e.g., v1, v2")
    p.add_argument("--ontology-version", default="v1", help="Ontology prompt version, e.g., v1, v2")
    args = p.parse_args()

    input_xlsx = Path(args.input)
    outdir = Path(args.outdir)
    run_name = args.run_name or f"{args.dataset}_{args.model}".replace("/", "_")

    pricing = None
    if args.input_per_1m is not None and args.output_per_1m is not None:
        pricing = Pricing(input_per_1m=args.input_per_1m, output_per_1m=args.output_per_1m)

    run_cfg = RunConfig(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_sec=args.timeout_sec,
        use_json_mode=(not args.no_json_mode),
        pricing=pricing,
    )
    io_cfg = IOConfig(
        input_xlsx=input_xlsx,
        outdir=outdir,
        run_name=run_name,
        store_raw_text=bool(args.store_raw_text),
    )

    df = pd.read_excel(input_xlsx)
    client = make_client(timeout_sec=args.timeout_sec)

    out_jsonl, out_xlsx, out_meta = run_screening(
        df=df,
        client=client,
        run_cfg=run_cfg,
        io_cfg=io_cfg,
        system_version=args.system_version,
        ontology_version=args.ontology_version,
    )
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_xlsx}")
    print(f"Wrote: {out_meta}")

if __name__ == "__main__":
    main()
