from __future__ import annotations
import argparse
import pandas as pd

from ellma.evaluation import evaluate_multi_model

def main():
    p = argparse.ArgumentParser(description="Evaluate model outputs stored in an Excel workbook (multiple sheets).")
    p.add_argument("--xlsx", required=True, help="Excel workbook containing model sheets")
    p.add_argument("--sheets", nargs="+", required=True, help="Sheet names, e.g., 4.0 5.0 5.0-mini")
    p.add_argument("--human-col", default="final", help="Column name for human reference (e.g., 'final')")
    p.add_argument("--out", required=True, help="Output xlsx for all evaluation tables")
    args = p.parse_args()

    tables = evaluate_multi_model(
        xlsx_path=args.xlsx,
        sheet_names=args.sheets,
        human_col=args.human_col,
        pricing=None,
    )

    with pd.ExcelWriter(args.out, engine="xlsxwriter") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31])

    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
