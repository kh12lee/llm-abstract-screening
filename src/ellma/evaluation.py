from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, wilcoxon
from sklearn.metrics import confusion_matrix, roc_auc_score
from statsmodels.stats.contingency_tables import cochrans_q

@dataclass(frozen=True)
class PricingMap:
    prompt_rate_per_1m: Dict[str, float]
    completion_rate_per_1m: Dict[str, float]

def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue

def holm_bonferroni(pvals: Dict[Tuple[str,str], float]) -> Dict[Tuple[str,str], float]:
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    adj = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        p_adj = min((m - i) * p, 1.0)
        running_max = max(running_max, p_adj)
        adj[k] = running_max
    return adj

def format_p(p: float) -> str:
    return "<0.001" if p < 1e-3 else f"{p:.3f}"

def compute_metrics(df: pd.DataFrame, human_col: str, pred_col: str = "ai_answer") -> Dict[str, float]:
    df = df.copy()
    df["expert_bin"] = (df[human_col].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["gpt_bin"] = (df[pred_col].astype(str).str.strip().str.lower() == "yes").astype(int)
    tn, fp, fn, tp = confusion_matrix(df["expert_bin"], df["gpt_bin"], labels=[0,1]).ravel()
    agreement = (df["expert_bin"] == df["gpt_bin"]).mean()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    auc = roc_auc_score(df["expert_bin"], df["gpt_bin"])  # binary AUC
    return {
        "Agreement": round(float(agreement), 3),
        "Sensitivity": round(float(sensitivity), 3),
        "Specificity": round(float(specificity), 3),
        "NPV": round(float(npv), 3),
        "PPV": round(float(ppv), 3),
        "AUC": round(float(auc), 3),
    }

def evaluate_multi_model(
    xlsx_path: str,
    sheet_names: List[str],
    human_col: str,
    pricing: PricingMap | None = None,
) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for name in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=name).copy()
        df["expert_bin"] = (df[human_col].astype(str).str.strip().str.lower() == "yes").astype(int)
        df["gpt_bin"] = (df["ai_answer"].astype(str).str.strip().str.lower() == "yes").astype(int)
        df["response_time_min"] = df.get("response_time_sec", 0) / 60.0
        if pricing is not None and "ai_prompt_tokens" in df.columns and "ai_completion_tokens" in df.columns:
            pr = pricing.prompt_rate_per_1m.get(name, list(pricing.prompt_rate_per_1m.values())[0])
            cr = pricing.completion_rate_per_1m.get(name, list(pricing.completion_rate_per_1m.values())[0])
            df["cost"] = (df["ai_prompt_tokens"]/1e6)*pr + (df["ai_completion_tokens"]/1e6)*cr
        else:
            df["cost"] = np.nan
        dfs[name] = df

    summary_rows = []
    for name, df in dfs.items():
        m = compute_metrics(df, human_col=human_col)
        m.update({
            "Model": name,
            "response_time_min_sum": round(float(df["response_time_min"].sum()), 3),
            "cost_sum": round(float(df["cost"].sum()), 3) if np.isfinite(df["cost"]).any() else np.nan,
            "N": int(len(df)),
        })
        summary_rows.append(m)
    summary_df = pd.DataFrame(summary_rows).set_index("Model")

    acc_mat = [ (dfs[n]["expert_bin"] == dfs[n]["gpt_bin"]).astype(int).to_numpy() for n in sheet_names ]
    acc_array = np.column_stack(acc_mat)
    q_acc_p = cochrans_q(acc_array).pvalue if acc_array.shape[1] > 2 else np.nan

    pos_mask = dfs[sheet_names[0]]["expert_bin"].to_numpy() == 1
    sens_mat = [ (dfs[n].loc[pos_mask, "gpt_bin"] == 1).astype(int).to_numpy() for n in sheet_names ]
    sens_array = np.column_stack(sens_mat)
    q_sens_p = cochrans_q(sens_array).pvalue if sens_array.shape[1] > 2 else np.nan

    neg_mask = dfs[sheet_names[0]]["expert_bin"].to_numpy() == 0
    spec_mat = [ (dfs[n].loc[neg_mask, "gpt_bin"] == 0).astype(int).to_numpy() for n in sheet_names ]
    spec_array = np.column_stack(spec_mat)
    q_spec_p = cochrans_q(spec_array).pvalue if spec_array.shape[1] > 2 else np.nan

    global_df = pd.DataFrame({
        "Metric": ["Accuracy","Sensitivity","Specificity"],
        "Cochran_Q_p": [q_acc_p, q_sens_p, q_spec_p],
    }).set_index("Metric")

    pairs = list(combinations(sheet_names, 2))

    def pairwise_mcnemar(vecs: List[np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p_raw: Dict[Tuple[str,str], float] = {}
        for a, b in pairs:
            i, j = sheet_names.index(a), sheet_names.index(b)
            av, bv = vecs[i], vecs[j]
            b01 = int(np.sum((av == 1) & (bv == 0)))
            c10 = int(np.sum((av == 0) & (bv == 1)))
            p_raw[(a,b)] = mcnemar_exact_p(b01, c10)
        p_adj = holm_bonferroni(p_raw)

        idx = sorted(set([a for a,_ in p_raw.keys()] + [b for _,b in p_raw.keys()]))
        def table(pdict: Dict[Tuple[str,str], float]) -> pd.DataFrame:
            out = pd.DataFrame(index=idx, columns=idx, data="")
            for (a,b), p in pdict.items():
                out.loc[a,b] = format_p(p)
                out.loc[b,a] = format_p(p)
            return out
        return table(p_raw), table(p_adj)

    acc_raw, acc_adj = pairwise_mcnemar(acc_mat)
    sens_raw, sens_adj = pairwise_mcnemar(sens_mat)
    spec_raw, spec_adj = pairwise_mcnemar(spec_mat)

    rt_series = [dfs[n]["response_time_min"].to_numpy() for n in sheet_names]
    friedman_rt_p = friedmanchisquare(*rt_series).pvalue if len(sheet_names) >= 3 else np.nan

    cost_series = [dfs[n]["cost"].to_numpy() for n in sheet_names]
    friedman_cost_p = friedmanchisquare(*cost_series).pvalue if len(sheet_names) >= 3 else np.nan

    def pairwise_wilcoxon(series_list: List[np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p_raw: Dict[Tuple[str,str], float] = {}
        for a, b in pairs:
            ia, ib = sheet_names.index(a), sheet_names.index(b)
            _, p = wilcoxon(series_list[ia], series_list[ib])
            p_raw[(a,b)] = float(p)
        p_adj = holm_bonferroni(p_raw)

        idx = sorted(set([a for a,_ in p_raw.keys()] + [b for _,b in p_raw.keys()]))
        def table(pdict: Dict[Tuple[str,str], float]) -> pd.DataFrame:
            out = pd.DataFrame(index=idx, columns=idx, data="")
            for (a,b), p in pdict.items():
                out.loc[a,b] = format_p(p)
                out.loc[b,a] = format_p(p)
            return out
        return table(p_raw), table(p_adj)

    rt_raw, rt_adj = pairwise_wilcoxon(rt_series)
    cost_raw, cost_adj = pairwise_wilcoxon(cost_series)

    friedman_df = pd.DataFrame({
        "Metric": ["response_time_min","cost"],
        "Friedman_p": [friedman_rt_p, friedman_cost_p],
    }).set_index("Metric")

    return {
        "summary": summary_df,
        "global_tests": global_df,
        "mcnemar_acc_raw": acc_raw,
        "mcnemar_acc_holm": acc_adj,
        "mcnemar_sens_raw": sens_raw,
        "mcnemar_sens_holm": sens_adj,
        "mcnemar_spec_raw": spec_raw,
        "mcnemar_spec_holm": spec_adj,
        "friedman": friedman_df,
        "wilcoxon_rt_raw": rt_raw,
        "wilcoxon_rt_holm": rt_adj,
        "wilcoxon_cost_raw": cost_raw,
        "wilcoxon_cost_holm": cost_adj,
    }
