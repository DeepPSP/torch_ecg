"""
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import requests
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfg import BaseCfg  # noqa: E402

_URLS = {
    "summary": "https://moody-challenge.physionet.org/2022/results/summary.tsv",
    "official_murmur_scores": "https://moody-challenge.physionet.org/2022/results/official_murmur_scores.tsv",
    "official_outcome_scores": "https://moody-challenge.physionet.org/2022/results/official_outcome_scores.tsv",
    "unofficial_murmur_scores": "https://moody-challenge.physionet.org/2022/results/unofficial_murmur_scores.tsv",
    "unofficial_outcome_scores": "https://moody-challenge.physionet.org/2022/results/unofficial_outcome_scores.tsv",
}

_FINAL_SCORES_FILE = BaseCfg.log_dir.parent / "results" / "final_scores.xlsx"


def _fetch_final_results() -> Dict[str, pd.DataFrame]:
    """ """
    df = {}
    for name, url in _URLS.items():
        _http_get(url, BaseCfg.log_dir / f"{name}.tsv")
        df[name] = pd.read_csv(BaseCfg.log_dir / f"{name}.tsv", sep="\t")
    df["official_murmur_scores"].insert(
        3,
        "Ranking Metric (Weighted Accuracy on Test Set)",
        df["official_murmur_scores"]["Weighted Accuracy on Test Set"].values,
    )
    df["official_outcome_scores"].insert(
        3,
        "Ranking Metric (Cost on Test Set)",
        df["official_outcome_scores"]["Cost on Test Set"].values,
    )
    df["unofficial_murmur_scores"].insert(
        2,
        "Ranking Metric (Weighted Accuracy on Test Set)",
        df["unofficial_murmur_scores"]["Weighted Accuracy on Test Set"].values,
    )
    df["unofficial_outcome_scores"].insert(
        2,
        "Ranking Metric (Cost on Test Set)",
        df["unofficial_outcome_scores"]["Cost on Test Set"].values,
    )
    for name in _URLS:
        os.remove(BaseCfg.log_dir / f"{name}.tsv")
    return df


def _http_get(url: str, fname: os.PathLike) -> None:
    """ """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=Path(fname).name,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=1.0,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def _update_final_results() -> None:
    """ """
    df = _fetch_final_results()
    updated = False
    save_path = _FINAL_SCORES_FILE
    df_old = pd.read_excel(save_path, sheet_name=None, engine="openpyxl") if save_path.exists() else None
    if df_old is None:
        updated = True
    else:
        for name in _URLS:
            if not df_old[name].equals(df[name]):
                updated = True
                break
    if updated:
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            for name, df_ in df.items():
                df_.to_excel(writer, sheet_name=name, index=False)
        print(f"final results saved to {str(save_path)}")
    else:
        mtime = datetime.fromtimestamp(save_path.stat().st_mtime)
        mtime = datetime.strftime(mtime, "%Y-%m-%d %H:%M:%S")
        print(f"final results is up-to-date, last updated at {mtime}")


def _get_row(
    team_name: str,
    task: str,
    metric: str,
    evaluated_set: str = "test",
    col: Optional[str] = None,
    latest: bool = False,
) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    """ """
    if latest or not _FINAL_SCORES_FILE.exists():
        _update_final_results()
    task = task.lower()
    assert task in ["murmur", "outcome"]
    evaluated_set = evaluated_set.title()
    assert evaluated_set in ["Test", "Training", "Validation"]
    if col is None:
        assert metric in [
            "AUROC",
            "AUPRC",
            "F-measure",
            "Accuracy",
            "Weighted Accuracy",
            "Cost",
        ]
        col = f"{metric} on {evaluated_set} Set"
    df_all = pd.read_excel(_FINAL_SCORES_FILE, sheet_name=None, engine="openpyxl")
    df = df_all[f"official_{task}_scores"]
    df = df.sort_values(by=col, ascending=True if "Cost" in col else False).reset_index(drop=True)
    if team_name not in df.Team.to_list():
        unofficial_teams = df_all[f"unofficial_{task}_scores"].Team.to_list()
        newline = "\n"
        assert team_name in unofficial_teams, (
            f"Team \042{team_name}\042 not found, please check "
            f"whether the team name is correct{newline}official teams are {df.Team.to_list()}"
            f"{newline}unofficial teams are {unofficial_teams}"
        )
        row = None
        df = None
    else:
        row = df[df.Team == team_name].iloc[0]
    return row, df


def get_ranking(
    team_name: str,
    task: str,
    metric: str,
    evaluated_set: str = "test",
    col: Optional[str] = None,
    latest: bool = False,
) -> int:
    """ """
    row, df = _get_row(team_name, task, metric, evaluated_set, col, latest)
    if row is None:
        ranking = "unofficial"
    else:
        ranking = row.name + 1
        ranking = f"{ranking} / {len(df)}"
    return ranking


def get_score(
    team_name: str,
    task: str,
    metric: str,
    evaluated_set: str = "test",
    col: Optional[str] = None,
    latest: bool = False,
) -> str:
    """ """
    evaluated_set = evaluated_set.title()
    assert evaluated_set in ["Test", "Training", "Validation"]
    if col is None:
        assert metric in [
            "AUROC",
            "AUPRC",
            "F-measure",
            "Accuracy",
            "Weighted Accuracy",
            "Cost",
        ]
        col = f"{metric} on {evaluated_set} Set"
    row, _ = _get_row(team_name, task, metric, evaluated_set, col, latest)
    if row is None:
        score = "unofficial"
    else:
        score = str(row[col])
    return score


def get_team_digest(team_name: str, fmt: str = "pd", latest: bool = False) -> Union[str, pd.DataFrame]:
    """ """
    assert fmt.lower() in [
        "pd",
        "tex",
        "latex",
    ], f"`fmt` must be pd, tex or latex, but got {fmt}"
    metrics = [
        "Weighted Accuracy",
        "Cost",
        "Accuracy",
        "F-measure",
        "AUROC",
        "AUPRC",
    ]
    evaluated_sets = ["Test", "Validation", "Training"]
    tasks = ["murmur", "outcome"]
    rows = []
    for metric in metrics:
        for es in evaluated_sets:
            rows.append(
                [
                    get_score(team_name, "murmur", metric, es, latest=latest),
                    get_ranking(team_name, "murmur", metric, es, latest=latest),
                    get_score(team_name, "outcome", metric, es, latest=latest),
                    get_ranking(team_name, "outcome", metric, es, latest=latest),
                ]
            )
            if metric == "Cost":
                try:
                    rows[-1][0] = int(float(rows[-1][0]))
                except ValueError:
                    pass
                try:
                    rows[-1][2] = int(float(rows[-1][2]))
                except ValueError:
                    pass
    df = pd.DataFrame(rows)
    tasks = [item.capitalize() for item in tasks]
    df.columns = pd.MultiIndex.from_product([tasks, ["Score", "Rank"]])
    df.index = pd.MultiIndex.from_product([metrics, evaluated_sets])
    if fmt.lower() == "pd":
        return df
    elif fmt.lower() in ["latex", "tex"]:
        tex = re.sub("[ \t]+", " ", df.to_latex())
        tex = tex.replace("Weighted Accuracy", "\\multirow{3}{*}{Wt. Acc.}")
        tex = tex.replace("\\\\\nCost", "\\\\ \\hline\n\\multirow{3}{*}{Cost}")
        tex = tex.replace("\\\\\nAccuracy", "\\\\ \\hline\n\\multirow{3}{*}{Accuracy}")
        tex = tex.replace("\\\\\nCost", "\\\\ \\hline\n\\multirow{3}{*}{Cost}")
        tex = tex.replace("\\\\\nAUROC", "\\\\ \\hline\n\\multirow{3}{*}{AUROC}")
        tex = tex.replace("\\\\\nAUPRC", "\\\\ \\hline\n\\multirow{3}{*}{AUPRC}")
        tex = tex.replace("bottomrule", "hlineB{3.5}")
        tex = [
            r"% requires packages boldline, multirow",
            r"% put the following in preamble",
            r"% \usepackage{multirow}",
            r"% \usepackage{boldline}",
            "\\setlength\\tabcolsep{1pt}",
            "\\begin{tabular}{@{\\extracolsep{4pt}}llllll@{}}",
            "\\hlineB{3.5}",
            " & & \\multicolumn{2}{c}{Murmur} & \\multicolumn{2}{c}{Outcome} \\\\ \\cline{3-4} \\cline{5-6}",
            " & & Score & Ranking & Score & Ranking \\\\",
            "\\hline",
        ] + tex.splitlines()[5:]

        # emphasize the ranking metrics
        # Wt. Acc. of murmur on test set
        row_7 = tex[7].split(" & ")
        row_7[2] = f"\\textbf{{{row_7[2]}}}"
        row_7[3] = f"\\textbf{{{row_7[3]}}}"
        tex[7] = " & ".join(row_7)
        # Cost of outcome on test set
        row_10 = tex[10].rstrip(r" \\").split(" & ")
        row_10[4] = f"\\textbf{{{row_10[4]}}}"
        row_10[5] = f"\\textbf{{{row_10[5]}}}"
        tex[10] = " & ".join(row_10) + r" \\"

        tex = "\n".join(tex)
        return tex


def main():
    _update_final_results()


if __name__ == "__main__":
    main()
