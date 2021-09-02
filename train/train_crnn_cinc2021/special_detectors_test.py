"""
test of special detectors
"""
import os, json, sys, importlib
from pathlib import Path
from typing import Sequence, Optional, NoReturn
from random import sample
import argparse

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base_dir)

def import_parents(level:int=1) -> NoReturn:
    # https://gist.github.com/vaultah/d63cb4c86be2774377aa674b009f759a
    import sys, importlib
    from pathlib import Path
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]
    
    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError: # already removed
        pass
    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__) # won't be needed after that

if __name__ == "__main__" and __package__ is None:
    import_parents(level=1)


from torch_ecg.utils.misc import list_sum, get_date_str
from .data_reader import CINC2021Reader
from .cfg import (
    BaseCfg, Standard12Leads,
    twelve_leads, six_leads, four_leads, three_leads, two_leads,
)
from .special_detectors import special_detectors


__all__ = ["test_sd",]


try:
    DR = CINC2021Reader(db_dir=BaseCfg.db_dir)
except:
    DR = None


def test_sd(rec:str,
            leads:Sequence[str]=Standard12Leads,
            dr:Optional[type(CINC2021Reader)]=None,
            verbose:int=0) -> dict:
    """
    """
    try:
        ret_val = _test_sd(rec, leads, dr=dr, verbose=verbose)
    except:
        ret_val = {"rec": rec, "label": [], "pred": [], "err": "True"}
    return ret_val


def _test_sd(rec:str,
             leads:Sequence[str]=Standard12Leads,
             dr:Optional[type(CINC2021Reader)]=None,
             verbose:int=0) -> dict:
    """
    """
    _dr = dr or DR
    fs = _dr.get_fs(rec)
    data = _dr.load_data(rec, leads=list(leads))
    # ann = _dr.load_ann(rec)["diagnosis_scored"]["diagnosis_abbr"]
    ann = _dr.get_labels(rec, scored_only=True, fmt="a", normalize=True)
    cc = special_detectors(data, fs, rpeak_fn="xqrs", leads=list(leads), verbose=verbose)
    cc = [k.replace("is_", "") for k,v in cc.items() if v]
    ret_val = {"rec": rec, "label": ann, "pred": cc, "err": "False"}
    return ret_val


def get_parser() -> dict:
    """
    """
    description = "performing test using the special detectors"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--db-dir", type=str,
        help="directory of the CinC2021 database",
        dest="db_dir",
    )
    parser.add_argument(
        "-p", "--prop", type=float,
        help="proportion of the records used for testing, should be within ]0,1[",
        dest="proportion",
    )
    parser.add_argument(
        "-f", "--full", action="store_true",
        help=f"run over the full dataset, overrides proportion",
        dest="full",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help=f"verbosity",
        dest="verbose",
    )
    parser.add_argument(
        "--log-every", type=int, default=200,
        help="write to file per `log-every` records",
        dest="log_every",
    )

    args = vars(parser.parse_args())

    return args


if __name__ == "__main__":
    args = get_parser()
    print(f"args = {args}")
    size = args.get("proportion") or 1  # up to 1
    log_every = args.get("log_every")
    db_dir = args.get("db_dir", None)
    _dr = CINC2021Reader(db_dir) if db_dir else DR
    if _dr is None:
        raise ValueError(f"data directory could not be found!")
    if args.get("full", False):
        print("full dataset")
        all_candidates = _dr.df_stats[_dr.df_stats.diagnosis_scored.apply(lambda v:len(v))!=0].record.tolist()
    else:
        print("partial dataset")
        all_candidates = sorted(set(list_sum(
            sample(_dr.diagnoses_records_list[k], int(round(len(_dr.diagnoses_records_list[k])*size))) \
                for k in ["Brady", "STach", "SB", "LQRSV", "RAD", "LAD", "PR",]
        )))
    print(f"number of candidate records: {len(all_candidates)}")
    all_results = {
        "twelve_leads": [],
        "six_leads": [],
        "four_leads": [],
        "three_leads": [],
        "two_leads": [],
    }
    leads = {
        "twelve_leads": list(twelve_leads),
        "six_leads": list(six_leads),
        "four_leads": list(four_leads),
        "three_leads": list(three_leads),
        "two_leads": list(two_leads),
    }
    # NOTE that multiprocessing is used inside the signal preprocessing function,
    # hence test_sd could not be parallel computed
    for k, l in all_results.items():
        save_fp = os.path.join(BaseCfg.log_dir, f"{k}_sd_res_{get_date_str()}.txt")
        gather_res = ""
        for idx, rec in enumerate(all_candidates):
            sd_res = test_sd(rec, leads=leads[k], dr=_dr)
            gather_res += f"{json.dumps(sd_res, ensure_ascii=False)}\n"
            if idx % log_every == 0:
                with open(save_fp, "a") as f:
                    f.write(gather_res)
                gather_res = ""
            l.append(sd_res)
            print(f"{k} --- {idx+1}/{len(all_candidates)}", end="\r")
        with open(save_fp, "a") as f:
            f.write(gather_res)
        print("\n")

    os.makedirs(BaseCfg.log_dir, exist_ok=True)
    save_fp = os.path.join(BaseCfg.log_dir, "special_detectors_test_results.json")
    with open(save_fp, "w") as f:
        json.dump(all_results, f, ensure_ascii=False)
