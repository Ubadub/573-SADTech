from collections import defaultdict
import os
import pickle
import sys
from typing import Any, Callable, Sequence

from sklearn.metrics import classification_report
import yaml

infinite_defaultdict: Callable[[], defaultdict] = lambda: defaultdict(
    infinite_defaultdict
)

avg: Callable[[Sequence[float]], float] = lambda x: sum(x) / len(x)


def recurs_split(path: str) -> Sequence[str]:
    components = []
    cur_path = path
    while cur_path and cur_path != "/":
        cur_path, new_comp = os.path.split(cur_path)
        components.append(new_comp)
    components.reverse()
    return components


def flatten_default_dict(dd: Any) -> Any:
    if isinstance(dd, defaultdict):
        dd = dict(dd)
        for k, v in dd.items():
            dd[k] = flatten_default_dict(v)
    return dd


def multiseed_average(root_dir: str) -> dict:
    root_dir = os.path.realpath(root_dir)
    d = infinite_defaultdict()
    for parent_path, child_dirs, child_files in os.walk(root_dir, topdown=False):
        if child_files:
            if child_dirs:
                sys.exit(
                    f"File(s) {child_files} nested at same level"
                    f"as director(y/ies) {child_dirs}; aborting."
                )
            components = recurs_split(parent_path.removeprefix(root_dir))
            curr_d = d
            for component in components[:-1]:
                curr_d = curr_d[component]
            if not isinstance(curr_d[components[-1]], defaultdict):
                sys.exit(f"Detected already initialized value at {parent_path}")
            else:
                files = [os.path.join(parent_path, c) for c in child_files]
                scores = defaultdict(list)
                for f in files:
                    with open(f, "rb") as _:
                        res_dict = pickle.load(file=_)
                    report = classification_report(
                        res_dict["y_true"],
                        res_dict["y_pred"],
                        output_dict=True,
                        zero_division=0,
                    )
                    scores["macro_f1"].append(report["macro avg"]["f1-score"])
                    scores["macro_prec"].append(report["macro avg"]["precision"])
                    scores["macro_rec"].append(report["macro avg"]["recall"])
                    scores["accuracy"].append(report["accuracy"])
                curr_d[components[-1]] = {
                    score_name: avg(score_list)
                    # score_name: sum(score_list) / len(score_list)
                    for score_name, score_list in scores.items()
                }

    return flatten_default_dict(d)


if __name__ == "__main__":
    print(yaml.dump(multiseed_average(sys.argv[1])))
