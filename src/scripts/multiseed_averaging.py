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
        if child_dirs:
            seed_check = [_.startswith("seed") for _ in child_dirs]
            if all(seed_check):
                if child_files:
                    sys.exit(
                        f"Path {parent_path} contains files nested at same level as"
                        "seed folders; aborting."
                    )
                components = recurs_split(parent_path.removeprefix(root_dir))
                curr_d = d
                for component in components[:-1]:
                    curr_d = curr_d[component]
                if not isinstance(curr_d[components[-1]], defaultdict):
                    sys.exit(f"Detected already initialized value at {parent_path}")
                else:
                    files = [os.path.join(parent_path, c, "results.pkl") for c in child_dirs]
                    scores = defaultdict(list)
                    for f in files:
                        try:
                            with open(f, "rb") as _:
                                res_dict = pickle.load(file=_)
                        except FileNotFoundError:
                            continue
                        except EOFError:
                            continue

                        y_true = res_dict.get("dev_y_true", res_dict.get("y_true"))
                        y_pred = res_dict.get("dev_y_pred", res_dict.get("y_pred"))
                        if not y_true or not y_pred:
                            raise ValueError(f"No y_true found. Keys: {res_dict.keys()}")
                        dev_report = classification_report(
                            y_true,
                            y_pred,
                            # res_dict["dev_y_pred"],
                            output_dict=True,
                            zero_division=0,
                        )

                        scores["dev_macro_f1"].append(dev_report["macro avg"]["f1-score"])
                        scores["dev_macro_prec"].append(dev_report["macro avg"]["precision"])
                        scores["dev_macro_rec"].append(dev_report["macro avg"]["recall"])
                        scores["dev_accuracy"].append(dev_report["accuracy"])

                        if "test_y_true" in res_dict:
                            test_report = classification_report(
                                res_dict["test_y_true"],
                                res_dict["test_y_pred"],
                                output_dict=True,
                                zero_division=0,
                            )
                            scores["test_macro_f1"].append(test_report["macro avg"]["f1-score"])
                            scores["test_macro_prec"].append(test_report["macro avg"]["precision"])
                            scores["test_macro_rec"].append(test_report["macro avg"]["recall"])
                            scores["test_accuracy"].append(test_report["accuracy"])
                    curr_d[components[-1]] = {
                        score_name: avg(score_list)
                        # score_name: sum(score_list) / len(score_list)
                        for score_name, score_list in scores.items()
                    }

            elif any(seed_check):
                sys.exit(
                    f"Dir(s) {child_dirs} contains mix of seed/non-seed dirs; aborting."
                )
            # if child_files:
            #     parent_path_base_name = os.path.basename(parent_path)
            #     if parent_path_base_name.startswith("seed"):
            #         parent_path_dir_name = os.path.dirname(parent_path)
            #     if child_dirs:
            #         sys.exit(
            #             f"File(s) {child_files} nested at same level"
            #             f"as director(y/ies) {child_dirs}; aborting."
            #         )

    return flatten_default_dict(d)


if __name__ == "__main__":
    print(yaml.dump(multiseed_average(sys.argv[1])))
