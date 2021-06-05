
import ast
from typing import Dict
import pandas as pd


def compute_normalize_map(data: pd.DataFrame, name: str, target: str) -> Dict:
    m = dict()
    for line, revenue in zip(data[name], data[target]):
        if type(line) == str:
            l = ast.literal_eval(line)
            for segment in l:
                if m.get(segment["name"]) is None:
                    m[segment["name"]] = [0, 0.]
                m[segment["name"]][0] += 1
                m[segment["name"]][1] += revenue

    for key, val in m.items():
        m[key][1] = val[1] / val[0]

    m = dict(filter(lambda x: x[1][0] >= 20, m.items()))
    max_average = max(map(lambda x: x[1], m.values()))
    for key, val in m.items():
        m[key] = val[1] / max_average
    return m


