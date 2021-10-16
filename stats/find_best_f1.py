#!/usr/bin/env python3

#!/usr/bin/env python3

import json
from pathlib import Path
from itertools import chain
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import random


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def restruct_gold(g):
    out = {}
    for i in g:
        out[i["tatbestand"]] = i["begründung"]
    return out


files = list(
    zip(
        map(load_json, sorted(Path("computed/structure").glob("*.json"))),
        map(load_json, sorted(Path("computed/references").glob("*.json"))),
        map(restruct_gold, map(load_json, sorted(Path("gold").glob("*.json")))),
        map(lambda n: n.stem, sorted(Path("computed/structure").glob("*.json"))),
    )
)

random.seed(0)
files = random.choices(files, k=3)

METHODS = [
    {
        "name": "Jaccard-Index",
        "internal_name": "jaccard",
    },
    {
        "name": "Wortembedding",
        "internal_name": "word_embedding",
    },
    {
        "name": "Satzembedding",
        "internal_name": "sent_embedding",
    },
    {
        "name": "Levenshtein",
        "internal_name": "levenshtein",
    },
]

BEST_THRESH = {
    "word_embedding": [],
    "sent_embedding": [],
    "jaccard": [],
    "levenshtein": [],
}

for struct, refs, gold, id in files:
    print(id, struct["slug"])
    begr_sents = list(
        chain.from_iterable(
            map(
                lambda s: s["sentences"],
                chain.from_iterable(struct["tatbestand"].values()),
            )
        )
    )

    for metric in BEST_THRESH.keys():
        possible_thresholds = set()
        for sent in begr_sents:
            for k, v in gold.items():
                if k not in refs:
                    continue

                for i in refs[k][metric]:
                    if any(map(lambda j: i["text"] == j, v)):
                        if metric == "relation":
                            print(i)
                        possible_thresholds.add(i["score"])

            if sent not in refs:
                continue
        if metric == "relation":
            print(possible_thresholds)

        best_f1 = 0
        best_thresh = 0

        for threshold in possible_thresholds:
            method = []
            is_gold = []

            for sent in begr_sents:
                metric_r = refs[sent][metric] if sent in refs else []
                results = list(filter(lambda i: i["score"] > threshold, metric_r))

                if sent in gold:
                    for g in gold[sent]:
                        is_gold.append(True),
                        method.append(any(map(lambda i: i["text"] == g, results)))
                for r in results:
                    if sent in gold and not any(
                        map(lambda j: r["text"] == j, gold[sent])
                    ):
                        is_gold.append(False)
                        method.append(True)

            f1 = f1_score(is_gold, method, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold

        BEST_THRESH[metric].append(best_thresh)

table = r"""\begin{tabular}{ll} \toprule
    Methode & Grenzwert \\ \midrule
"""
for method in METHODS:
    metric = method["internal_name"]
    threshold = str(round(np.mean(BEST_THRESH[metric]), 3))

    table += (
        " " * 4 + method["name"] + " & "
        r"\(" + str(threshold).replace(".", "{,}") + r"\)" + r" \\" + "\n"
    )
table += r"\end{tabular}"

with (Path("plots") / f"thresholds.tex").open("w") as f:
    f.write(table)
