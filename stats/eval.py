#!/usr/bin/env python3

import json
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import chain, combinations
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats.mstats import gmean
import numpy as np

plt.style.use("seaborn")

plot_out = Path("plots")


METHODS = [
    {
        "name": "Relationen",
        "internal_name": "relation",
    },
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

F1 = {
    "word_embedding": [],
    "sent_embedding": [],
    "jaccard": [],
    "levenshtein": [],
    "relation": [],
}

RECALL = {
    "word_embedding": [],
    "sent_embedding": [],
    "jaccard": [],
    "levenshtein": [],
    "relation": [],
}

PRECISION = {
    "word_embedding": [],
    "sent_embedding": [],
    "jaccard": [],
    "levenshtein": [],
    "relation": [],
}

RANKING_GMEAN = {
    "word_embedding": [],
    "sent_embedding": [],
    "jaccard": [],
    "levenshtein": [],
    "relation": [],
}

RANKING_GMEAN_FIRST = {
    "word_embedding": [],
    "sent_embedding": [],
    "jaccard": [],
    "levenshtein": [],
    "relation": [],
}

RANKING_GMEAN_NONZERO = {
    "word_embedding": [],
    "sent_embedding": [],
    "jaccard": [],
    "levenshtein": [],
    "relation": [],
}

RANKING_GMEAN_MRR = {
    "word_embedding": [],
    "sent_embedding": [],
    "jaccard": [],
    "levenshtein": [],
    "relation": [],
}

THRESHOLDS = {
    "word_embedding": 0.728,
    "sent_embedding": 0.418,
    "jaccard": 0.081,
    "levenshtein": 0.274,
    "relation": 1,
}

def make_table(dataset, cols, out_name, higher_is_better=True):
    table = (
        "\\begin{tabular}{llll} \\toprule\n"
        + " " * 4 + "Methode & "
        + " & ".join(map(lambda c: c[0], cols))
        + r"\\ \midrule"
        + "\n"
    )

    best = {}
    best_v = {}
    values = {}

    for i in METHODS:
        values[i["internal_name"]] = {}

    for name, func in cols:
        best[name] = None
        best_v[name] = None
        for i in METHODS:
            values[i["internal_name"]][name] = func(dataset[i["internal_name"]])

    for i in METHODS:
        k = i["internal_name"]

        for name, _ in cols:
            value = values[k][name]

            if (
                best[name] is None
                or (higher_is_better and best_v[name] < value)
                or (not higher_is_better and best_v[name] > value)
            ):
                best[name] = k
                best_v[name] = value

    for i in METHODS:
        k = i["internal_name"]

        table += " " * 4 + i["name"]

        for name, _ in cols:
            value = values[k][name]
            table += " & "

            out = str(round(value, 2)).replace(".", "{,}")

            if best[name] == k:
                out = f"\\symbf{{{out}}}"
            table += f"\\({out}\\)"

        table += r" \\" + "\n"
    table += r"\end{tabular}"

    with (plot_out / (out_name + ".tex")).open("w") as f:
        f.write(table)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def restruct_gold(g):
    out = {}
    for i in g:
        out[i["tatbestand"]] = i["begründung"]
    return out


for structure, refs, gold, p in zip(
    map(load_json, sorted(Path("computed/structure").glob("*.json"))),
    map(load_json, sorted(Path("computed/references").glob("*.json"))),
    map(restruct_gold, map(load_json, sorted(Path("gold").glob("*.json")))),
    sorted(Path("computed/structure").glob("*.json")),
):
    skip_f1 = structure["slug"] in [
        "ag-koln-2020-11-05-125-c-30220",
        "lg-koln-2020-10-27-3-o-519",
        "lg-dortmund-2020-07-15-10-o-2720",
    ]

    RANKING = {
        "word_embedding": [],
        "sent_embedding": [],
        "jaccard": [],
        "levenshtein": [],
        "relation": [],
    }

    RANKING_NONZERO = {
        "word_embedding": [],
        "sent_embedding": [],
        "jaccard": [],
        "levenshtein": [],
        "relation": [],
    }

    RANKING_WEIGHTED = {
        "word_embedding": [],
        "sent_embedding": [],
        "jaccard": [],
        "levenshtein": [],
        "relation": [],
    }

    RANKING_FIRST = {
        "word_embedding": [],
        "sent_embedding": [],
        "jaccard": [],
        "levenshtein": [],
        "relation": [],
    }

    RANKING_MRR = {
        "word_embedding": [],
        "sent_embedding": [],
        "jaccard": [],
        "levenshtein": [],
        "relation": [],
    }

    MATCHES = {
        "word_embedding": {"method": [], "gold": []},
        "sent_embedding": {"method": [], "gold": []},
        "jaccard": {"method": [], "gold": []},
        "levenshtein": {"method": [], "gold": []},
        "relation": {"method": [], "gold": []},
    }

    for sent in chain.from_iterable(
        map(
            lambda s: s["sentences"],
            chain.from_iterable(structure["tatbestand"].values()),
        )
    ):
        for method in METHODS:
            added_first = False
            ranking_out = []
            ranking_weighted_out = []
            ranking_first_out = []
            ranking_nonzero_out = []

            metric = method["internal_name"]

            if sent not in refs:
                continue

            results = list(
                filter(lambda i: i["score"] > THRESHOLDS[metric], refs[sent][metric])
            )

            ranking = list(
                sorted(refs[sent][metric], key=lambda i: i["score"], reverse=True)
            )

            if not skip_f1:
                if sent in gold:
                    for g in gold[sent]:
                        MATCHES[metric]["gold"].append(True),
                        MATCHES[metric]["method"].append(
                            any(map(lambda i: i["text"] == g, results))
                        )
                for r in results:
                    if sent in gold and r["text"] not in gold[sent]:
                        MATCHES[metric]["gold"].append(False)
                        MATCHES[metric]["method"].append(True)

            staggered_rank = 0

            rank = 1
            last_result = 0.0
            last_rank = 1
            for match in ranking:
                score = match["score"]
                is_gold_match = sent in gold and match["text"] in gold[sent]
                if is_gold_match:
                    ranking_out.append(rank)
                    ranking_weighted_out.append(rank)
                    if not added_first:
                        ranking_first_out.append(rank)
                        added_first = True
                    if not score == 0:
                        ranking_nonzero_out.append(rank)

                if last_result != score:
                    for i in range(len(ranking_weighted_out)):
                        if ranking_weighted_out[i] == last_rank:
                            ranking_weighted_out[i] += staggered_rank / 2
                    for i in range(len(ranking_first_out)):
                        if ranking_first_out[i] == last_rank:
                            ranking_first_out[i] += staggered_rank / 2
                    for i in range(len(ranking_nonzero_out)):
                        if ranking_nonzero_out[i] == last_rank:
                            ranking_nonzero_out[i] += staggered_rank / 2

                    rank += staggered_rank + int(not is_gold_match)
                    staggered_rank = 0
                else:
                    staggered_rank += 1

                last_result = score
                last_rank = rank

            for i in range(len(ranking_weighted_out)):
                if ranking_weighted_out[i] == last_rank:
                    ranking_weighted_out[i] += staggered_rank / 2
            for i in range(len(ranking_first_out)):
                if ranking_first_out[i] == last_rank:
                    ranking_first_out[i] += staggered_rank / 2
            for i in range(len(ranking_nonzero_out)):
                if ranking_nonzero_out[i] == last_rank:
                    ranking_nonzero_out[i] += staggered_rank / 2
            RANKING[metric] += ranking_out
            RANKING_WEIGHTED[metric] += ranking_weighted_out
            RANKING_FIRST[metric] += ranking_first_out
            RANKING_NONZERO[metric] += ranking_nonzero_out

            if len(ranking_first_out) == 0:
                RANKING_MRR[metric].append(0)
            else:
                RANKING_MRR[metric].append(1 / ranking_first_out[0])

    plt.ylabel("Rang")
    plt.xlabel("Methode")

    plt.boxplot(
        list(map(lambda m: RANKING[m["internal_name"]], METHODS)),
        showfliers=False,
    )
    plt.xticks(
        list(range(1, len(METHODS) + 1)), list(map(lambda m: m["name"], METHODS))
    )
    plt.tight_layout()
    plt.savefig(plot_out / f"ranking_box_{p.stem}.png", dpi=300)

    plt.cla()
    plt.clf()
    plt.close()

    plt.ylabel("Rang")
    plt.xlabel("Methode")

    plt.boxplot(
        list(map(lambda m: RANKING_FIRST[m["internal_name"]], METHODS)),
        showfliers=False,
    )
    plt.xticks(
        list(range(1, len(METHODS) + 1)), list(map(lambda m: m["name"], METHODS))
    )
    plt.tight_layout()
    plt.savefig(plot_out / f"ranking_first_box_{p.stem}.png", dpi=300)

    plt.cla()
    plt.clf()
    plt.close()

    plt.ylabel("Rang")
    plt.xlabel("Methode")

    plt.boxplot(
        list(map(lambda m: RANKING_NONZERO[m["internal_name"]], METHODS)),
        showfliers=False,
    )
    plt.xticks(
        list(range(1, len(METHODS) + 1)), list(map(lambda m: m["name"], METHODS))
    )
    plt.tight_layout()
    plt.savefig(plot_out / f"ranking_nonzero_box_{p.stem}.png", dpi=300)

    plt.cla()
    plt.clf()
    plt.close()

    for method in METHODS:
        metric = method["internal_name"]
        v = RANKING_WEIGHTED[metric]
        if len(v) == 0:
            continue
        RANKING_GMEAN[metric].append(np.mean(v))

        v = RANKING_FIRST[metric]
        if len(v) == 0:
            continue
        RANKING_GMEAN_FIRST[metric].append(np.mean(v))

        v = RANKING_NONZERO[metric]
        if len(v) == 0:
            continue
        RANKING_GMEAN_NONZERO[metric].append(np.mean(v))
        v = RANKING_MRR[metric]
        RANKING_GMEAN_MRR[metric].append(np.mean(v))

    plt.ylabel("Rang")
    plt.xlabel("Methode")
    plt.boxplot(
        list(map(lambda m: RANKING_WEIGHTED[m["internal_name"]], METHODS)),
        showfliers=False,
    )
    plt.xticks(
        list(range(1, len(METHODS) + 1)), list(map(lambda m: m["name"], METHODS))
    )
    plt.tight_layout()
    plt.savefig(plot_out / f"ranking_box_weighted_{p.stem}.png", dpi=300)

    plt.cla()
    plt.clf()
    plt.close()

    table = r"""
\begin{tabular}{llll} \toprule
    Methode & Genauigkeit & Trefferquote & F1-Maß \\ \midrule
"""

    if not skip_f1:
        for method in METHODS:
            metric = method["internal_name"]
            scores = [
                precision_score(
                    MATCHES[metric]["gold"], MATCHES[metric]["method"], zero_division=0
                ),
                recall_score(
                    MATCHES[metric]["gold"], MATCHES[metric]["method"], zero_division=0
                ),
                f1_score(
                    MATCHES[metric]["gold"], MATCHES[metric]["method"], zero_division=0
                ),
            ]
            PRECISION[metric].append(scores[0])
            RECALL[metric].append(scores[1])
            F1[metric].append(scores[2])
        for method in METHODS:
            metric = method["internal_name"]
            scores = [PRECISION[metric][-1], RECALL[metric][-1], F1[metric][-1]]
        
            scores_str = []
            

            for s, d in zip(scores, [PRECISION, RECALL, F1]):
                b = True
                for m in METHODS:
                    if metric != m["internal_name"] and d[m["internal_name"]][-1] > s:
                        b = False
                
                o = str(round(s, 2)).replace(".", "{,}")
                if b:
                    o = f"\\symbf{{{o}}}"
                o =  r"\(" + o  + r"\)"

                scores_str.append(o)


            table += (
                " " * 4
                + method["name"]
                + " & "
                + " & ".join(scores_str)
                + r" \\"
                + "\n"
            )
        table += r"\end{tabular}"

        with (plot_out / f"{p.stem}_f1.tex").open("w") as f:
            f.write(table)

plt.ylabel("MRR (Mean Reciprocal Rank)")
plt.xlabel("Methode")
plt.boxplot(
    list(map(lambda m: RANKING_GMEAN_MRR[m["internal_name"]], METHODS)),
    showfliers=False,
)
plt.xticks(list(range(1, len(METHODS) + 1)), list(map(lambda m: m["name"], METHODS)))
plt.savefig(plot_out / f"mrr_distribution.png", dpi=300)

plt.cla()
plt.clf()
plt.close()
plt.ylabel("F1-Maß")
plt.xlabel("Methode")

plt.boxplot(
    list(map(lambda m: F1[m["internal_name"]], METHODS)),
    showfliers=False,
)
plt.xticks(list(range(1, len(METHODS) + 1)), list(map(lambda m: m["name"], METHODS)))
plt.savefig(plot_out / f"f1_distribution.png", dpi=300)

plt.cla()
plt.clf()
plt.close()


COLS = [
    ("Durchschnitt", np.average),
    ("Median", np.median),
    ("geometrisches Mittel", gmean),
]
make_table(F1, COLS, "f1_dist")
make_table(RECALL, COLS, "recall_dist")
make_table(PRECISION, COLS, "precision_dist")
make_table(RANKING_GMEAN, COLS, "ranking_dist", higher_is_better=False)
make_table(RANKING_GMEAN_FIRST, COLS, "ranking_first_dist", higher_is_better=False)
make_table(RANKING_GMEAN_NONZERO, COLS, "ranking_nonzero_dist", higher_is_better=False)
make_table(RANKING_GMEAN_MRR, COLS, "ranking_mrr_dist")
