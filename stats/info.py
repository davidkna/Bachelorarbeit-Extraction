#!/usr/bin/env python3

import json
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import chain, repeat
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats.mstats import gmean
import numpy as np

plt.style.use("seaborn")

plot_out = Path("plots")


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def restruct_gold(g):
    out = {}
    for i in g:
        out[i["tatbestand"]] = i["begründung"]
    return out


def restruct_gold2(g):
    out = {}
    for i in g:
        for j in i["begründung"]:
            if j not in out:
                out[j] = []
            out[j].append(i)
    return out


SEC_NAME = {
    # "einleitungssatz":  "Einleitungssatz",
    "untreitiges": "Unstreitig",
    "stretiges_kläger": "Streitiger Klägervortrag",
    "antrag_kläger": "Antrag Kläger",
    # I.d.R. Die Klage abzuweisen
    "antrag_beklagte": "Antrag Beklagte",
    "stretiges_beklagte": "Streitiger Beklagtenvortrag",
    # "prozessgeschichte": "Prozessgeschichte",
    # Verweise
    # "schriftsätze": "Bezugnahme auf Beweismittel",
}
sub_g = {
    "untreitiges": [],
    "stretiges_kläger": [],
    "antrag_kläger": [],
    # I.d.R. Die Klage abzuweisen
    "antrag_beklagte": [],
    "stretiges_beklagte": [],
}

for structure, refs, gold, p in zip(
    map(load_json, sorted(Path("computed/structure").glob("*.json"))),
    map(load_json, sorted(Path("computed/references").glob("*.json"))),
    map(restruct_gold, map(load_json, sorted(Path("gold").glob("*.json")))),
    map(lambda p: p.name, sorted(Path("computed/structure").glob("*.json"))),
):
    sub = {
        "untreitiges": 0,
        "stretiges_kläger": 0,
        "antrag_kläger": 0,
        # I.d.R. Die Klage abzuweisen
        "antrag_beklagte": 0,
        "stretiges_beklagte": 0,
    }
    for subsection, sent in chain.from_iterable(
        map(
            lambda s: zip(repeat(s[0]), s[1]["sentences"]),
            chain.from_iterable(
                map(lambda i: zip(repeat(i[0]), i[1]), structure["tatbestand"].items())
            ),
        )
    ):
        if subsection in sub:
            sub[subsection] += 1

    # plt.bar(np.arange(len(sub)), list(sub.values()))

    for k, v in sub.items():
        sub_g[k].append(v)


total = sum(map(np.average, sub_g.values()))

for k, v in sub_g.items():
    print(k, (100 * np.average(v) / total).round(2))


plt.bar(np.arange(len(sub_g)), list(map(np.mean, sub_g.values())))
plt.xticks(np.arange(len(SEC_NAME)), list(SEC_NAME.values()))
plt.tight_layout()
plt.savefig(plot_out / f"tatbestand.png", dpi=300)
plt.cla()
plt.clf()
plt.close()

sub_b = {
    "untreitiges": [],
    "stretiges_kläger": [],
    "antrag_kläger": [],
    # I.d.R. Die Klage abzuweisen
    "antrag_beklagte": [],
    "stretiges_beklagte": [],
}

for structure, refs, gold, p in zip(
    map(load_json, sorted(Path("computed/structure").glob("*.json"))),
    map(load_json, sorted(Path("computed/references").glob("*.json"))),
    map(restruct_gold, map(load_json, sorted(Path("gold").glob("*.json")))),
    map(lambda p: p.name, sorted(Path("computed/structure").glob("*.json"))),
):
    sub = {
        "untreitiges": 0,
        "stretiges_kläger": 0,
        "antrag_kläger": 0,
        # I.d.R. Die Klage abzuweisen
        "antrag_beklagte": 0,
        "stretiges_beklagte": 0,
    }
    for subsection, sent in chain.from_iterable(
        map(
            lambda s: zip(repeat(s[0]), s[1]["sentences"]),
            chain.from_iterable(
                map(lambda i: zip(repeat(i[0]), i[1]), structure["tatbestand"].items())
            ),
        )
    ):

        if subsection in sub and sent in gold:
            sub[subsection] += len(gold[sent])

    plt.bar(np.arange(len(sub)), list(sub.values()))
    for k, v in sub.items():
        sub_b[k].append(v)

total = sum(map(np.average, sub_b.values()))

for k, v in sub_b.items():
    print(k, (100 * np.average(v) / total).round(2))


plt.xticks(np.arange(len(SEC_NAME)), list(SEC_NAME.values()))
plt.tight_layout()
plt.savefig(plot_out / f"tatbestand_gold.png", dpi=300)
plt.cla()
plt.clf()
plt.close()

for structure, refs, gold, p in zip(
    map(load_json, sorted(Path("computed/structure").glob("*.json"))),
    map(load_json, sorted(Path("computed/references").glob("*.json"))),
    map(restruct_gold, map(load_json, sorted(Path("gold").glob("*.json")))),
    sorted(Path("computed/structure").glob("*.json")),
):
    pos = []
    unstreitiges = list(
        chain.from_iterable(
            map(lambda s: s["sentences"], structure["tatbestand"]["untreitiges"])
        )
    )
    print(len(unstreitiges))
    xpos = []
    for i, u in enumerate(unstreitiges):
        if u in gold:
            pos.append(len(gold[u]))

            xpos += len(gold[u]) * [i / len(unstreitiges)]
        else:
            pos.append(0)
    # print(pos, xpos)
    print(np.median(xpos), np.mean(xpos), gmean(xpos))
    plt.violinplot(xpos, vert=False)
    plt.tight_layout()
    plt.savefig(plot_out / f"unstreitig_dist_{p.stem}.png", vert=True, dpi=300)
    plt.cla()
    plt.clf()
    plt.close()


def get_subsections(c):
    yield c["self"]

    for i in c["subsections"]:
        yield chain.from_iterable(get_subsections(i))


for structure, refs, gold, p in zip(
    map(load_json, sorted(Path("computed/structure").glob("*.json"))),
    map(load_json, sorted(Path("computed/references").glob("*.json"))),
    map(restruct_gold2, map(load_json, sorted(Path("gold").glob("*.json")))),
    sorted(Path("computed/structure").glob("*.json")),
):
    subs = list(
        map(
            lambda i: list(chain.from_iterable(get_subsections(i))),
            structure["entscheidungsgründe"]["nebenentscheidungen"],
        ),
    )

    gold_pos = []
    for sub in subs:
        sents = list(chain.from_iterable(map(lambda j: j["sentences"], sub)))
        for i, sent in enumerate(sents):
            if sent in gold:
                gold_pos += len(gold[sent]) * [i / len(sents)]

    print(gold_pos)
    print(np.median(gold_pos), np.mean(gold_pos), gmean(gold_pos))

    plt.violinplot(gold_pos, vert=False)
    plt.tight_layout()
    plt.savefig(plot_out / f"subsection_dist_{p.stem}.png", vert=True, dpi=300)
    plt.cla()
    plt.clf()
    plt.close()
