#!/usr/bin/env python3

from __future__ import annotations

import re
import string
from typing import Iterator, Optional
from itertools import product, combinations
from bs4 import BeautifulSoup
import httpx
import spacy
from pathlib import Path
from pydantic import BaseModel
from itertools import chain
import uvicorn
from typing import List, Dict
from fastapi import FastAPI
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import stanza
from polyleven import levenshtein as calc_levenshtein
from fastapi.middleware.cors import CORSMiddleware
import sys

DEBUG = False

sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

stanza.download("de", processors="tokenize", package="hdt")
snlp = stanza.Pipeline("de", processors="tokenize", package="hdt")

nlp = spacy.load("de_core_news_lg")

# stanza already handles this and it only serves to confuse spaCy
# https://github.com/explosion/spaCy/issues/1032#issuecomment-343233390
@spacy.Language.component("prevent-sbd")
def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


nlp.add_pipe("prevent-sbd", before="parser")


app = FastAPI()

origins = ["http://localhost:1234", "http://[::]:1234", "http://127.0.0.1:1234"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROM = {
    "VIII",
    "II",
    "XLIII",
    "I",
    "XVIII",
    "XLVII",
    "XXIV",
    "XXVI",
    "XXXIII",
    "XXVIII",
    "XXXVIII",
    "XXXIX",
    "XIII",
    "XIX",
    "XXIII",
    "XXXV",
    "XXI",
    "IV",
    "IX",
    "XXXIV",
    "X",
    "XXXVI",
    "XXX",
    "VII",
    "XLIV",
    "XLVIII",
    "XX",
    "XLI",
    "XV",
    "XLII",
    "III",
    "XVII",
    "XL",
    "XXXI",
    "XVI",
    "XII",
    "XXXII",
    "VI",
    "XLVI",
    "XI",
    "XXXVII",
    "V",
    "XXIX",
    "XXV",
    "XXVII",
    "XLV",
    "XXII",
    "XIV",
}
ROM_LOWER = set(map(lambda r: r.lower(), ROM))


def remove_stop_words(s):
    for i in nlp.Defaults.stop_words:
        s.replace(i, "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


tatbestand = r"\s*".join("^Tatbestand") + r"\s*(?::\s*)?$"
tatbestand_re = re.compile(tatbestand, re.IGNORECASE)
entscheidungsgründe = (
    r"^\s*(:?(:?"
    + r"\s*".join("Entscheidungsgründe")
    + ")|(?:"
    + r"\s*".join("Gründe")
    + r"))\s*(?::\s*)?$"
)
entscheidungsgründe_re = re.compile(entscheidungsgründe, re.IGNORECASE)

KEYWORDS_KOSTEN = [
    ["Kosten", "Verfahren"],
    ["Kosten", "Rechtsstreit"],
    ["Kosten", "Rechtsstreits"],
    ["verurteilen", "zahlen"],
    ["Basiszinssatz"],
]
KEYWORDS_VOLLSTRECKUNG = ["vollstreckbar", "Vollstreckbarkeit", "Vollstreckung"]

# "ich" ist Lemma zu allen Pronomen
KEYWORDS_KLÄGER = [
    "Kläger",
    "Klägerin",
    "Klaeger",
    "Klaegerin",
    "Antragsteller",
    "Antragstellerin",
    "ich",
]
KEYWORDS_BEKLAGTE = ["Beklagte", "beklagt", "Antragsgegner", "Antragsgegnerin", "ich"]
KEYWORDS_ANTRAG = [["verlangern"], ["beantragen"]]
KEYWORDS_BEHAUPTUNG = [
    ["behaupten"],
    ["vorwerfen"],
    ["wirft", "vor"],
    ["vortragen"],
    ["tragen", "vor"],
    ["meinen"],
    ["Auffassung", "sein"],
    ["Meinung", "sein"],
    ["Ansicht", "sein"],
    ["bestreiten"],
    ["Nichtwissen"],
]
KEYWORDS_SCHRIFTSÄTZE = [
    "Schriftsatz",
    "Schriftsätze",
    "Schriftsaetze",
    "Beweis",
    "Beweismittel",
    "Beweisaufnahme",
    "Sitzungsprotokoll",
    "Verfügung",
    "Beschluss",
    "Verhandlung",
]
KEYWORDS_BEZUG = ["Bezug", "erheben", "verweisen"]


def get_first_sent(sents):
    if len(sents) == 0:
        return ""
    elif len(sents) == 1:
        return sents[0]

    if sents[0][-1] != "." or len(sents[0]) - 1 <= 0:
        return sents[0]

    if any(
        map(
            lambda char_set: all(map(lambda char: char in char_set, sents[0][:-1])),
            [
                ROM,
                ROM_LOWER,
                string.ascii_lowercase,
                string.ascii_uppercase,
                string.digits,
            ],
        )
    ):
        return sents[1]
    return sents[0]


class OLDP_Case(BaseModel):
    slug: str
    content: str


class TextWithLineNo(BaseModel):
    line_no: Optional[str] = None
    text: str


class Row(BaseModel):
    line_no: Optional[str] = None
    sentences: List[str]


class Case_Tenor(BaseModel):
    hauptsache: List[Row]
    kosten: List[Row]
    vollstreckung: List[Row]


class Case_Tatbestand(BaseModel):
    einleitungssatz: List[Row] = []
    untreitiges: List[Row] = []
    stretiges_kläger: List[Row] = []
    antrag_kläger: List[Row] = []
    # I.d.R. Die Klage abzuweisen
    antrag_beklagte: List[Row] = []
    stretiges_beklagte: List[Row] = []
    prozessgeschichte: List[Row] = []
    # Verweise
    schriftsätze: List[Row] = []


class Case_Nebenentscheidung(BaseModel):
    number: Optional[str] = None
    self: List[Row] = []
    subsections: List[Case_Nebenentscheidung] = []


Case_Nebenentscheidung.update_forward_refs()


class Case_Entscheidungsgründe(BaseModel):
    einleitungssatz: Optional[Row] = None
    # Oft weggellasen
    zuständigkeit: Optional[Row] = None
    nebenentscheidungen: List[Case_Nebenentscheidung] = []


class Case(BaseModel):
    slug: str
    tenor: Case_Tenor
    tatbestand: Case_Tatbestand
    entscheidungsgründe: Case_Entscheidungsgründe


def get_subsections(c):
    yield c["self"]

    for i in c["subsections"]:
        yield chain.from_iterable(get_subsections(i))


def get_sents(text: str) -> Iterator[str]:
    return map(lambda s: s.text, snlp(text).sentences)


@app.get("/fetch/{case_no}", response_model=OLDP_Case)
async def fetch_case(case_no: int) -> OLDP_Case:
    try:
        # Try to fetch from disk to avoid potential network errors for select urteile
        p = (Path("oldp-urteile") / str(case_no)).with_suffix(".json")
        with p.open() as f:
            return OLDP_Case.parse_file(f)
    except Exception as e:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://de.openlegaldata.io/api/cases/{case_no}/?format=json"
            )
        return OLDP_Case.parse_obj(response.json())


@app.post("/structure", response_model=Case)
def structure(case: OLDP_Case) -> Case:
    out_1 = {"tenor": [], "tatbestand": [], "gründe": []}
    current_section = "tenor"

    # Pass 1 (general sections)
    for row in case.content.split("\n"):
        soup = BeautifulSoup(row, "html5lib")

        # Possible structures
        # * {text} (span/h4/strong)
        # * span.absatzRechts {id} p.absatzLinks {text}
        # * dl > {dt {id}, dd {text}}
        # * div > [dl {dt {id}, dd {text}]

        line_no = soup.find(class_="absatzRechts")
        if line_no is None:
            line_no = soup.find("dt")
        if line_no is not None:
            line_no.extract()
            line_no = int(line_no.text)

        text = soup.find(class_="absatzLinks")
        if text is None:
            text = soup.find("dd")
        if text is None:
            text = soup
        text = soup.text

        if text.strip() == "":
            continue

        if tatbestand_re.fullmatch(text) is not None:
            current_section = "tatbestand"
            continue
        elif entscheidungsgründe_re.fullmatch(text) is not None:
            current_section = "gründe"
            continue

        out_1[current_section].append(
            Row(line_no=line_no, sentences=list(get_sents(text)))
        )

    out_2 = {
        "slug": case.slug,
        "tenor": {"hauptsache": [], "kosten": [], "vollstreckung": []},
        "tatbestand": {
            "einleitungssatz": None,
            "untreitiges": [],
            "stretiges_kläger": [],
            "antrag_kläger": [],
            # I.d.R. "Die Klage abzuweisen"
            "antrag_beklagte": [],
            "stretiges_beklagte": [],
            "prozessgeschichte": [],
            # Verweise
            "schriftsätze": [],
        },
        "entscheidungsgründe": {
            "einleitungssatz": None,
            # Oft weggellasen
            "zuständigkeit": None,
            "nebenentscheidungen": [
                {
                    "number": None,
                    "self": [],
                    "subsections": [],
                }
            ],
        },
    }

    out_2 = Case.construct(**out_2)

    current_section = "hauptsache"

    for row in out_1["tenor"]:
        sents = row.sentences
        first_sent = nlp(get_first_sent(row.sentences))
        lemma = set(map(lambda i: i.lemma_, first_sent))
        old_section = current_section

        if (
            any(map(lambda i: all(map(lambda j: j in lemma, i)), KEYWORDS_KOSTEN))
            and not "verurteilen" in lemma
        ):
            current_section = "kosten"
        elif any(map(lambda i: i in lemma, KEYWORDS_VOLLSTRECKUNG)):
            current_section = "vollstreckung"

        out_2.tenor[current_section].append(row)

        if DEBUG:
            fst = get_first_sent(sents)
            print(f"{old_section} -> {current_section}: {fst} {lemma}", file=sys.stderr)
            print(
                list(
                    filter(lambda i: all(map(lambda j: j in lemma, i)), KEYWORDS_KOSTEN)
                ),
                list(
                    filter(
                        lambda i: all(map(lambda j: j in lemma, i)),
                        KEYWORDS_VOLLSTRECKUNG,
                    )
                ),
                file=sys.stderr,
            )

    current_section = "einleitungssatz"

    for row in out_1["tatbestand"]:
        if len(row.sentences) == 0 or sum(map(len, row.sentences)) == 0:
            continue

        sents = row.sentences
        first_sent = nlp(sents[0])
        lemma = set(map(lambda i: i.lemma_, first_sent))

        old_section = current_section

        if current_section == "einleitungssatz":
            out_2.tatbestand["einleitungssatz"] = [row]
            current_section = "untreitiges"
        elif (
            current_section == "untreitiges"
            and any(map(lambda i: i in lemma, KEYWORDS_KLÄGER))
            and any(
                map(lambda i: all(map(lambda j: j in lemma, i)), KEYWORDS_BEHAUPTUNG)
            )
        ):
            current_section = "stretiges_kläger"
        elif (
            (current_section == "stretiges_kläger" or current_section == "untreitiges")
            and any(map(lambda i: i in lemma, KEYWORDS_KLÄGER))
            and any(map(lambda i: all(map(lambda j: j in lemma, i)), KEYWORDS_ANTRAG))
        ):
            current_section = "antrag_kläger"
        elif (
            current_section == "antrag_kläger"
            and any(map(lambda i: i in lemma, KEYWORDS_BEKLAGTE))
            and any(map(lambda i: all(map(lambda j: j in lemma, i)), KEYWORDS_ANTRAG))
        ):
            current_section = "antrag_beklagte"
        elif (
            (current_section == "antrag_beklagte" or current_section == "antrag_kläger")
            and any(map(lambda i: i in lemma, KEYWORDS_BEKLAGTE))
            and any(
                map(lambda i: all(map(lambda j: j in lemma, i)), KEYWORDS_BEHAUPTUNG)
            )
        ):
            current_section = "stretiges_beklagte"
        elif (
            current_section == "stretiges_beklagte"
            and any(map(lambda i: i in lemma, KEYWORDS_SCHRIFTSÄTZE))
            and any(map(lambda i: i in lemma, KEYWORDS_BEZUG))
        ):
            current_section = "schriftsätze"

        if DEBUG and old_section != current_section:
            fst = sents[0]
            print(f"{old_section} -> {current_section}: {fst}", file=sys.stderr)
            print(
                list(
                    filter(lambda i: all(map(lambda j: j in lemma, i)), KEYWORDS_ANTRAG)
                ),
                list(
                    filter(
                        lambda i: all(map(lambda j: j in lemma, i)), KEYWORDS_BEHAUPTUNG
                    )
                ),
                file=sys.stderr,
            )

        out_2.tatbestand[current_section].append(row)

    current_section = "einleitungssatz"

    def is_int(p) -> bool:
        try:
            int(p)
            return True
        except:
            return False

    numbering_style = []
    for row in out_1["gründe"]:
        sents = row.sentences

        first_sent = nlp(sents[0])
        text = sents[0]
        lemma = set(map(lambda i: i.lemma_, first_sent))

        if current_section == "einleitungssatz":
            out_2.entscheidungsgründe["zuständigkeit"] = row
            current_section = "nebenentscheidungen"
            continue

        if current_section == "nebenentscheidungen":
            prefix = ""
            found_style = None
            for delim in [".", ")"]:
                prefix = text.split(delim)[0].strip()
                if prefix == "":
                    continue
                if prefix[0] == "(":
                    delim = "(" + delim
                    prefix = prefix.replace("(", "")
                if prefix in ROM:
                    found_style = "ROM" + delim
                    break
                elif prefix in ROM_LOWER:
                    found_style = "rom" + delim
                    break
                elif is_int(prefix):
                    found_style = "int" + delim
                    break
                elif prefix in string.ascii_lowercase:
                    found_style = "ascii" + delim
                    break
                elif prefix in string.ascii_uppercase:
                    found_style = "ASCII" + delim
                    break

            if found_style is not None:
                if found_style[-2] == "(":
                    prefix = "(" + prefix
                prefix += found_style[-1]

            if found_style in numbering_style:
                numbering_style = numbering_style[
                    : numbering_style.index(found_style) + 1
                ]
            elif found_style is not None:
                numbering_style.append(found_style)

            item = out_2.entscheidungsgründe["nebenentscheidungen"]

            added_new_subsection = False
            for _style in numbering_style:
                if len(item[-1]["subsections"]) == 0 and found_style is not None:
                    added_new_subsection = True
                    item[-1]["subsections"].append(
                        {
                            "number": prefix,
                            "self": [row],
                            "subsections": [],
                        }
                    )
                item = item[-1]["subsections"]
            if found_style is None:
                item[-1]["self"].append(row)
            elif not added_new_subsection:
                item.append(
                    {
                        "number": prefix,
                        "self": [row],
                        "subsections": [],
                    }
                )
    return out_2


def get_triples_sdp(doc):
    edges = []

    for token in doc:
        for child in token.children:
            edges.append((token, child))

    graph = nx.Graph(edges)
    spans = list(doc.ents) + list(doc.noun_chunks)

    triples = []

    span_tokens = set()
    for s in spans:
        for token in s:
            span_tokens.add(token.text)

    for span_0, span_1 in combinations(spans, 2):
        if span_0.text == span_1.text:
            continue
        shortest_path = None
        for token_0, token_1 in product(span_0, span_1):
            path = nx.shortest_path(graph, token_0, token_1)
            if shortest_path is None or len(path) > len(shortest_path):
                shortest_path = path

        if shortest_path is None:
            continue

        shortest_path = list(
            filter(lambda i: i not in span_0 and i not in span_1, shortest_path)
        )
        if len(shortest_path) == 0:
            continue

        verb = None
        for i in shortest_path:
            if i.text in span_tokens:
                break
            if i.pos_ == "VERB":
                verb = i
                break
        if verb is None:
            continue

        triples.append((span_0.lemma_, span_1.lemma_, verb.lemma_))
    return triples


def fetch_triples(sent):
    triples = get_triples_sdp(sent)
    out = []
    for triple in triples:
        sub, obj_a, obj_b = triple
        text = (str(sub), str(obj_a), str(obj_b))
        out.append(
            {
                "text": text,
                "no_stop": (
                    remove_stop_words(sub),
                    remove_stop_words(obj_a),
                    remove_stop_words(obj_b),
                ),
            }
        )
    return out


class WordWiseMatch(BaseModel):
    word: str
    matches: List[str]
    lemma: str


class ReferenceMatch(BaseModel):
    score: float
    text: str


class ReferenceItem(BaseModel):
    section: str
    jaccard: Optional[List[ReferenceMatch]]
    wordwise: Optional[List[WordWiseMatch]]
    relation: Optional[List[ReferenceMatch]]
    word_embedding: Optional[List[ReferenceMatch]]
    sent_embedding: Optional[List[ReferenceMatch]]
    levenshtein: Optional[List[ReferenceMatch]]


def add_match(out, method, a, b, score):
    if a not in out:
        out[a] = ReferenceItem(section="tatbestand")
    if b not in out:
        out[b] = ReferenceItem(section="begründung")

    for p, s in [(a, b), (b, a)]:
        if method == "jaccard":
            if out[p].jaccard is None:
                out[p].jaccard = []
            out[p].jaccard.append(
                ReferenceMatch(
                    score=score,
                    text=s,
                )
            )
        elif method == "relation":
            if out[p].relation is None:
                out[p].relation = []
            out[p].relation.append(
                ReferenceMatch(
                    score=score,
                    text=s,
                )
            )
        elif method == "word_embedding":
            if out[p].word_embedding is None:
                out[p].word_embedding = []
            out[p].word_embedding.append(
                ReferenceMatch(
                    score=score,
                    text=s,
                )
            )
        elif method == "sent_embedding":
            if out[p].sent_embedding is None:
                out[p].sent_embedding = []
            out[p].sent_embedding.append(
                ReferenceMatch(
                    score=score,
                    text=s,
                )
            )
        elif method == "levenshtein":
            if out[p].levenshtein is None:
                out[p].levenshtein = []
            out[p].levenshtein.append(
                ReferenceMatch(
                    score=score,
                    text=s,
                )
            )


@app.post("/semantic_references", response_model=Dict[str, ReferenceItem])
def find_refs(
    case: Case,
    jaccard: Optional[float] = None,
    word_embedding: Optional[float] = None,
    sent_embedding: Optional[float] = None,
    levenshtein: Optional[float] = None,
    relation: Optional[float] = None,
):
    cache = {}

    out_2 = case.dict()

    real_out = {}

    for a, b in product(
        chain.from_iterable(
            map(
                lambda i: i["sentences"],
                chain.from_iterable(out_2["tatbestand"].values()),
            )
        ),
        chain.from_iterable(
            map(
                lambda j: j["sentences"],
                chain.from_iterable(
                    map(
                        lambda i: chain.from_iterable(get_subsections(i)),
                        out_2["entscheidungsgründe"]["nebenentscheidungen"],
                    ),
                ),
            )
        ),
    ):

        for item in [a, b]:
            if item in cache:
                continue
            else:
                cache[item] = {}

            if (
                jaccard is not None
                or sent_embedding is not None
                or word_embedding is not None
                or levenshtein is not None
                or relation is not None
            ) and "nlp" not in cache[item]:
                cache[item]["nlp"] = nlp(item)

            if (
                jaccard is not None or levenshtein is not None
            ) and "words" not in cache[item]:
                cache[item]["words"] = set(
                    [
                        t.lemma_
                        for t in cache[item]["nlp"]
                        if not t.is_stop and not t.is_punct and not t.pos_ in ["AUX"]
                    ]
                )

            if relation is not None and "triples" not in cache[item]:
                cache[item]["triples"] = fetch_triples(cache[item]["nlp"])

            if sent_embedding is not None and "sent_embedding" not in cache[item]:
                cache[item]["embedding"] = sentence_model.encode(
                    [item], show_progress_bar=False
                )

            if word_embedding is not None and "word_embedding" not in cache[item]:
                cache[item]["word_embedding"] = nlp(
                    " ".join(
                        [
                            str(t)
                            for t in cache[item]["nlp"]
                            if not t.is_stop
                            and not t.is_punct
                            and not t.pos_ in ["AUX"]
                        ]
                    )
                )

        if len(a) < 4 or len(b) < 5:
            continue

        if jaccard is not None:
            a_words = cache[a]["words"]
            b_words = cache[b]["words"]
            inter = a_words.intersection(b_words)
            union = a_words.union(b_words)
            metric_overlap = 0
            if len(union) != 0:
                metric_overlap = float(len(inter)) / float(len(union))

            if metric_overlap >= jaccard:
                add_match(real_out, "jaccard", a, b, metric_overlap)
                for (id_prim, id_sec) in [(a, b), (b, a)]:
                    if real_out[id_prim].wordwise is None:
                        real_out[id_prim].wordwise = list(
                            map(
                                lambda word: WordWiseMatch(
                                    word=word.text_with_ws,
                                    matches=[],
                                    lemma=word.lemma_,
                                ),
                                cache[id_prim]["nlp"],
                            )
                        )
                    for idx, item in enumerate(real_out[id_prim].wordwise):
                        if item.lemma in inter:
                            real_out[id_prim].wordwise[idx].matches.append(id_sec)

        if levenshtein is not None:
            a_words = cache[a]["words"]
            b_words = cache[b]["words"]
            distance_levenshtein = calc_levenshtein("".join(a_words), "".join(b_words))

            metric_levenshtein = 1.0 - distance_levenshtein / max(
                sum(map(len, a_words)), sum(map(len, b_words))
            )

            if metric_levenshtein >= levenshtein:
                add_match(real_out, "levenshtein", a, b, metric_levenshtein)

        if sent_embedding is not None:
            metric_se = float(
                util.pytorch_cos_sim(cache[a]["embedding"], cache[b]["embedding"])
            )
            if metric_se >= sent_embedding:
                add_match(real_out, "sent_embedding", a, b, metric_se)

        if word_embedding is not None:
            metric_we = cache[a]["word_embedding"].similarity(
                cache[b]["word_embedding"]
            )
            if metric_we >= word_embedding:
                add_match(real_out, "word_embedding", a, b, metric_we)

        if relation is not None:
            metric_rel = 0
            for t_a, t_b in product(cache[a]["triples"], cache[b]["triples"]):

                sub_a, obj_a, _pred_a = t_a["no_stop"]
                sub_b, obj_b, _pred_b = t_b["no_stop"]

                if (sub_a == sub_b and obj_a == obj_b) or (
                    sub_a == obj_b and obj_a == sub_b
                ):
                    metric_rel += 1

            if metric_rel >= relation:
                add_match(real_out, "relation", a, b, metric_rel)

    return real_out


def start():
    uvicorn.run("urteile_server.main:app", reload=True)


if __name__ == "__main__":
    start()
