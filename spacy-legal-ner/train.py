import spacy
import random
from spacy.util import minibatch
from spacy.scorer import Scorer
from spacy.training import Corpus

random.seed(0)

nlp = spacy.load("de_core_news_sm")

ner = nlp.get_pipe("ner")
for i in [
    # in Spacy
    # LOC, MISC, ORG, PER
    # Personenname
    "RR", "AN",
    # Ortsnamen
    "LD", "ST", "STR", "LDS",
    # Organisationsnamen
    "INN", "GRT", "UN", "MRK"
    # Normanenamen und -zitate
    "GS", "VO", "EUN",
    # Einzelfallreglungnamen und -zitate
    "VS", "VT",
    # Rechtssprechungszitate
    "RS",
    # Rechtsliteraturzitate
    "LIT"
]:
    ner.add_label(i)

corpus = Corpus("./bag.spacy")


with nlp.select_pipes(enable="ner"):
    optimizer = nlp.resume_training()
    scorer = Scorer(nlp)

    for itn in range(100):
        print(f"Epoch {itn}")
        train_data = list(corpus(nlp))
        random.shuffle(train_data)
        losses = {}
        examples = []
        for batch in minibatch(train_data, 100):
            examples = batch

            nlp.update(batch, sgd=optimizer, losses=losses, drop=0.4)

            scores = scorer.score(examples)

        print(f"losses: {losses}")
        scores = scorer.score(examples)
        for i in ["ents_p", "ents_r", "ents_f"]:
            print(f"{i}: {scores[i]}")

nlp.from_disk("out")
