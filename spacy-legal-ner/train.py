import spacy
import random
from spacy.util import minibatch
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer

nlp = spacy.load("de_core_news_sm")

to_train = "ner"
to_exclude = [pipe for pipe in nlp.pipe_names if pipe != to_train]

nlp = spacy.load("de_core_news_sm", disable=to_exclude)
scorer = Scorer(nlp)

doc_bin = DocBin().from_disk("bag.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

optimizer = nlp.create_optimizer()

for itn in range(100):
    print(f"Epoch {itn}")
    random.shuffle(docs)
    losses = {}
    examples = []
    for batch in minibatch(docs, 100):
        last_batch = batch
        examples = list(map(lambda d: Example(nlp.make_doc(d.text), d), batch))

        nlp.update(examples, sgd=optimizer, losses=losses, drop=0.4)

        scores = scorer.score(examples)

    print(f"losses: {losses}")
    scores = scorer.score(examples)
    for i in ["ents_p", "ents_r", "ents_f"]:
        print(f"{i}: {scores[i]}")

nlp.from_disk("out")
