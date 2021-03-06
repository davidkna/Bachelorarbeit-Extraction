import spacy
from pathlib import Path
from spacy import displacy
import os

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")
svg = displacy.render(doc, style="dep", jupyter=False)

STYLE = """
<style>
.displacy-word, .displacy-label {
    font-size: 2em;
}

.displacy-tag {
    font-size: 1.4em;
}
</style>
</svg>
"""

svg = svg.replace("</svg>", STYLE)

with Path("demo_dep.svg").open("w", encoding="utf-8") as f:
    f.write(svg)

# os.system('inkscape -D demo_dep.svg -o demo_dep.pdf --export-latex')
os.system("inkscape -D demo_dep.svg -o demo_dep.png")
