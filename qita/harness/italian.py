"""Deterministic check that an answer is written in Italian.

Counts frequent function words of Italian and English. Paths, numbers and code
do not count either way. Short answers pass if they have at least one Italian
function word and no more English ones; an answer with no function words at all
(e.g. a bare number) passes, because it is not in another language.
"""

from __future__ import annotations

import re

IT = set("""il lo la i gli le un una uno di da in con su per tra fra del della dei delle al alla ai alle
dal dalla nel nella sul sulla che chi non è e ed o ma se come anche più sono ho hai ha abbiamo hanno
questo questa questi queste quello quella ci si mi ti vi ne c'è puoi posso vuoi ecco già ancora
tutto tutti solo dopo prima ora oggi file cartella nota""".split())
# No 'a', 'so', 'do': they are common Italian words too.
EN = set("""the an of to in on for with and or but is are was were be been have has had this that
these those it its you your i we they there here what which who can could would should will not
does did from by at as if then than all any some just only also into about file folder""".split())
# Words spelled the same in both languages do not decide anything.
SHARED = IT & EN


def counts(text: str) -> tuple[int, int]:
    # Code, URLs, paths and quoted material (a translation, a commit message) do not count.
    text = re.sub(r"`[^`]*`|https?://\S+|/\S+|«[^»]*»|“[^”]*”|\"[^\"]*\"", " ", text)
    # Emails and file names (marta@posta.it, liste.py) are neither language: ".it" is not the English "it".
    text = re.sub(r"\S+@\S+|\b[\w-]+\.[a-z]{2,4}\b", " ", text)
    toks = re.findall(r"[a-zàèéìòù']+", text.casefold())
    toks = [t.strip("'") for t in toks]
    it = sum(1 for t in toks if t in IT and t not in SHARED)
    en = sum(1 for t in toks if t in EN and t not in SHARED)
    return it, en


def is_italian(text: str) -> bool:
    it, en = counts(text)
    if it == 0 and en == 0:
        return bool(text.strip())
    return it >= 1 and it >= 2 * en
