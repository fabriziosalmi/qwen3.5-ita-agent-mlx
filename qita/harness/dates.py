"""Days named in an Italian text, without the harness's dependencies, so scripts/textcheck.py can use it in
CLM's environment."""

from __future__ import annotations

import re

MONTHS = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
          "novembre", "dicembre")
_DAYLIST = re.compile(rf"\b((?:\d{{1,2}}\s*(?:,|e|-|–|al|a)\s*)*\d{{1,2}})\s+({'|'.join(MONTHS)})\b")


def named_days(text: str) -> set[tuple[int, int]]:
    """(month, day) of every day a text names: "7, 8 e 9 maggio", "dal 27 al 31 ottobre", 2027-05-07, 7/5."""
    out = set()
    for m in _DAYLIST.finditer(text):
        month = MONTHS.index(m.group(2)) + 1
        parts = re.split(r"\s*(,|\be\b|-|–|\bal?\b)\s*", m.group(1))
        nums, seps = [int(x) for x in parts[::2]], parts[1::2]
        for k, n in enumerate(nums):
            out.add((month, n))
            if k and seps[k - 1] in ("-", "–", "al", "a"):
                out |= {(month, d) for d in range(nums[k - 1], n)}
    out |= {(int(mo), int(d)) for _, mo, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)}
    out |= {(int(mo), int(d)) for d, mo in re.findall(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])", text) if 1 <= int(mo) <= 12}
    return out
