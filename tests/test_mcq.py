"""Prompt and letter choice of the exam-question probe (qita/mcq.py)."""

from qita import mcq

ITEM = {"question": "Quale fiume attraversa Roma?", "options": ["Po", "Tevere", "Arno"], "answer": 1, "category": "x"}


def test_the_prompt_letters_every_option_and_asks_for_the_letter():
    p = mcq.prompt(ITEM)
    assert "A) Po\nB) Tevere\nC) Arno" in p
    assert p.endswith(mcq.INSTRUCTION) and "D)" not in p


def test_the_pick_is_the_highest_letter_among_the_item_options_only():
    ids = [10, 11, 12, 13, 14]
    logits = [0.0] * 20
    logits[11], logits[13] = 2.0, 5.0  # D is higher, but the item has three options
    assert mcq.pick(logits, ids, 3) == "B"
    assert mcq.pick(logits, ids, 4) == "D"


def test_the_sign_test_is_two_sided_and_one_when_tied():
    assert mcq.sign_p(5, 5) == 1.0
    assert mcq.sign_p(0, 0) == 1.0
    assert abs(mcq.sign_p(10, 0) - 2 / 1024) < 1e-12
