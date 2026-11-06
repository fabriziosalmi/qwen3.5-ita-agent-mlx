"""Round-6 chats (qita/worlds/chats.py) with scripted teacher and writer."""

import asyncio
import io
import json
import logging

import pytest

from qita.worlds import chats
from tests.test_harness import Scripted

logging.getLogger("mcp").setLevel(logging.CRITICAL)
DOCS = [{"id": "itaist-y", "text": "Il servizio di raccolta dei rifiuti passa il lunedì e il giovedì. " * 20}]
SYSTEM = "Sei un assistente che risponde in italiano."


class Teacher(Scripted):
    """Scripted outputs that also take clone(), as MLXBackend does."""

    seed = 0

    def __init__(self, outputs):
        super().__init__(outputs)
        self.seen = []

    def clone(self, **kw):
        self.clones = getattr(self, "clones", []) + [kw]
        return self

    def complete(self, messages, tools):
        self.seen.append(messages)
        return super().complete(messages, tools)


@pytest.fixture(autouse=True)
def fixed_system(monkeypatch):  # the system message the writer would phrase, already gated
    monkeypatch.setattr(chats, "system_text", lambda *a, **k: (SYSTEM, None))


def _run(kind, writer_out, teacher_out, plain_share=0.0, i=0, teacher=None):
    log, kept = io.StringIO(), io.StringIO()
    writer = Teacher(list(writer_out))
    teacher = teacher or Teacher(teacher_out)
    ok = asyncio.run(chats.one(i, kind, teacher, writer, DOCS, log, kept, plain_share))
    rows = [json.loads(x) for x in kept.getvalue().splitlines()]
    return ok, rows, json.loads(log.getvalue().splitlines()[-1])


REQUEST = "Mi fai un riassunto in due righe di questo avviso?"
ANSWER = "Il servizio raccoglie i rifiuti il lunedì e il giovedì."


def test_only_document_requests_are_generated():  # smokes 2-4: trivia, yes/no explanations, false premises
    assert chats.KINDS == ("doc",)


def test_a_doc_request_is_kept_under_the_header_with_the_teacher_answer():
    ok, rows, log = _run("doc", [REQUEST], [ANSWER])
    assert ok and log["kept"]
    users = [m["content"] for m in rows[0]["messages"] if m["role"] == "user"]
    assert "[INTENTO]\nrispondi" in users[0] and rows[0]["meta"]["source"] == "itaist-y"
    assert rows[0]["messages"][-1]["content"] == ANSWER


def test_a_plain_chat_has_no_header_no_tools_and_no_system_message():
    ok, rows, _ = _run("doc", [REQUEST], [ANSWER], plain_share=1.0)
    users = [m["content"] for m in rows[0]["messages"] if m["role"] == "user"]
    assert ok and users == [f"{REQUEST}\n\n{DOCS[0]['text']}"] and rows[0]["tools"] == []
    assert rows[0]["messages"][0]["role"] == "user"


def test_the_teacher_gets_the_chat_hint_not_the_agent_policy():
    from qita.worlds.style import POLICY_HINT
    teacher = Teacher([ANSWER])
    _run("doc", [REQUEST], [], teacher=teacher)
    system = teacher.seen[0][0]["content"]
    assert chats.CHAT_HINT in system and POLICY_HINT not in system


def test_a_doc_request_carries_the_excerpt():
    ok, rows, _ = _run("doc", ["Mi fai un riassunto in due righe di questo avviso?"],
                       ["Il servizio raccoglie i rifiuti il lunedì e il giovedì."])
    user = [m["content"] for m in rows[0]["messages"] if m["role"] == "user"][0]
    assert ok and "Mi fai un riassunto" in user and "raccolta dei rifiuti" in user


def test_a_doc_request_in_the_third_person_is_rejected():
    for text in ("La persona chiede un riassunto del bando.", "Chiedi un riassunto in poche righe di questo avviso."):
        ok, rows, log = _run("doc", [text], ["Riassunto."])
        assert not ok and log["reason"] == "third person"


def test_a_doc_request_in_english_is_rejected():
    ok, rows, log = _run("doc", ["Can you summarise this notice for me in two lines please?"], [])
    assert not ok and not rows and log["reason"] == "not Italian"


def test_each_pass_over_the_documents_asks_something_else():
    asked = []
    for i in (0, len(DOCS)):
        writer = Teacher(["Me lo riassumi in due righe?"])
        chats.user_turn("doc", i, writer, DOCS)
        asked.append(writer.seen[0][0]["content"].split(" e chiede ")[1].split(". Scrivi")[0])
    assert asked[0] != asked[1]


def test_a_failed_answer_is_retried_thinking_within_a_capped_budget():
    teacher = Teacher(["You crush basil, pine nuts and garlic with oil in a mortar.", ANSWER])
    ok, rows, _ = _run("doc", [REQUEST], [], teacher=teacher)
    assert ok and rows[0]["meta"]["attempt"] == 1 and teacher.clones[0]["thinking"] is False
    assert teacher.clones[1]["thinking"] is True and teacher.clones[1]["max_tokens"] == chats.THINK_TOKENS == 2048


def test_an_answer_not_in_italian_is_dropped():
    ok, rows, log = _run("doc", [REQUEST], ["You crush basil, pine nuts and garlic with oil in a mortar.",
                                      "You crush basil, pine nuts and garlic with oil in a mortar."])
    assert not ok and not rows and all("not Italian" in a["problems"] for a in log["attempts"])
