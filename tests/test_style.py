"""Style gate for training answers: light markdown, no emoji, short."""

from qita.worlds.style import conversation_problems, style_problems


def test_good_answers_pass():
    assert style_problems("Nella lista ci sono **latte**, uova e caffè.") == []
    assert style_problems("Ecco cosa manca:\n- testi della home\n- pagina contatti") == []


def test_each_rule():
    assert "emoji" in style_problems("Fatto! 😊")
    assert "heading" in style_problems("### Riepilogo\nTutto ok.")
    assert "table" in style_problems("| mese | totale |\n|---|---|\n| luglio | 3210 |")
    assert "rule" in style_problems("Prima parte\n---\nSeconda parte")
    assert "too much bold" in style_problems("**a** **b** **c** **d**")
    assert any("words" in w for w in style_problems("parola " * 61))
    assert style_problems("parola " * 90, "summarize") == []
    assert any("words" in w for w in style_problems("parola " * 26, before_call=True))


def test_conversation_checks_only_assistant_messages():
    msgs = [{"role": "user", "content": "### titolo 😊"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x", "arguments": {}}}]},
            {"role": "tool", "content": "| a | b |"},
            {"role": "assistant", "content": "Fatto."}]
    assert conversation_problems(msgs, "single") == []
    msgs[-1]["content"] = "Fatto ✅"
    assert conversation_problems(msgs, "single") == ["emoji"]
