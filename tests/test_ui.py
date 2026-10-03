from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src.providers.llm import CompletionResult

APP_PATH = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


class Stubs:
    def __init__(self):
        self.planner_calls = []
        self.generate_calls = []
        self.safety_blocks = {"bad request"}
        self.safety_error = None

    def safety(self, message):
        if self.safety_error:
            raise self.safety_error
        if message in self.safety_blocks:
            return False, "I can't help with that request."
        return True, None

    def planner(self, messages, **kwargs):
        self.planner_calls.append(messages)
        return CompletionResult(content="how do I scale a Deployment?", provider="nim", model="m")

    def generate(self, messages, **kwargs):
        self.generate_calls.append(messages)
        return CompletionResult(content="Here is a grounded answer.", provider="groq", model="m")


@pytest.fixture
def stubs():
    stub = Stubs()
    with ExitStack() as stack:
        stack.enter_context(patch("src.retrieval.rerank.preload"))
        stack.enter_context(patch("src.providers.clients.qdrant_client.count", side_effect=Exception("offline")))
        stack.enter_context(patch("src.graph.exact_cache_get", return_value=None))
        stack.enter_context(patch("src.graph.safety_gate", side_effect=stub.safety))
        stack.enter_context(patch("src.graph.topic_gate", return_value=(True, None)))
        stack.enter_context(patch("src.graph.response_safety_gate", return_value=(True, None)))
        stack.enter_context(patch("src.graph.generate_planner", side_effect=stub.planner))
        stack.enter_context(patch("src.graph.generate_main", side_effect=stub.generate))
        stack.enter_context(patch("src.graph.embed_canonical_question", return_value=[0.1] * 4))
        stack.enter_context(patch("src.graph.semantic_cache_get", return_value=None))
        stack.enter_context(
            patch("src.graph.retrieve", return_value=[{"text": "ctx", "metadata": {"source_path": "true_data/a.md"}, "retrieval_score": 0.9}])
        )
        stack.enter_context(
            patch(
                "src.graph.rerank_and_gate",
                return_value=[{"text": "ctx", "metadata": {"source_path": "true_data/<x>.md", "manifest_kind": "Pod", "manifest_name": "web"}, "rerank_score": 0.9}],
            )
        )
        stack.enter_context(patch("src.graph._submit_cache_write"))
        yield stub


def new_app():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    return app


def page_text(app):
    return " ".join(element.value for element in app.markdown)


def ask(app, question):
    app.chat_input[0].set_value(question).run()
    assert not app.exception
    return app


def test_landing_page_renders_with_all_providers_offline(stubs):
    app = new_app()
    assert len(app.chat_input) == 1
    assert "Ask about" in page_text(app)


def test_question_gets_a_grounded_answer(stubs):
    app = ask(new_app(), "what is a pod")
    assert "Here is a grounded answer." in page_text(app)
    assert [turn["role"] for turn in app.session_state.history] == ["user", "assistant"]


def test_interrupted_turn_is_completed_from_history_not_a_none_prompt(stubs):
    app = new_app()
    app.session_state.history = [{"role": "user", "content": "what is a pod"}]
    app.run()
    assert not app.exception
    assert [turn["role"] for turn in app.session_state.history] == ["user", "assistant"]
    assert "Here is a grounded answer." in app.session_state.history[-1]["content"]


def test_widget_interaction_after_an_answer_does_not_rerun_the_turn(stubs):
    app = ask(new_app(), "what is a pod")
    calls_before = len(stubs.generate_calls)
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert len(stubs.generate_calls) == calls_before
    assert len(app.session_state.history) == 2


def test_refused_turn_is_excluded_from_the_next_turns_history(stubs):
    app = ask(new_app(), "bad request")
    assert "can't help" in page_text(app)
    ask(app, "what is a pod")
    assert stubs.planner_calls == []
    assert len(stubs.generate_calls) == 1


def test_answered_turn_feeds_the_next_turns_rewrite(stubs):
    app = ask(new_app(), "what is a Deployment?")
    ask(app, "how do I scale it?")
    assert len(stubs.planner_calls) == 1
    assert "what is a Deployment?" in stubs.planner_calls[0][1]["content"]


def test_greeting_is_answered_without_any_model_call(stubs):
    app = ask(new_app(), "hello")
    assert "Ask me a Kubernetes question" in page_text(app)
    assert stubs.generate_calls == [] and stubs.planner_calls == []


def test_pipeline_error_is_escaped_not_rendered_as_html(stubs):
    stubs.safety_error = RuntimeError("<b>injected</b>")
    app = ask(new_app(), "what is a pod")
    app.toggle[0].set_value(True).run()
    text = page_text(app)
    assert "<b>injected</b>" not in text
    assert "&lt;b&gt;injected&lt;/b&gt;" in text


def test_source_paths_are_escaped_and_show_the_manifest(stubs):
    app = ask(new_app(), "what is a pod")
    app.toggle[0].set_value(True).run()
    text = page_text(app)
    assert "true_data/<x>.md" not in text
    assert "true_data/&lt;x&gt;.md" in text
    assert "Pod/web" in text


def test_new_chat_clears_the_conversation(stubs):
    app = ask(new_app(), "what is a pod")
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.history == []
