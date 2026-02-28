"""Tests for DocumentClassifier."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification import DocumentClassifier
from utils import EntityFeature


class FakeChatClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.last_args = None
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, outer):
            self.outer = outer

        def __call__(self, **kwargs):
            self.outer.last_args = kwargs
            content = json.dumps(self.outer.payload)
            message = SimpleNamespace(content=content)
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

        def complete(self, **kwargs):
            return self.__call__(**kwargs)


class FakeAgentClient(FakeChatClient):
    class Agents:
        def __init__(self, outer):
            self.outer = outer

        def complete(self, **kwargs):
            self.outer.last_agent_args = kwargs
            content = json.dumps(self.outer.payload)
            message = SimpleNamespace(content=content)
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    def __init__(self, payload: dict):
        super().__init__(payload)
        self.chat = self.Chat(self)
        self.agents = self.Agents(self)


class DocumentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [
            EntityFeature(
                entity_type="text_block",
                name="page_0_text",
                page_index=0,
                text_snippet="Net income increased",
            )
        ]

    def test_classify_parses_json_result(self) -> None:
        fake_client = FakeChatClient(
            {
                "risk_tier": "High",
                "rationale": "Negative cash flow",
                "confidence": 0.82,
            }
        )
        classifier = DocumentClassifier(client=fake_client)

        result = classifier.classify("doc-1", self.features)

        self.assertEqual(result.risk_tier, "High")
        self.assertAlmostEqual(result.confidence, 0.82)
        self.assertIn("messages", fake_client.last_args)
        self.assertEqual(fake_client.last_args["model"], "mistral-large-latest")

    def test_classify_requires_features(self) -> None:
        fake_client = FakeChatClient({})
        classifier = DocumentClassifier(client=fake_client)
        with self.assertRaises(ValueError):
            classifier.classify("doc-1", [])

    def test_invalid_json_raises(self) -> None:
        class BadClient(FakeChatClient):
            class Chat(FakeChatClient.Chat):
                def __call__(self, **kwargs):
                    self.outer.last_args = kwargs
                    message = SimpleNamespace(content="not-json")
                    choice = SimpleNamespace(message=message)
                    return SimpleNamespace(choices=[choice])

        classifier = DocumentClassifier(client=BadClient({}))
        with self.assertRaises(ValueError):
            classifier.classify("doc-1", self.features)

    def test_agent_path_invokes_agents_api(self) -> None:
        payload = {
            "risk_tier": "Low",
            "rationale": "Strong liquidity",
            "confidence": 0.9,
        }
        fake_client = FakeAgentClient(payload)
        classifier = DocumentClassifier(client=fake_client, agent_id="agent-123")

        result = classifier.classify("doc-2", self.features)

        self.assertEqual(result.risk_tier, "Low")
        self.assertEqual(fake_client.last_agent_args["agent_id"], "agent-123")


if __name__ == "__main__":
    unittest.main()
