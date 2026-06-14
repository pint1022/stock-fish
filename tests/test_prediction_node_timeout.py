import json
import sys
import types
import unittest
from types import SimpleNamespace

from analysis.nodes.prediction_node import PredictionNode


class FakeOpenAI:
    captured_kwargs = []

    def __init__(self, **kwargs):
        self.captured_kwargs.append(kwargs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    def _create_completion(self, **_kwargs):
        content = json.dumps({
            "outlook": "中性",
            "confidence": "低",
            "score": 0,
            "key_points": [],
            "risks": [],
            "analysis_text": "ok",
            "reason": "ok",
        })
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class PredictionNodeTimeoutTest(unittest.TestCase):
    def setUp(self):
        self.original_openai = sys.modules.get("openai")
        FakeOpenAI.captured_kwargs = []
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=FakeOpenAI)

    def tearDown(self):
        if self.original_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = self.original_openai

    def test_multi_agent_client_has_timeout_and_bounded_retries(self):
        node = PredictionNode(api_key="test-key", base_url="https://example.invalid/v1", model="test-model")

        node._multi_agent_predict({"quote": {"price": 1}, "symbol": "TEST"})

        self.assertTrue(FakeOpenAI.captured_kwargs)
        client_kwargs = FakeOpenAI.captured_kwargs[0]
        self.assertIn("timeout", client_kwargs)
        self.assertGreater(client_kwargs["timeout"], 0)
        self.assertEqual(client_kwargs.get("max_retries"), 1)


if __name__ == "__main__":
    unittest.main()
