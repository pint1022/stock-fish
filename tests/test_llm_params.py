import unittest
from types import SimpleNamespace
from unittest.mock import patch


class LLMParamsTests(unittest.TestCase):
    def test_default_temperature_is_omitted_for_restricted_models(self):
        from analysis.llm_params import temperature_kwargs

        with patch("config.settings", SimpleNamespace(LLM_TEMPERATURE=1)):
            self.assertEqual(temperature_kwargs(0.3), {})

    def test_non_default_temperature_is_sent_when_configured(self):
        from analysis.llm_params import temperature_kwargs

        with patch("config.settings", SimpleNamespace(LLM_TEMPERATURE=0.2)):
            self.assertEqual(temperature_kwargs(0.3), {"temperature": 0.2})

    def test_call_default_is_used_when_no_global_temperature_configured(self):
        from analysis.llm_params import temperature_kwargs

        with patch("config.settings", SimpleNamespace()):
            self.assertEqual(temperature_kwargs(0.3), {"temperature": 0.3})


if __name__ == "__main__":
    unittest.main()
