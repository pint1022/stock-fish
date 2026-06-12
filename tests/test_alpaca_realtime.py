import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from market_data.data_fetchers.alpaca_fetcher import (
    AlpacaFetcher,
    normalize_alpaca_crypto_symbol,
)
from market_data.data_fetchers.base import DataFetcherManager
from market_data.data_fetchers.realtime_types import RealtimeSource, UnifiedRealtimeQuote


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class AlpacaFetcherTests(unittest.TestCase):
    def get_realtime_quote_2(self):
        from alpaca.data.enums import DataFeed  # pyright: ignore[reportMissingImports]
        from alpaca.data.historical.stock import StockHistoricalDataClient  # pyright: ignore[reportMissingImports]
        from alpaca.data.requests import StockLatestQuoteRequest  # pyright: ignore[reportMissingImports]
        from config import settings

        api_key = settings.ALPACA_API_KEY
        secret_key = settings.ALPACA_SECRET_KEY
        data_api_url = None

        stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key, url_override=data_api_url)
        request = StockLatestQuoteRequest(
            symbol_or_symbols=["AAPL", "TSLA", "NVDA"],
            feed=DataFeed.IEX,
        )
        return stock_historical_data_client.get_stock_latest_quote(request)

    def test_get_realtime_quote_2_fetches_latest_quotes(self):
        quotes = {"AAPL": SimpleNamespace(ask_price=182.32, bid_price=182.30)}
        captured_requests = []

        def fake_get_stock_latest_quote(request):
            captured_requests.append(request)
            return quotes

        fake_client = SimpleNamespace(get_stock_latest_quote=fake_get_stock_latest_quote)
        fake_settings = SimpleNamespace(
            ALPACA_API_KEY="key",
            ALPACA_SECRET_KEY="secret",
            ALPACA_DATA_BASE_URL="None",
        )

        with (
            patch("config.settings", fake_settings),
            patch("alpaca.data.historical.stock.StockHistoricalDataClient", return_value=fake_client) as mock_client,
        ):
            result = self.get_realtime_quote_2()

        self.assertEqual(result, quotes)
        self.assertEqual(mock_client.call_args.args, ("key", "secret"))
        self.assertIsNone(mock_client.call_args.kwargs["url_override"])
        self.assertEqual(len(captured_requests), 1)
        request = captured_requests[0]
        self.assertEqual(request.symbol_or_symbols, ["AAPL", "TSLA", "NVDA"])
        self.assertEqual(getattr(request.feed, "value", None), "iex")

    def test_stock_latest_trade_returns_unified_quote(self):
        fetcher = AlpacaFetcher(api_key="key", secret_key="secret")
        captured_quote_requests = []

        def fake_get_stock_latest_quote(request):
            captured_quote_requests.append(request)
            return {
                "AAPL": SimpleNamespace(
                    ask_price=182.32,
                    bid_price=182.30,
                    ask_size=100,
                    bid_size=90,
                    timestamp=pd.Timestamp("2026-06-10 16:00:01"),
                )
            }

        fake_client = SimpleNamespace(
            get_stock_latest_quote=fake_get_stock_latest_quote,
        )

        with patch(
            "alpaca.data.historical.stock.StockHistoricalDataClient",
            return_value=fake_client,
        ) as mock_client:

            quote = fetcher.get_realtime_quote("aapl")

        self.assertIsNotNone(quote)
        self.assertEqual(quote.code, "AAPL")
        self.assertEqual(quote.source, RealtimeSource.ALPACA)
        self.assertEqual(quote.price, 182.31)
        self.assertEqual(quote.bid_price, 182.30)
        self.assertEqual(quote.bid_size, 90)
        self.assertEqual(quote.ask_price, 182.32)
        self.assertEqual(quote.ask_size, 100)
        self.assertEqual(quote.timestamp, "2026-06-10 16:00:01")
        self.assertIsNone(quote.open_price)
        self.assertIsNone(quote.high)
        self.assertIsNone(quote.low)
        self.assertIsNone(quote.volume)
        self.assertIsNone(quote.pre_close)
        self.assertIsNone(quote.change_amount)
        self.assertIsNone(quote.change_pct)
        self.assertEqual(len(captured_quote_requests), 1)
        self.assertEqual(captured_quote_requests[0].symbol_or_symbols, ["AAPL"])
        self.assertEqual(getattr(captured_quote_requests[0].feed, "value", None), "iex")
        self.assertEqual(mock_client.call_args.kwargs["api_key"], "key")
        self.assertEqual(mock_client.call_args.kwargs["secret_key"], "secret")
        self.assertNotIn("url_override", mock_client.call_args.kwargs)

    def test_get_stock_payloads_reads_latest_quote_directly(self):
        fetcher = AlpacaFetcher(api_key="key", secret_key="secret")
        captured_quote_requests = []

        def fake_get_stock_latest_quote(request):
            captured_quote_requests.append(request)
            return {
                "AAPL": SimpleNamespace(
                    ask_price=182.32,
                    bid_price=182.30,
                    ask_size=100,
                    bid_size=90,
                    timestamp=pd.Timestamp("2026-06-10 16:00:01"),
                )
            }

        fake_client = SimpleNamespace(get_stock_latest_quote=fake_get_stock_latest_quote)

        with patch(
            "alpaca.data.historical.stock.StockHistoricalDataClient",
            return_value=fake_client,
        ):
            quote_payload, bar_payload, previous_bar_payloads = fetcher._get_stock_payloads("AAPL")

        self.assertEqual(quote_payload["p"], 182.31)
        self.assertEqual(quote_payload["bp"], 182.30)
        self.assertEqual(quote_payload["bs"], 90)
        self.assertEqual(quote_payload["ap"], 182.32)
        self.assertEqual(quote_payload["as"], 100)
        self.assertEqual(quote_payload["s"], 100)
        self.assertEqual(quote_payload["t"], "2026-06-10 16:00:01")
        self.assertEqual(bar_payload, {})
        self.assertEqual(previous_bar_payloads, [])
        self.assertEqual(len(captured_quote_requests), 1)
        self.assertEqual(captured_quote_requests[0].symbol_or_symbols, ["AAPL"])
        self.assertEqual(getattr(captured_quote_requests[0].feed, "value", None), "iex")

    def test_get_daily_data_fetches_stock_bars(self):
        fetcher = AlpacaFetcher(api_key="key", secret_key="secret")
        captured_bar_requests = []

        def fake_get_stock_bars(request):
            captured_bar_requests.append(request)
            return {
                "MU": [
                    SimpleNamespace(
                        timestamp=pd.Timestamp("2026-06-09 00:00:00"),
                        open=123.0,
                        high=125.0,
                        low=122.0,
                        close=124.0,
                        volume=1000,
                        trade_count=10,
                        vwap=123.5,
                    ),
                    SimpleNamespace(
                        timestamp=pd.Timestamp("2026-06-10 00:00:00"),
                        open=124.0,
                        high=127.0,
                        low=123.0,
                        close=126.0,
                        volume=1200,
                        trade_count=12,
                        vwap=125.5,
                    ),
                ]
            }

        fake_client = SimpleNamespace(get_stock_bars=fake_get_stock_bars)

        with patch(
            "alpaca.data.historical.stock.StockHistoricalDataClient",
            return_value=fake_client,
        ):
            df = fetcher.get_daily_data(
                "MU",
                start_date="2026-06-09",
                end_date="2026-06-10",
            )

        self.assertEqual(len(captured_bar_requests), 1)
        request = captured_bar_requests[0]
        self.assertEqual(request.symbol_or_symbols, ["MU"])
        self.assertEqual(getattr(request.feed, "value", None), "iex")
        self.assertEqual(list(df["close"]), [124.0, 126.0])
        self.assertEqual(list(df["volume"]), [1000, 1200])
        self.assertIn("pct_chg", df.columns)

    @unittest.skipUnless(
        os.getenv("RUN_ALPACA_LIVE_TESTS") == "1",
        "set RUN_ALPACA_LIVE_TESTS=1 to call live Alpaca authentication",
    )
    def test_get_stock_payloads_live_authentication(self):
        fetcher = AlpacaFetcher()

        quote_payload, bar_payload, previous_bar_payloads = fetcher._get_stock_payloads("AAPL")

        self.assertIsInstance(quote_payload, dict)
        self.assertGreater(quote_payload.get("p") or 0, 0)
        self.assertTrue(
            quote_payload.get("bp") is not None or quote_payload.get("ap") is not None,
            quote_payload,
        )
        self.assertEqual(bar_payload, {})
        self.assertEqual(previous_bar_payloads, [])

    def test_get_realtime_quote_routes_stock_to_stock_payloads(self):
        fetcher = AlpacaFetcher(api_key="key", secret_key="secret")
        stock_payload = (
            {
                "p": 182.31,
                "t": "2026-06-10 16:00:01",
                "bp": 182.30,
                "bs": 90,
                "ap": 182.32,
                "as": 100,
                "s": 100,
            },
            {},
            [],
        )

        with (
            patch.object(fetcher, "_get_stock_payloads", return_value=stock_payload) as mock_stock_payloads,
            patch.object(fetcher, "_get_crypto_payloads") as mock_crypto_payloads,
        ):
            quote = fetcher.get_realtime_quote("AAPL")

        mock_stock_payloads.assert_called_once_with("AAPL")
        mock_crypto_payloads.assert_not_called()
        self.assertIsInstance(quote, UnifiedRealtimeQuote)
        self.assertEqual(quote.code, "AAPL")
        self.assertEqual(quote.source, RealtimeSource.ALPACA)
        self.assertEqual(quote.price, 182.31)
        self.assertEqual(quote.timestamp, "2026-06-10 16:00:01")
        self.assertEqual(quote.bid_price, 182.30)
        self.assertEqual(quote.bid_size, 90)
        self.assertEqual(quote.ask_price, 182.32)
        self.assertEqual(quote.ask_size, 100)

    def test_trading_api_url_is_not_used_for_market_data_quotes(self):
        fetcher = AlpacaFetcher(
            api_key="key",
            secret_key="secret",
            base_url="https://paper-api.alpaca.markets/v2",
        )

        fake_client = SimpleNamespace(
            get_stock_latest_quote=lambda request: {"AAPL": SimpleNamespace(ask_price=182.31, bid_price=182.31)},
        )

        with patch(
            "alpaca.data.historical.stock.StockHistoricalDataClient",
            return_value=fake_client,
        ) as mock_client:

            quote = fetcher.get_realtime_quote("AAPL")

        self.assertIsNotNone(quote)
        self.assertNotIn("url_override", mock_client.call_args.kwargs)

    def test_literal_none_urls_use_alpaca_py_defaults(self):
        fetcher = AlpacaFetcher(
            api_key="key",
            secret_key="secret",
            data_base_url="None",
            trading_base_url="None",
        )
        fake_client = SimpleNamespace(
            get_stock_latest_quote=lambda request: {"AAPL": SimpleNamespace(ask_price=182.31, bid_price=182.31)},
        )

        with patch(
            "alpaca.data.historical.stock.StockHistoricalDataClient",
            return_value=fake_client,
        ) as mock_client:
            quote = fetcher.get_realtime_quote("AAPL")

        self.assertIsNotNone(quote)
        self.assertEqual(fetcher._data_base_url, "https://data.alpaca.markets")
        self.assertEqual(fetcher._trading_base_url, "https://api.alpaca.markets/v2")
        self.assertNotIn("url_override", mock_client.call_args.kwargs)

    def test_crypto_symbols_normalize_to_alpaca_pair_format(self):
        self.assertEqual(normalize_alpaca_crypto_symbol("BTC/USD"), "BTC/USD")
        self.assertEqual(normalize_alpaca_crypto_symbol("btc-usd"), "BTC/USD")
        self.assertEqual(normalize_alpaca_crypto_symbol("BTCUSD"), "BTC/USD")
        self.assertIsNone(normalize_alpaca_crypto_symbol("AAPL"))

    def test_crypto_quote_requests_do_not_send_auth_headers(self):
        fetcher = AlpacaFetcher(api_key="bad-key", secret_key="bad-secret")

        with patch("market_data.data_fetchers.alpaca_fetcher.requests.get") as mock_get:
            mock_get.side_effect = [
                FakeResponse({"trades": {"BTC/USD": {"p": 63481.58, "s": 0.000084}}}),
                FakeResponse({"bars": {"BTC/USD": {"o": 63000.0, "h": 64000.0, "l": 62500.0, "c": 63481.58, "v": 12.5}}}),
                FakeResponse({"bars": {"BTC/USD": [{"c": 63200.0}]}}),
            ]

            quote = fetcher.get_realtime_quote("BTC/USD")

        self.assertIsNotNone(quote)
        self.assertEqual(quote.source, RealtimeSource.ALPACA)
        self.assertEqual(quote.price, 63481.58)
        self.assertEqual(quote.volume, 12.5)
        for call in mock_get.call_args_list:
            self.assertEqual(call.kwargs["headers"], {})

    def test_manager_uses_alpaca_first_for_us_when_priority_requests_it(self):
        calls = []

        class FakeAlpaca:
            name = "AlpacaFetcher"
            priority = 1

            def get_realtime_quote(self, stock_code):
                calls.append(("alpaca", stock_code))
                return UnifiedRealtimeQuote(
                    code=stock_code,
                    source=RealtimeSource.ALPACA,
                    price=10.0,
                    volume_ratio=1.0,
                    turnover_rate=0.1,
                    pe_ratio=20.0,
                    pb_ratio=3.0,
                    total_mv=1000000.0,
                    circ_mv=900000.0,
                    amplitude=1.0,
                )

        class FakeYfinance:
            name = "YfinanceFetcher"
            priority = 2

            def get_realtime_quote(self, stock_code):
                calls.append(("yfinance", stock_code))
                return UnifiedRealtimeQuote(
                    code=stock_code,
                    source=RealtimeSource.FALLBACK,
                    price=9.0,
                )

        manager = DataFetcherManager(fetchers=[FakeAlpaca(), FakeYfinance()])

        with patch(
            "market_data.compat.get_config",
            return_value=SimpleNamespace(
                enable_realtime_quote=True,
                realtime_source_priority="alpaca,yfinance",
            ),
        ):
            quote = manager.get_realtime_quote("AAPL")

        self.assertEqual(quote.source, RealtimeSource.ALPACA)
        self.assertEqual(calls, [("alpaca", "AAPL")])

    def test_manager_routes_us_daily_data_to_alpaca_before_yfinance(self):
        calls = []
        expected = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-10"]),
                "open": [10.0],
                "high": [11.0],
                "low": [9.5],
                "close": [10.5],
                "volume": [1000],
                "amount": [10500.0],
                "pct_chg": [0.0],
            }
        )

        class FakeAlpaca:
            name = "AlpacaFetcher"
            priority = 0

            def get_daily_data(self, **kwargs):
                calls.append(("alpaca", kwargs["stock_code"]))
                return expected

            def is_available_for_request(self, capability=""):
                return capability == "daily_data"

        class FakeAlphaVantage:
            name = "AlphaVantageFetcher"
            priority = 3

            def get_daily_data(self, **kwargs):
                raise AssertionError("AlphaVantage should not be used for US daily data")

            def is_available_for_request(self, capability=""):
                return capability == "daily_data"

        class FakeYfinance:
            name = "YfinanceFetcher"
            priority = 1

            def get_daily_data(self, **kwargs):
                calls.append(("yfinance", kwargs["stock_code"]))
                return expected

            def is_available_for_request(self, capability=""):
                return capability == "daily_data"

        manager = DataFetcherManager(fetchers=[FakeAlphaVantage(), FakeYfinance(), FakeAlpaca()])
        manager._fundamental_adapter = Mock()
        manager._yfinance_fundamental_adapter = Mock()

        df, source = manager.get_daily_data(
            "MU",
            start_date="2026-06-10",
            end_date="2026-06-10",
        )

        self.assertEqual(source, "AlpacaFetcher")
        self.assertEqual(calls, [("alpaca", "MU")])
        self.assertTrue(df.equals(expected))


if __name__ == "__main__":
    unittest.main()
