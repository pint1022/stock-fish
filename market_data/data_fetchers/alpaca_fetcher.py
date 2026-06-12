# -*- coding: utf-8 -*-
"""
AlpacaFetcher - realtime US stock and crypto quotes.

Data source: Alpaca Market Data REST API
Markets: US stocks and crypto pairs
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import requests

from market_data.data_fetchers.base import BaseFetcher, DataFetchError
from market_data.data_fetchers.realtime_types import RealtimeSource, UnifiedRealtimeQuote, safe_float, safe_int
from market_data.data_fetchers.us_index_mapping import is_us_stock_code
from market_data.stock_index.stock_mapping import STOCK_NAME_MAP

logger = logging.getLogger(__name__)

_DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"
_DEFAULT_TRADING_BASE_URL = "https://api.alpaca.markets/v2"
_DEFAULT_STOCK_FEED = "iex"
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")


def normalize_alpaca_crypto_symbol(stock_code: str) -> Optional[str]:
    """Normalize common crypto inputs to Alpaca pair format, e.g. BTC-USD -> BTC/USD."""
    symbol = (stock_code or "").strip().upper()
    if not symbol:
        return None

    for separator in ("/", "-"):
        if separator in symbol:
            base, quote = symbol.split(separator, 1)
            if base.isalpha() and quote in _CRYPTO_QUOTES:
                return f"{base}/{quote}"
            return None

    for quote in _CRYPTO_QUOTES:
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            if base.isalpha() and 2 <= len(base) <= 10:
                return f"{base}/{quote}"

    return None


def _nested_symbol_payload(payload: Dict[str, Any], key: str, symbol: str) -> Dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        if symbol in value and isinstance(value[symbol], dict):
            return value[symbol]
        return value
    return {}


class AlpacaFetcher(BaseFetcher):
    name = "AlpacaFetcher"
    priority = 1

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        data_base_url: Optional[str] = None,
        trading_base_url: Optional[str] = None,
        stock_feed: Optional[str] = None,
    ):
        from market_data.compat import get_config

        config = get_config()
        self._api_key = (
            api_key
            or getattr(config, "alpaca_api_key", None)
            or os.getenv("ALPACA_API_KEY")
        )
        self._secret_key = (
            secret_key
            or getattr(config, "alpaca_secret_key", None)
            or os.getenv("ALPACA_SECRET_KEY")
        )
        legacy_base_url = (
            base_url
            or getattr(config, "alpaca_base_url", None)
            or os.getenv("ALPACA_BASE_URL")
        )
        configured_data_url = (
            data_base_url
            or getattr(config, "alpaca_data_base_url", None)
            or os.getenv("ALPACA_DATA_BASE_URL")
        )
        configured_trading_url = (
            trading_base_url
            or getattr(config, "alpaca_trading_base_url", None)
            or os.getenv("ALPACA_TRADING_BASE_URL")
        )

        self._data_base_url, self._trading_base_url = self._resolve_base_urls(
            data_url=configured_data_url,
            trading_url=configured_trading_url,
            legacy_url=legacy_base_url,
        )
        self._stock_feed = (
            stock_feed
            or getattr(config, "alpaca_stock_feed", None)
            or os.getenv("ALPACA_STOCK_FEED")
            or _DEFAULT_STOCK_FEED
        )
        self._stock_client = None
        if not self._api_key or not self._secret_key:
            logger.debug("[Alpaca] API credentials not configured, fetcher disabled")

    @staticmethod
    def _looks_like_trading_url(url: str) -> bool:
        normalized = (url or "").lower()
        return "paper-api.alpaca.markets" in normalized or "api.alpaca.markets" in normalized

    @staticmethod
    def _clean_optional_url(url: Optional[str]) -> str:
        cleaned = (url or "").strip()
        if cleaned.lower() in {"", "none", "null"}:
            return ""
        return cleaned

    @classmethod
    def _resolve_base_urls(
        cls,
        data_url: Optional[str],
        trading_url: Optional[str],
        legacy_url: Optional[str],
    ) -> tuple[str, str]:
        """Resolve separate Alpaca data/trading URLs while tolerating legacy config."""
        resolved_data = cls._clean_optional_url(data_url)
        resolved_trading = cls._clean_optional_url(trading_url)
        legacy = cls._clean_optional_url(legacy_url)

        if resolved_data and cls._looks_like_trading_url(resolved_data):
            if not resolved_trading:
                resolved_trading = resolved_data
            resolved_data = ""

        if legacy:
            if cls._looks_like_trading_url(legacy):
                resolved_trading = resolved_trading or legacy
            else:
                resolved_data = resolved_data or legacy

        return (
            (resolved_data or _DEFAULT_DATA_BASE_URL).rstrip("/"),
            (resolved_trading or _DEFAULT_TRADING_BASE_URL).rstrip("/"),
        )

    def is_available_for_request(self, capability: str = "") -> bool:
        if capability and capability != "realtime_quote":
            return False
        return bool(self._api_key and self._secret_key)

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise DataFetchError("[Alpaca] daily data is not implemented in StockFish")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        return df

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key or "",
            "APCA-API-SECRET-KEY": self._secret_key or "",
        }

    def _get_stock_client(self):
        if self._stock_client is None:
            from alpaca.data.historical.stock import StockHistoricalDataClient  # pyright: ignore[reportMissingImports]

            self._stock_client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
        return self._stock_client

    def _stock_data_feed(self):
        from alpaca.data.enums import DataFeed  # pyright: ignore[reportMissingImports]

        try:
            return DataFeed(self._stock_feed)
        except ValueError:
            logger.warning("[Alpaca] Unknown ALPACA_STOCK_FEED=%r, falling back to iex", self._stock_feed)
            return DataFeed.IEX

    def _get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        authenticate: bool = True,
    ) -> Dict[str, Any]:
        response = requests.get(
            f"{self._data_base_url}{path}",
            headers=self._headers() if authenticate else {},
            params=params or {},
            timeout=15,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            request_id = response.headers.get("X-Request-ID")
            body = response.text[:200] if response.text else ""
            raise DataFetchError(
                f"[Alpaca] HTTP {response.status_code}"
                f"{f' request_id={request_id}' if request_id else ''}"
                f"{f' body={body}' if body else ''}"
            ) from exc
        return response.json()

    @staticmethod
    def _field(obj: Any, *names: str) -> Any:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return obj.get(name)
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    @classmethod
    def _symbol_value(cls, response: Any, symbol: str) -> Any:
        if isinstance(response, dict):
            return response.get(symbol) or response.get(symbol.upper())
        data = getattr(response, "data", None)
        if isinstance(data, dict):
            return data.get(symbol) or data.get(symbol.upper())
        return None

    @classmethod
    def _bar_list(cls, response: Any, symbol: str) -> list[Any]:
        bars = cls._symbol_value(response, symbol)
        if bars is None:
            return []
        if isinstance(bars, list):
            return bars
        return [bars]

    @classmethod
    def _bars_from_dataframe(cls, response: Any, symbol: str) -> list[Dict[str, Any]]:
        df = getattr(response, "df", None)
        if df is None or getattr(df, "empty", True):
            return []

        stock_df = df
        try:
            if isinstance(df.index, pd.MultiIndex) and "symbol" in (df.index.names or []):
                stock_df = df.xs(symbol.upper(), level="symbol", drop_level=False)
            elif "symbol" in df.columns:
                stock_df = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
        except (KeyError, ValueError):
            return []

        if stock_df.empty:
            return []

        stock_df = stock_df.sort_index()
        bars = []
        for _, row in stock_df.iterrows():
            bars.append(
                {
                    "o": row.get("open"),
                    "h": row.get("high"),
                    "l": row.get("low"),
                    "c": row.get("close"),
                    "v": row.get("volume"),
                }
            )
        return bars

    def _get_stock_payloads(self, symbol: str) -> tuple[Dict[str, Any], Dict[str, Any], list[Dict[str, Any]]]:
        from alpaca.data.enums import DataFeed  # pyright: ignore[reportMissingImports]
        from alpaca.data.requests import StockLatestQuoteRequest  # pyright: ignore[reportMissingImports]

        client = self._get_stock_client()
        latest_quote = self._symbol_value(
            client.get_stock_latest_quote(
                StockLatestQuoteRequest(
                    symbol_or_symbols=[symbol],
                    feed=DataFeed.IEX,
                )
            ),
            symbol,
        )
        bid_price = safe_float(self._field(latest_quote, "bid_price", "bp"))
        ask_price = safe_float(self._field(latest_quote, "ask_price", "ap"))
        bid_size = safe_float(self._field(latest_quote, "bid_size", "bs"))
        ask_size = safe_float(self._field(latest_quote, "ask_size", "as"))
        timestamp = self._field(latest_quote, "timestamp", "t")
        midpoint = None
        if bid_price is not None and ask_price is not None and bid_price > 0 and ask_price > 0:
            midpoint = (bid_price + ask_price) / 2
        price = midpoint or ask_price or bid_price

        quote_payload = {
            "p": price,
            "t": str(timestamp) if timestamp is not None else None,
            "bp": bid_price,
            "bs": bid_size,
            "ap": ask_price,
            "as": ask_size,
            "s": ask_size or bid_size,
        }
        return quote_payload, {}, []

    def _get_crypto_payloads(self, symbol: str) -> tuple[Dict[str, Any], Dict[str, Any], list[Dict[str, Any]]]:
        latest_trade_payload = self._get_json(
            "/v1beta3/crypto/us/latest/trades",
            params={"symbols": symbol},
            authenticate=False,
        )
        latest_bar_payload = self._get_json(
            "/v1beta3/crypto/us/latest/bars",
            params={"symbols": symbol},
            authenticate=False,
        )
        previous_bars_payload = self._get_json(
            "/v1beta3/crypto/us/bars",
            params={"symbols": symbol, "timeframe": "1Day", "limit": 2},
            authenticate=False,
        )
        latest_trade = _nested_symbol_payload(latest_trade_payload, "trades", symbol)
        latest_bar = _nested_symbol_payload(latest_bar_payload, "bars", symbol)
        bars_value = previous_bars_payload.get("bars", {})
        if isinstance(bars_value, dict):
            previous_bars = bars_value.get(symbol, [])
        else:
            previous_bars = bars_value
        return latest_trade, latest_bar, previous_bars if isinstance(previous_bars, list) else []

    @staticmethod
    def _previous_close(previous_bars: list[Dict[str, Any]], current_close: Optional[float]) -> Optional[float]:
        closes = [safe_float(bar.get("c")) for bar in previous_bars if isinstance(bar, dict)]
        closes = [close for close in closes if close is not None and close > 0]
        if len(closes) >= 2:
            if current_close is not None and closes[-1] != current_close:
                return closes[-1]
            return closes[-2]
        if len(closes) == 1 and current_close is not None and closes[0] != current_close:
            return closes[0]
        return None

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        if not self.is_available_for_request("realtime_quote"):
            return None

        raw_symbol = (stock_code or "").strip().upper()
        crypto_symbol = normalize_alpaca_crypto_symbol(raw_symbol)
        is_crypto = crypto_symbol is not None

        if is_crypto:
            symbol = crypto_symbol
        elif is_us_stock_code(raw_symbol):
            symbol = raw_symbol
        else:
            return None

        try:
            if is_crypto:
                trade, bar, previous_bars = self._get_crypto_payloads(symbol)
            else:
                trade, bar, previous_bars = self._get_stock_payloads(symbol)
        except Exception as exc:
            logger.warning(f"[Alpaca] Realtime quote failed for {symbol}: {exc}")
            return None

        trade_price = safe_float(trade.get("p"))
        close_price = safe_float(bar.get("c"))
        price = trade_price or close_price
        if price is None or price <= 0:
            return None

        open_price = safe_float(bar.get("o"))
        high = safe_float(bar.get("h"))
        low = safe_float(bar.get("l"))
        volume = safe_float(bar.get("v")) if is_crypto else safe_int(bar.get("v"))
        prev_close = self._previous_close(previous_bars, close_price)

        change_amount = None
        change_pct = None
        amplitude = None
        if prev_close is not None and prev_close > 0:
            change_amount = price - prev_close
            change_pct = change_amount / prev_close * 100
            if high is not None and low is not None:
                amplitude = (high - low) / prev_close * 100

        # Build the unified quote from the selected market payload:
        # stocks come from _get_stock_payloads, crypto from _get_crypto_payloads.
        return UnifiedRealtimeQuote(
            code=symbol,
            name=STOCK_NAME_MAP.get(symbol, ""),
            source=RealtimeSource.ALPACA,
            price=price,
            timestamp=str(trade.get("t")) if trade.get("t") is not None else None,
            bid_price=safe_float(trade.get("bp")),
            bid_size=safe_float(trade.get("bs")),
            ask_price=safe_float(trade.get("ap")),
            ask_size=safe_float(trade.get("as")),
            change_pct=round(change_pct, 2) if change_pct is not None else None,
            change_amount=round(change_amount, 4) if change_amount is not None else None,
            volume=volume,
            amount=price * volume if volume is not None else None,
            volume_ratio=None,
            turnover_rate=None,
            amplitude=round(amplitude, 2) if amplitude is not None else None,
            open_price=open_price,
            high=high,
            low=low,
            pre_close=prev_close,
        )
