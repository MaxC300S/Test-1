"""Utilities for downloading and preparing ETH/USDT candle data."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except Exception as exc:  # pragma: no cover - dependency import guard
    raise RuntimeError(
        "ccxt is required to fetch market data. Install dependencies from requirements.txt"
    ) from exc

LOGGER = logging.getLogger(__name__)


@dataclass
class CandleDataset:
    """Prepared dataset ready for training."""

    features: np.ndarray
    targets: np.ndarray
    current_close: np.ndarray


def fetch_ethusdt_candles(
    timeframe: str,
    limit: int = 1000,
    since: Optional[int] = None,
    rate_limit_sleep: float = 0.25,
) -> pd.DataFrame:
    """Download historical ETH/USDT candles from Binance.

    Parameters
    ----------
    timeframe: str
        Binance compatible timeframe (e.g. "30m", "1h").
    limit: int
        Number of candles to fetch per request (max 1500 per ccxt docs).
    since: Optional[int]
        Millisecond timestamp to start fetching from.
    rate_limit_sleep: float
        Sleep time between paginated requests to avoid rate limits.
    """

    exchange = ccxt.binance({"enableRateLimit": True})
    all_candles = []
    fetched = 0
    max_limit = min(limit, 1500)

    while fetched < limit:
        LOGGER.info("Fetching candles batch starting from %s", since)
        batch = exchange.fetch_ohlcv(
            symbol="ETH/USDT",
            timeframe=timeframe,
            since=since,
            limit=max_limit,
        )

        if not batch:
            break

        all_candles.extend(batch)
        fetched += len(batch)
        since = batch[-1][0] + 1
        if len(batch) < max_limit:
            break
        time.sleep(rate_limit_sleep)

    if not all_candles:
        raise RuntimeError("Failed to download ETH/USDT candles from Binance")

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df = df.sort_index()
    return df


def _normalise(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Normalise columns and return scaling factors for denormalisation."""

    scale = df.max() - df.min()
    scale.replace(0, 1.0, inplace=True)
    normalised = (df - df.min()) / scale
    return normalised, scale


def prepare_dataset(
    df: pd.DataFrame,
    window: int = 32,
) -> CandleDataset:
    """Prepare sliding window sequences for model training."""

    if len(df) <= window:
        raise ValueError("Not enough data to build training windows")

    features, targets, current_close = [], [], []
    normalised, scale = _normalise(df)

    values = normalised.values
    for idx in range(window, len(values)):
        window_slice = values[idx - window : idx]
        target_slice = values[idx]
        features.append(window_slice)
        targets.append(target_slice)
        current_close.append(df.iloc[idx - 1]["close"])

    features_arr = np.asarray(features, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    current_close_arr = np.asarray(current_close, dtype=np.float32)

    return CandleDataset(
        features=features_arr,
        targets=targets_arr,
        current_close=current_close_arr,
    )
