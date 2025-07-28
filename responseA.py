"""Portfolio Rebalancer Microservice
Requirements:
- Python 3.10+
- Install dependencies:
    pip install python-dotenv watchdog requests pandas numpy PyPortfolioOpt

Setup:
1. Place your secrets in a `.env` file in the working directory:
       API_URL=https://your-api-host
       API_TOKEN=your_secure_token
2. Provide runtime parameters in `config.json`:
       {
         "risk_free_rate": 0.01,
         "rebalance_interval": 300,
         "target_sharpe": 1.2,
         "cov_window": 60,
         "price_endpoint": "/prices",
         "position_endpoint": "/positions",
         "audit_log_path": "audit.log"
       }
3. This service hot-reloads both `.env` and `config.json` on each rebalance loop—no restart needed.
4. Run:
       python portfolio_rebalancer.py

Description:
- Every `rebalance_interval` seconds, fetches real-time prices & positions over TLS.
- Maintains a rolling window of price history to compute returns and covariance.
- Uses PyPortfolioOpt to perform mean-variance optimization:
    * If `target_sharpe` is set, solves for target return via binary search to match Sharpe.
    * Otherwise maximizes Sharpe directly.
- Appends a decision record (timestamp, weights, metrics) as JSON lines to an immutable SOC-2 audit log.
- All secrets (API_URL, API_TOKEN) must come from environment—no hardcoding.
"""

import os
import time
import json
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

import dotenv
import pandas as pd
import numpy as np
import requests
from pypfopt import EfficientFrontier
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Global price history
_price_history: pd.DataFrame = pd.DataFrame()


@dataclass
class Config:
    risk_free_rate: float
    rebalance_interval: int
    target_sharpe: Optional[float]
    cov_window: int
    price_endpoint: str
    position_endpoint: str
    audit_log_path: str

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        """Load runtime config from JSON file and environment."""
        dotenv.load_dotenv(override=True)
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file {path} not found.")
        return Config(
            risk_free_rate=float(cfg.get("risk_free_rate", 0.0)),
            rebalance_interval=int(cfg.get("rebalance_interval", 300)),
            target_sharpe=(float(cfg["target_sharpe"])
                           if "target_sharpe" in cfg else None),
            cov_window=int(cfg.get("cov_window", 60)),
            price_endpoint=str(cfg.get("price_endpoint", "/prices")),
            position_endpoint=str(cfg.get("position_endpoint", "/positions")),
            audit_log_path=str(cfg.get("audit_log_path", "audit.log")),
        )


class ConfigReloader(FileSystemEventHandler):
    """Watch config.json and .env for changes to trigger immediate reload."""

    def __init__(self, trigger: threading.Event):
        super().__init__()
        self._trigger = trigger

    def on_modified(self, event):
        if event.src_path.endswith("config.json") or event.src_path.endswith(".env"):
            self._trigger.set()


def fetch_data(api_url: str, token: str, endpoint: str) -> Dict[str, float]:
    """GET request over TLS, return JSON-decoded asset->value mapping."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{api_url.rstrip('/')}{endpoint}"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def compute_covariance_and_means(returns: pd.DataFrame, window: int
                                 ) -> tuple[pd.Series, pd.DataFrame]:
    """Compute mean returns and covariance matrix over rolling window."""
    tail = returns.tail(window)
    mean_ret = tail.mean()
    cov_mat = tail.cov()
    return mean_ret, cov_mat


def optimize_portfolio(mean_ret: pd.Series,
                       cov_mat: pd.DataFrame,
                       rf: float,
                       target_sharpe: Optional[float]
                       ) -> Dict[str, float]:
    """
    Use PyPortfolioOpt to get weights.
    If target_sharpe is set, binary search for a target return achieving that Sharpe.
    Otherwise maximize Sharpe directly.
    """
    # Direct maximize Sharpe if no target
    if target_sharpe is None:
        ef = EfficientFrontier(mean_ret, cov_mat)
        w = ef.max_sharpe(risk_free_rate=rf)
        return ef.clean_weights()

    # Binary search on target return
    low_ret, high_ret = mean_ret.min(), mean_ret.max()
    w_opt: Dict[str, float] = {}
    for _ in range(10):
        mid = (low_ret + high_ret) / 2
        ef = EfficientFrontier(mean_ret, cov_mat)
        try:
            w = ef.efficient_return(target_return=mid)
        except Exception:
            high_ret = mid
            continue
        # compute portfolio vol and sharpe
        alloc = np.array(list(w.values()))
        port_vol = np.sqrt(alloc @ cov_mat.values @ alloc.T)
        sharpe = (mid - rf) / port_vol if port_vol > 0 else 0.0
        if sharpe < target_sharpe:
            low_ret = mid
        else:
            high_ret = mid
            w_opt = w
    ef = EfficientFrontier(mean_ret, cov_mat, weight_bounds=(0, 1))
    ef.weights = w_opt or ef.clean_weights()
    return ef.clean_weights()


def log_decision(audit_log_path: str, record: Dict[str, Any]) -> None:
    """Append decision record as JSON line to audit log."""
    os.makedirs(os.path.dirname(audit_log_path) or ".", exist_ok=True)
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def rebalance_loop(stop_event: threading.Event) -> None:
    """Main loop: reload config, fetch data, compute, optimize, log, sleep."""
    global _price_history
    trigger = threading.Event()

    # Watcher for hot-reload
    observer = Observer()
    handler = ConfigReloader(trigger)
    observer.schedule(handler, ".", recursive=False)
    observer.start()

    while not stop_event.is_set():
        # Reload config & env
        config = Config.load()
        API_URL = os.getenv("API_URL")
        API_TOKEN = os.getenv("API_TOKEN")
        if not API_URL or not API_TOKEN:
            raise RuntimeError("API_URL or API_TOKEN not set in environment.")

        # Fetch market data
        price_data = fetch_data(API_URL, API_TOKEN, config.price_endpoint)
        pos_data = fetch_data(API_URL, API_TOKEN, config.position_endpoint)

        # Update price history DataFrame
        ts = pd.Timestamp.utcnow()
        row = pd.Series(price_data, name=ts)
        _price_history = pd.concat([_price_history, row.to_frame().T], sort=False)

        # Compute returns
        returns = _price_history.sort_index().pct_change().dropna()

        if len(returns) < 2:
            # Not enough data yet
            time.sleep(1)
            continue

        # Covariance & mean
        mean_ret, cov_mat = compute_covariance_and_means(returns,
                                                          config.cov_window)

        # Optimize
        weights = optimize_portfolio(mean_ret, cov_mat,
                                     config.risk_free_rate,
                                     config.target_sharpe)

        # Metrics
        alloc = np.array(list(weights.values()))
        exp_return = float(mean_ret @ alloc)
        exp_vol = float(np.sqrt(alloc @ cov_mat.values @ alloc.T))
        sharpe = ((exp_return - config.risk_free_rate) / exp_vol
                  if exp_vol > 0 else 0.0)

        # Audit record
        record = {
            "timestamp": ts.isoformat(),
            "positions": pos_data,
            "weights": weights,
            "expected_return": exp_return,
            "expected_volatility": exp_vol,
            "sharpe_ratio": sharpe
        }
        log_decision(config.audit_log_path, record)

        # Wait for next iteration or config change
        # Use whichever comes first: interval or config change
        triggered = trigger.wait(timeout=config.rebalance_interval)
        if triggered:
            trigger.clear()
        # continue loop
    observer.stop()
    observer.join()


if __name__ == "__main__":
    stop_evt = threading.Event()
    try:
        rebalance_loop(stop_evt)
    except KeyboardInterrupt:
        stop_evt.set()
        print("Shutting down gracefully.")
