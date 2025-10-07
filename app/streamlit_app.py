"""Interactive UI for training the ETH/USDT forecasting model."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

try:  # pragma: no cover - prefer package-relative imports when available
    from .trainer import CHECKPOINT_PATH, METRICS_PATH, CandleTrainer, TrainingConfig
except ImportError:  # pragma: no cover - allow running via "streamlit run app/streamlit_app.py"
    import sys

    package_root = Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))
    from trainer import CHECKPOINT_PATH, METRICS_PATH, CandleTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO)


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def init_session_state() -> None:
    if "trainer" not in st.session_state:
        st.session_state.trainer = CandleTrainer()
    if "config" not in st.session_state:
        st.session_state.config = st.session_state.trainer.config


def format_checkpoint(path: Path) -> str:
    if not path.exists():
        return "No checkpoint saved yet"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return f"Last checkpoint: {ts.isoformat()}"


def main() -> None:
    st.set_page_config(page_title="ETH/USDT AI Trainer", layout="wide")
    init_session_state()
    trainer: CandleTrainer = st.session_state.trainer
    config: TrainingConfig = st.session_state.config

    st.title("ETH/USDT Transformer Trainer")
    st.caption(
        "Start, pause, and monitor a state-of-the-art transformer that learns to forecast "
        "Ethereum price action and execute trades with a simulated budget."
    )

    with st.sidebar:
        st.header("Training Controls")
        with st.form("config_form"):
            st.subheader("Session Parameters")
            timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            try:
                timeframe_index = timeframes.index(config.timeframe)
            except ValueError:
                timeframe_index = timeframes.index("30m")
            timeframe = st.selectbox(
                "Timeframe",
                options=timeframes,
                index=timeframe_index,
            )
            history_limit = st.slider("Historical candles", 500, 10000, config.history_limit, step=100)
            window = st.slider("Context window", 16, 256, config.window, step=8)
            batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=config.batch_size)
            lr = st.number_input("Learning rate", value=config.lr, format="%.1e")
            weight_decay = st.number_input("Weight decay", value=config.weight_decay, format="%.1e")
            commission = st.number_input("Commission per trade", value=config.commission, format="%.4f")
            reward_scale = st.number_input("Reward scale", value=config.reward_scale, format="%.2f")
            budget = st.number_input("Simulated budget (USDT)", value=config.budget, step=100.0)
            autosave_minutes = st.number_input("Autosave frequency (minutes)", value=config.autosave_minutes, step=1.0)
            submitted = st.form_submit_button("Apply configuration")

        if submitted:
            config = TrainingConfig(
                timeframe=timeframe,
                history_limit=history_limit,
                window=window,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                commission=commission,
                reward_scale=reward_scale,
                budget=budget,
                autosave_minutes=autosave_minutes,
            )
            st.session_state.config = config
            trainer.apply_config(config, reset_state=True)
            st.success("Configuration updated")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        if col1.button("Start training", use_container_width=True):
            trainer.start()
            st.toast("Training started", icon="✅")
        if col2.button("Stop training", use_container_width=True):
            trainer.stop()
            st.toast("Training stopped and checkpoint saved", icon="🛑")
        if col3.button("Resume from checkpoint", use_container_width=True):
            try:
                trainer.resume()
                st.toast("Resumed from checkpoint", icon="▶️")
            except FileNotFoundError:
                st.error("No checkpoint found. Start training first.")

        st.markdown("---")
        st.caption(format_checkpoint(CHECKPOINT_PATH))
        if st.button("Manual checkpoint"):
            trainer.save_checkpoint()
            st.success("Checkpoint saved")

    metrics = trainer.latest_metrics() or _load_metrics()

    if metrics:
        col_budget, col_profit, col_accuracy, col_epoch = st.columns(4)
        col_budget.metric("Simulated budget (USDT)", f"{metrics.get('budget', 0):.2f}")
        col_profit.metric("Cumulative profit (USDT)", f"{metrics.get('profit', 0):.2f}")
        col_accuracy.metric(
            "Direction accuracy",
            f"{metrics.get('direction_accuracy', 0) * 100:.2f}%",
        )
        col_epoch.metric("Epoch", int(metrics.get("epoch", 0)))
    else:
        st.info("No training metrics available yet. Start training to populate this section.")

    st.markdown("---")
    st.subheader("Live training log")
    log_placeholder = st.empty()
    log_lines = []
    log_file = Path("training.log")
    if log_file.exists():
        log_lines = log_file.read_text().splitlines()[-200:]
    if log_lines:
        log_placeholder.code("\n".join(log_lines), language="text")
    else:
        st.write("Logs will appear here once training starts.")

    st.markdown("---")
    st.subheader("Quick start")
    st.markdown(
        "1. Adjust the configuration from the sidebar to match your trading horizon.\n"
        "2. Click **Start training** to begin downloading data and optimising the model.\n"
        "3. Use **Stop training** to pause – the session checkpoints automatically every 10 minutes\n"
        "   and whenever you stop.\n"
        "4. Resume at any time from the last checkpoint, even after closing the app."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
