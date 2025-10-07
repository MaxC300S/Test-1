# ETH/USDT AI Trainer

This project provides a Streamlit interface for training a transformer-based model that
forecasts ETH/USDT candle movements and simulates trading behaviour with a configurable
budget. The trainer automatically checkpoints state every 10 minutes (configurable) and
whenever you stop the session, ensuring you can pause and resume long-running experiments
without losing progress.

## Features

- **State-of-the-art transformer backbone** optimised to predict OHLCV values and trade
actions simultaneously.
- **Live ETH/USDT data download** from Binance using `ccxt`.
- **Reward shaping** that combines price prediction accuracy with simulated trading
performance including commission costs.
- **Streamlit dashboard** to start/stop/resume training, tune hyperparameters and review
budget, profit and accuracy metrics in real time.
- **Auto-checkpointing** every 10 minutes and on shutdown with manual save support.
- **Resume from checkpoint** at any time, even after closing the application.

## Getting started

1. Install dependencies (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. Pick your desired timeframe, budget and other hyperparameters from the sidebar, then
   press **Start training**.
4. Use **Stop training** to pause – training state is saved automatically.
5. Resume training with **Resume from checkpoint** or trigger **Manual checkpoint** at any
   time.

> **Note:** The trainer downloads live data from Binance. Ensure your environment allows
> outbound HTTPS connections. Training can run indefinitely (up to the configured epoch
> count) and is optimised for week-long sessions.

## Project structure

- `app/data.py` – Candle download and preprocessing helpers.
- `app/model.py` – Transformer architecture.
- `app/trainer.py` – Training loop, checkpointing and threading utilities.
- `app/streamlit_app.py` – Streamlit UI for controlling the trainer.
- `requirements.txt` – Runtime dependencies.

## Tests

A quick sanity check that all modules compile:
```bash
python -m compileall app
```
