"""Training orchestration for the ETH/USDT forecasting model."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn, optim

try:  # pragma: no cover - prefer package-relative imports when available
    from .torch_compat import ensure_torch_classes
except ImportError:  # pragma: no cover - allow running as a script
    import sys

    package_root = Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))
    from torch_compat import ensure_torch_classes

ensure_torch_classes()

try:  # pragma: no cover - prefer package-relative imports when available
    from .data import CandleDataset, fetch_ethusdt_candles, prepare_dataset
    from .model import CandleTransformer
except ImportError:  # pragma: no cover - allow running as a script
    import sys

    package_root = Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))
    from data import CandleDataset, fetch_ethusdt_candles, prepare_dataset
    from model import CandleTransformer

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler("training.log")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
LOGGER.setLevel(logging.INFO)
CHECKPOINT_PATH = Path("checkpoints/trainer_state.pth")
METRICS_PATH = Path("checkpoints/metrics.json")
CHECKPOINT_PATH.parent.mkdir(exist_ok=True, parents=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    timeframe: str = "30m"
    history_limit: int = 1500
    window: int = 64
    batch_size: int = 32
    epochs: int = 100000
    lr: float = 1e-4
    weight_decay: float = 1e-5
    commission: float = 0.0006
    reward_scale: float = 10.0
    budget: float = 1000.0
    autosave_minutes: float = 10.0


@dataclass
class TrainingState:
    epoch: int = 0
    budget: float = 0.0
    profit: float = 0.0
    direction_accuracy: float = 0.0
    last_checkpoint_time: float = field(default_factory=time.time)


class CandleTrainer:
    """High level interface that manages long running training sessions."""

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()
        self.device = _device()
        self.model = CandleTransformer().to(self.device)
        self.criterion = nn.MSELoss()
        self.direction_loss = nn.BCEWithLogitsLoss()
        self.optimizer: optim.Optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.stop_event = threading.Event()
        self.training_thread: Optional[threading.Thread] = None
        self.state = TrainingState(budget=self.config.budget)
        self.metrics: Dict[str, float] = {}
        self.autosave_lock = threading.Lock()
        self.apply_config(self.config, reset_state=True)

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def apply_config(self, config: TrainingConfig, reset_state: bool = False) -> None:
        self.config = config
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        if reset_state:
            self.state = TrainingState(budget=self.config.budget)

    def start(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            LOGGER.warning("Training is already running")
            return
        self.stop_event.clear()
        self.training_thread = threading.Thread(target=self._train_loop, daemon=True)
        self.training_thread.start()

    def stop(self) -> None:
        if not self.training_thread:
            return
        self.stop_event.set()
        self.training_thread.join(timeout=30)
        self.save_checkpoint()
        LOGGER.info("Training stopped at epoch %s", self.state.epoch)

    def resume(self) -> None:
        self.load_checkpoint()
        self.start()

    # ------------------------------------------------------------------
    def _train_loop(self) -> None:
        LOGGER.info("Starting training loop on %s", self.device)
        autosave_interval = self.config.autosave_minutes * 60
        while not self.stop_event.is_set() and self.state.epoch < self.config.epochs:
            try:
                dataset = self._load_dataset()
            except Exception as exc:  # pragma: no cover - requires network
                LOGGER.exception("Failed to load dataset: %s", exc)
                time.sleep(5)
                continue

            loader = self._make_loader(dataset)
            epoch_profit, correct_direction, total_samples = 0.0, 0, 0
            trade_size_sum, trade_size_batches = 0.0, 0

            for batch in loader:
                (
                    _loss,
                    batch_return,
                    batch_correct,
                    batch_total,
                    avg_trade_size,
                ) = self._step(batch)

                growth = max(0.0, 1.0 + batch_return)
                previous_budget = self.state.budget
                self.state.budget = previous_budget * growth
                batch_profit = self.state.budget - previous_budget
                epoch_profit += batch_profit
                correct_direction += batch_correct
                total_samples += batch_total
                trade_size_sum += avg_trade_size
                trade_size_batches += 1

                if self.stop_event.is_set():
                    break

            self.state.epoch += 1
            self.state.profit = self.state.budget - self.config.budget
            if total_samples:
                self.state.direction_accuracy = correct_direction / total_samples

            avg_trade_size_epoch = trade_size_sum / trade_size_batches if trade_size_batches else 0.0

            self.metrics = {
                "epoch": self.state.epoch,
                "profit": self.state.profit,
                "budget": self.state.budget,
                "direction_accuracy": self.state.direction_accuracy,
                "epoch_profit": epoch_profit,
                "avg_trade_size": avg_trade_size_epoch,
            }

            LOGGER.info(
                "Epoch %s | Epoch PnL %.2f | Total profit %.2f | Budget %.2f | Direction accuracy %.3f | Avg trade %.3f",
                self.state.epoch,
                epoch_profit,
                self.state.profit,
                self.state.budget,
                self.state.direction_accuracy,
                avg_trade_size_epoch,
            )

            now = time.time()
            if now - self.state.last_checkpoint_time >= autosave_interval:
                self.save_checkpoint()

            if self.stop_event.is_set():
                break

        if not self.stop_event.is_set():
            self.save_checkpoint()

    # ------------------------------------------------------------------
    def _load_dataset(self) -> CandleDataset:
        df = fetch_ethusdt_candles(
            timeframe=self.config.timeframe,
            limit=self.config.history_limit,
        )
        return prepare_dataset(df, window=self.config.window)

    def _make_loader(self, dataset: CandleDataset):
        tensor_x = torch.from_numpy(dataset.features).to(self.device)
        tensor_y = torch.from_numpy(dataset.targets).to(self.device)
        current_close = torch.from_numpy(dataset.current_close).to(self.device)
        next_close = torch.from_numpy(dataset.next_close).to(self.device)
        ds = torch.utils.data.TensorDataset(tensor_x, tensor_y, current_close, next_close)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def _step(self, batch):
        inputs, targets, current_close, next_close = batch
        self.optimizer.zero_grad()
        preds, direction_logits, trade_logits = self.model(inputs)

        price_targets = targets[:, :5]
        direction_targets = (targets[:, 3] > targets[:, 0]).float().unsqueeze(-1)

        mse = self.criterion(preds[:, :5], price_targets)
        direction_loss = self.direction_loss(direction_logits, direction_targets)

        direction_prob = torch.sigmoid(direction_logits)
        trade_size = torch.sigmoid(trade_logits.squeeze(-1))
        trade_direction = (direction_prob - 0.5) * 2.0

        prev_close = current_close
        actual_close = next_close

        price_change = (actual_close - prev_close) / (prev_close + 1e-6)
        trade_return = trade_size * trade_direction.squeeze(-1) * price_change
        trade_return = trade_return - trade_size * self.config.commission
        reward_loss = -self.config.reward_scale * trade_return.mean()

        loss = mse + direction_loss + reward_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        batch_return = float(trade_return.mean().item())
        correct_direction = (direction_prob > 0.5).eq(direction_targets).sum()
        batch_total = direction_targets.size(0)
        avg_trade_size = float(trade_size.mean().item())
        return (
            loss.item(),
            batch_return,
            int(correct_direction),
            int(batch_total),
            avg_trade_size,
        )

    # ------------------------------------------------------------------
    def save_checkpoint(self) -> None:
        with self.autosave_lock:
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "state": self.state.__dict__,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            with METRICS_PATH.open("w") as fp:
                json.dump(self.metrics, fp, indent=2)
            self.state.last_checkpoint_time = time.time()
            LOGGER.info("Checkpoint saved to %s", CHECKPOINT_PATH)

    def load_checkpoint(self) -> None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError("No checkpoint found to resume from")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        cfg_dict = checkpoint.get("config", {})
        self.config = TrainingConfig(**{**self.config.__dict__, **cfg_dict})
        state_dict = checkpoint.get("state", {})
        self.state = TrainingState(**{**self.state.__dict__, **state_dict})
        LOGGER.info(
            "Loaded checkpoint at epoch %s with budget %.2f",
            self.state.epoch,
            self.state.budget,
        )

    def latest_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)
