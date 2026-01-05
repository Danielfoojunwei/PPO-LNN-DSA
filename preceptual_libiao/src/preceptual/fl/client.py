"""
Federated Learning Client

Flower-based FL client for site-level training
and weight updates to central server.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import flwr as fl
    from flwr.common import (
        Parameters,
        FitIns,
        FitRes,
        EvaluateIns,
        EvaluateRes,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False


@dataclass
class FLClientConfig:
    """Configuration for FL client."""
    # Server connection
    server_address: str = "localhost:8090"

    # Client identity
    client_id: str = "site_001"
    site_name: str = "Default Site"

    # Training
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-4

    # Data
    data_dir: str = "./telemetry"
    min_samples: int = 1000

    # Artifacts
    local_artifact_dir: str = "./local_artifacts"


class LocalTrainer:
    """
    Local training for FL client.

    Trains on site telemetry data.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: FLClientConfig,
    ):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

        # Training state
        self.epoch = 0
        self.total_samples = 0
        self.training_losses: List[float] = []

    def load_data(self) -> Optional[torch.utils.data.DataLoader]:
        """Load local telemetry data."""
        # Placeholder - would load actual telemetry recordings
        # For now, generate synthetic data
        num_samples = max(self.config.min_samples, 2000)

        global_obs = torch.randn(num_samples, 42)
        robot_obs = torch.randn(num_samples, 50, 14)
        rewards = torch.randn(num_samples)

        dataset = torch.utils.data.TensorDataset(global_obs, robot_obs, rewards)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self.total_samples = num_samples
        return loader

    def train(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Run local training.

        Args:
            epochs: Number of epochs (uses config if None)

        Returns:
            Training metrics
        """
        epochs = epochs or self.config.local_epochs
        loader = self.load_data()

        if loader is None:
            return {"error": "No data available"}

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in loader:
                global_obs, robot_obs, rewards = batch

                self.optimizer.zero_grad()

                # Forward pass
                _, _, value, _ = self.model.get_action(
                    global_obs, robot_obs, deterministic=True
                )

                # Simple value loss for pretraining
                loss = torch.nn.functional.mse_loss(value, rewards)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / (len(loader) or 1)
            self.training_losses.append(avg_epoch_loss)
            total_loss += avg_epoch_loss

            logger.debug(f"Epoch {epoch + 1}/{epochs}: loss={avg_epoch_loss:.4f}")

        self.epoch += epochs

        return {
            "loss": total_loss / epochs,
            "epochs": epochs,
            "samples": self.total_samples,
        }

    def get_weights(self) -> List[np.ndarray]:
        """Get model weights as numpy arrays."""
        return [
            param.detach().cpu().numpy()
            for param in self.model.parameters()
        ]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set model weights from numpy arrays."""
        for param, weight in zip(self.model.parameters(), weights):
            param.data = torch.from_numpy(weight).to(param.device)


if FLOWER_AVAILABLE:
    class PreceptualFlowerClient(fl.client.NumPyClient):
        """Flower client for Preceptual FL."""

        def __init__(
            self,
            trainer: LocalTrainer,
            config: FLClientConfig,
        ):
            self.trainer = trainer
            self.config = config

        def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
            """Get model parameters."""
            return self.trainer.get_weights()

        def fit(
            self,
            parameters: List[np.ndarray],
            config: Dict[str, Any],
        ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
            """
            Train on local data.

            Args:
                parameters: Global model parameters
                config: Training configuration

            Returns:
                (updated_parameters, num_samples, metrics)
            """
            # Set global weights
            self.trainer.set_weights(parameters)

            # Train locally
            epochs = config.get("local_epochs", self.config.local_epochs)
            metrics = self.trainer.train(epochs)

            # Return updated weights
            return (
                self.trainer.get_weights(),
                self.trainer.total_samples,
                metrics,
            )

        def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Dict[str, Any],
        ) -> Tuple[float, int, Dict[str, Scalar]]:
            """
            Evaluate model on local data.

            Args:
                parameters: Model parameters to evaluate
                config: Evaluation configuration

            Returns:
                (loss, num_samples, metrics)
            """
            # Set weights
            self.trainer.set_weights(parameters)

            # Evaluate
            loader = self.trainer.load_data()
            if loader is None:
                return 0.0, 0, {"error": "No data"}

            self.trainer.model.eval()
            total_loss = 0.0
            num_samples = 0

            with torch.no_grad():
                for batch in loader:
                    global_obs, robot_obs, rewards = batch

                    _, _, value, _ = self.trainer.model.get_action(
                        global_obs, robot_obs, deterministic=True
                    )

                    loss = torch.nn.functional.mse_loss(value, rewards)
                    total_loss += loss.item() * len(rewards)
                    num_samples += len(rewards)

            avg_loss = total_loss / max(1, num_samples)

            return avg_loss, num_samples, {"eval_loss": avg_loss}


class FLClient:
    """
    Federated Learning Client for Preceptual.

    Manages local training and communication with FL server.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[FLClientConfig] = None,
    ):
        self.config = config or FLClientConfig()
        self.model = model

        # Create trainer
        self.trainer = LocalTrainer(model, self.config)

        # Flower client
        if FLOWER_AVAILABLE:
            self.flower_client = PreceptualFlowerClient(self.trainer, self.config)
        else:
            self.flower_client = None

        # State
        self.is_connected = False

    def start(self) -> None:
        """Start FL client and connect to server."""
        if not FLOWER_AVAILABLE:
            logger.error("Cannot start FL client without Flower")
            return

        logger.info(f"Starting FL client {self.config.client_id}")
        logger.info(f"Connecting to {self.config.server_address}")

        try:
            fl.client.start_numpy_client(
                server_address=self.config.server_address,
                client=self.flower_client,
            )
        except Exception as e:
            logger.error(f"FL client error: {e}")
            raise

    def train_locally(self, epochs: int = 5) -> Dict[str, float]:
        """Run local training without FL server."""
        return self.trainer.train(epochs)

    def save_local_weights(self, path: str) -> None:
        """Save current weights locally."""
        weights = self.trainer.get_weights()
        np.savez(path, *weights)

    def load_local_weights(self, path: str) -> None:
        """Load weights from local file."""
        data = np.load(path)
        weights = [data[f"arr_{i}"] for i in range(len(data.files))]
        self.trainer.set_weights(weights)
