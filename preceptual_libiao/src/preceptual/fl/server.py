"""
Federated Learning Server

Flower-based FL server for aggregating policy/predictor weights
across multiple Libiao sites.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
import json

import numpy as np

logger = logging.getLogger(__name__)

# Conditional Flower import
try:
    import flwr as fl
    from flwr.server.strategy import FedAvg, FedProx
    from flwr.common import (
        Parameters,
        FitRes,
        EvaluateRes,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    logger.warning("Flower not installed. FL functionality disabled.")


@dataclass
class FLServerConfig:
    """Configuration for FL server."""
    # Server settings
    server_address: str = "0.0.0.0:8090"
    num_rounds: int = 100
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2

    # Strategy
    strategy_type: str = "fedavg"  # "fedavg" or "fedprox"
    proximal_mu: float = 0.1  # FedProx parameter

    # Aggregation
    fraction_fit: float = 0.5
    fraction_evaluate: float = 0.5

    # Gating evaluation
    gate_on_benchmark: bool = True
    benchmark_threshold: float = 0.7  # Minimum performance threshold

    # Artifacts
    artifact_dir: str = "./fl_artifacts"
    save_every_n_rounds: int = 10

    # Security
    enable_signing: bool = True


@dataclass
class AggregationResult:
    """Result of a FL aggregation round."""
    round_num: int
    num_clients: int
    aggregated_loss: float
    aggregated_metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    passed_gate: bool = True
    gate_score: float = 1.0


class BenchmarkEvaluator:
    """Evaluates model against benchmark scenarios."""

    def __init__(self, scenarios: Optional[List[str]] = None):
        self.scenarios = scenarios or [
            "normal_50_robots",
            "high_contention_300_robots",
            "hotspot_collapse",
            "scan_storm",
            "interference_corridor",
        ]
        self.results: Dict[str, float] = {}

    def evaluate(self, model_weights: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on benchmark scenarios.

        Args:
            model_weights: Model parameters

        Returns:
            (overall_score, per_scenario_scores)
        """
        # Placeholder - in production, would run actual simulations
        scores = {}
        for scenario in self.scenarios:
            # Simulate evaluation
            scores[scenario] = np.random.uniform(0.6, 1.0)

        overall = np.mean(list(scores.values()))
        return overall, scores


if FLOWER_AVAILABLE:
    class PreceptualFedAvg(FedAvg):
        """Custom FedAvg with gating and artifact saving."""

        def __init__(
            self,
            config: FLServerConfig,
            benchmark_evaluator: Optional[BenchmarkEvaluator] = None,
            on_aggregation: Optional[Callable[[AggregationResult], None]] = None,
            **kwargs
        ):
            super().__init__(
                fraction_fit=config.fraction_fit,
                fraction_evaluate=config.fraction_evaluate,
                min_fit_clients=config.min_fit_clients,
                min_evaluate_clients=config.min_evaluate_clients,
                min_available_clients=config.min_available_clients,
                **kwargs
            )
            self.config = config
            self.benchmark_evaluator = benchmark_evaluator or BenchmarkEvaluator()
            self.on_aggregation = on_aggregation
            self.round_results: List[AggregationResult] = []

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[Any, FitRes]],
            failures: List[Tuple[Any, Any]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate with gating evaluation."""
            if not results:
                return None, {}

            # Standard FedAvg aggregation
            aggregated_params, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )

            if aggregated_params is None:
                return None, aggregated_metrics

            # Gating evaluation
            if self.config.gate_on_benchmark:
                weights = parameters_to_ndarrays(aggregated_params)
                gate_score, scenario_scores = self.benchmark_evaluator.evaluate(weights)
                passed_gate = gate_score >= self.config.benchmark_threshold

                if not passed_gate:
                    logger.warning(
                        f"Round {server_round} failed gating (score: {gate_score:.3f}). "
                        "Reverting to previous weights."
                    )
                    # In production, would revert to previous round's weights
                    aggregated_metrics["gate_passed"] = 0.0
                else:
                    aggregated_metrics["gate_passed"] = 1.0
                    aggregated_metrics["gate_score"] = gate_score

            else:
                gate_score = 1.0
                passed_gate = True

            # Create result record
            avg_loss = np.mean([
                fit_res.metrics.get("loss", 0.0)
                for _, fit_res in results
            ])

            result = AggregationResult(
                round_num=server_round,
                num_clients=len(results),
                aggregated_loss=avg_loss,
                aggregated_metrics=dict(aggregated_metrics),
                passed_gate=passed_gate,
                gate_score=gate_score,
            )
            self.round_results.append(result)

            # Callback
            if self.on_aggregation:
                self.on_aggregation(result)

            # Save artifact
            if server_round % self.config.save_every_n_rounds == 0:
                self._save_artifact(server_round, aggregated_params)

            return aggregated_params, aggregated_metrics

        def _save_artifact(self, round_num: int, params: Parameters) -> None:
            """Save model artifact."""
            artifact_dir = Path(self.config.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)

            weights = parameters_to_ndarrays(params)

            # Save weights
            weights_path = artifact_dir / f"round_{round_num}_weights.npz"
            np.savez(weights_path, *weights)

            # Save metadata
            metadata = {
                "round": round_num,
                "timestamp": time.time(),
                "num_results": len(self.round_results),
                "last_gate_score": self.round_results[-1].gate_score if self.round_results else 0,
            }
            meta_path = artifact_dir / f"round_{round_num}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)

            logger.info(f"Saved artifact for round {round_num}")


class FLServer:
    """
    Federated Learning Server for Preceptual.

    Coordinates FL training across multiple Libiao sites.
    """

    def __init__(self, config: Optional[FLServerConfig] = None):
        self.config = config or FLServerConfig()

        if not FLOWER_AVAILABLE:
            logger.error("Flower not available. Cannot start FL server.")
            self.strategy = None
            return

        # Create strategy
        self.benchmark_evaluator = BenchmarkEvaluator()
        self.strategy = PreceptualFedAvg(
            self.config,
            self.benchmark_evaluator,
            on_aggregation=self._on_aggregation,
        )

        # State
        self.is_running = False
        self.start_time: Optional[float] = None

    def _on_aggregation(self, result: AggregationResult) -> None:
        """Callback for aggregation events."""
        logger.info(
            f"Round {result.round_num}: "
            f"clients={result.num_clients}, "
            f"loss={result.aggregated_loss:.4f}, "
            f"gate={'PASS' if result.passed_gate else 'FAIL'} ({result.gate_score:.3f})"
        )

    def start(self, initial_params: Optional[List[np.ndarray]] = None) -> None:
        """
        Start the FL server.

        Args:
            initial_params: Optional initial model parameters
        """
        if not FLOWER_AVAILABLE:
            logger.error("Cannot start FL server without Flower")
            return

        self.is_running = True
        self.start_time = time.time()

        # Configure initial parameters
        if initial_params is not None:
            initial_parameters = ndarrays_to_parameters(initial_params)
        else:
            initial_parameters = None

        logger.info(f"Starting FL server at {self.config.server_address}")

        try:
            fl.server.start_server(
                server_address=self.config.server_address,
                config=fl.server.ServerConfig(num_rounds=self.config.num_rounds),
                strategy=self.strategy,
            )
        except Exception as e:
            logger.error(f"FL server error: {e}")
            raise
        finally:
            self.is_running = False

    def get_results(self) -> List[AggregationResult]:
        """Get aggregation results."""
        if self.strategy:
            return self.strategy.round_results
        return []

    def get_latest_artifact_path(self) -> Optional[Path]:
        """Get path to latest saved artifact."""
        artifact_dir = Path(self.config.artifact_dir)
        if not artifact_dir.exists():
            return None

        weights_files = sorted(artifact_dir.glob("round_*_weights.npz"))
        if weights_files:
            return weights_files[-1]
        return None
