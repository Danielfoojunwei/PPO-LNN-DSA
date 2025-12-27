"""
Main training script for hierarchical federated learning with PPO-LFM.

This implements the complete client-edge-cloud architecture for
dynamic spectrum access in IoT networks.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path

from federated.client import FederatedClient
from federated.edge_server import EdgeServer
from federated.cloud_server import CloudServer
from utils.config import load_config, save_config
from utils.metrics import MetricsTracker, Logger, save_metrics_to_json


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_federated_system(config):
    """
    Create the hierarchical federated learning system.

    Returns:
        cloud_server: CloudServer instance with edge servers and clients
    """
    # Determine device
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # State and action dimensions from environment
    state_dim = config.env.num_channels * 3  # quality + interference + history
    action_dim = config.env.num_channels

    # Create cloud server
    cloud_server = CloudServer(device=device)

    # Create edge servers
    edge_servers = []
    for edge_id in range(config.federated.num_edge_servers):
        edge_server = EdgeServer(edge_id=edge_id, device=device)
        edge_servers.append(edge_server)
        cloud_server.register_edge_server(edge_server)

    # Create clients and assign to edge servers
    clients_per_edge = config.federated.clients_per_edge
    for client_id in range(config.federated.num_clients):
        # Determine which edge server this client belongs to
        edge_idx = client_id // clients_per_edge

        # Create client
        client = FederatedClient(
            client_id=client_id,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.model.hidden_dim,
            num_lfm_layers=config.model.num_lfm_layers,
            num_channels=config.env.num_channels,
            num_devices=config.env.num_devices,
            device=device
        )

        # Register with edge server
        edge_servers[edge_idx].register_client(client)

    print(f"Created federated system:")
    print(f"  - 1 cloud server")
    print(f"  - {config.federated.num_edge_servers} edge servers")
    print(f"  - {config.federated.num_clients} clients")
    print(f"  - ~{clients_per_edge} clients per edge")

    return cloud_server


def train_federated(config, cloud_server):
    """
    Train the hierarchical federated learning system.

    Args:
        config: Training configuration
        cloud_server: CloudServer instance
    """
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize logger and metrics tracker
    log_file = os.path.join(config.log_dir, 'training.log')
    logger = Logger(log_file=log_file, console=True)
    metrics_tracker = MetricsTracker(window_size=10)

    logger.log("=" * 80)
    logger.log("PPO-LFM Hierarchical Federated Learning for Dynamic Spectrum Access")
    logger.log("=" * 80)
    logger.log(f"Configuration:")
    logger.log(f"  Model: hidden_dim={config.model.hidden_dim}, layers={config.model.num_lfm_layers}")
    logger.log(f"  Environment: channels={config.env.num_channels}, devices={config.env.num_devices}")
    logger.log(f"  Federated: {config.federated.num_clients} clients, {config.federated.num_edge_servers} edges")
    logger.log(f"  Training: {config.federated.num_rounds} rounds")
    logger.log("=" * 80)

    # Training loop
    all_metrics = []

    for round_num in range(1, config.federated.num_rounds + 1):
        logger.log(f"\n{'=' * 80}")
        logger.log(f"Round {round_num}/{config.federated.num_rounds}")
        logger.log(f"{'=' * 80}")

        # Execute federated training round
        round_metrics = cloud_server.training_round(
            num_episodes_per_client=config.federated.episodes_per_round
        )

        # Track metrics
        metrics_tracker.add('mean_reward', round_metrics['mean_edge_reward'])
        metrics_tracker.add('std_reward', round_metrics['std_edge_reward'])

        # Log metrics
        if round_num % config.log_interval == 0:
            logger.log_metrics(round_num, {
                'mean_reward': round_metrics['mean_edge_reward'],
                'std_reward': round_metrics['std_edge_reward'],
                'num_edges': round_metrics['num_edges'],
                'total_clients': round_metrics['total_clients']
            })

        # Save checkpoint
        if round_num % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'global_model_round_{round_num}.pt'
            )
            cloud_server.save_global_model(checkpoint_path)
            logger.log(f"Saved checkpoint: {checkpoint_path}")

        # Evaluation
        if round_num % config.eval_interval == 0:
            logger.log(f"\nEvaluating at round {round_num}...")
            eval_metrics = evaluate_federated_system(cloud_server, config)
            logger.log_metrics(round_num, eval_metrics)

            # Track evaluation metrics
            metrics_tracker.add('eval_mean_reward', eval_metrics['mean_reward'])
            metrics_tracker.add('eval_success_rate', eval_metrics['mean_success_rate'])
            metrics_tracker.add('eval_collision_rate', eval_metrics['mean_collision_rate'])

        # Store round metrics
        all_metrics.append(round_metrics)

    # Final checkpoint
    final_checkpoint = os.path.join(config.checkpoint_dir, 'global_model_final.pt')
    cloud_server.save_global_model(final_checkpoint)
    logger.log(f"\nSaved final model: {final_checkpoint}")

    # Save all metrics
    metrics_path = os.path.join(config.log_dir, 'metrics.json')
    save_metrics_to_json({'rounds': all_metrics}, metrics_path)
    logger.log(f"Saved metrics: {metrics_path}")

    # Final summary
    logger.log("\n" + "=" * 80)
    logger.log("Training completed!")
    logger.log("=" * 80)
    summary = metrics_tracker.get_summary(window=False)
    logger.log_dict(summary, "Final metrics:")


def evaluate_federated_system(cloud_server, config, num_eval_episodes=10):
    """
    Evaluate the federated learning system.

    Args:
        cloud_server: CloudServer instance
        config: Training configuration
        num_eval_episodes: Number of episodes for evaluation

    Returns:
        eval_metrics: Evaluation metrics
    """
    all_client_metrics = []

    # Evaluate all clients
    for edge_server in cloud_server.edge_servers:
        for client in edge_server.clients:
            client_metrics = client.evaluate(num_episodes=num_eval_episodes)
            all_client_metrics.append(client_metrics)

    # Aggregate metrics
    eval_metrics = {
        'mean_reward': np.mean([m['mean_reward'] for m in all_client_metrics]),
        'std_reward': np.std([m['mean_reward'] for m in all_client_metrics]),
        'mean_success_rate': np.mean([m['mean_success_rate'] for m in all_client_metrics]),
        'mean_collision_rate': np.mean([m['mean_collision_rate'] for m in all_client_metrics])
    }

    return eval_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train PPO-LFM with hierarchical federated learning'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda/auto)')
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = load_config()
        print("Using default config")

    # Override with command-line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device

    # Set random seed
    set_seed(config.seed)

    # Create federated system
    cloud_server = create_federated_system(config)

    # Train
    train_federated(config, cloud_server)


if __name__ == '__main__':
    main()
