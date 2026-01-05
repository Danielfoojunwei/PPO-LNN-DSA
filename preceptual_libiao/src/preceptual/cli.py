"""
Preceptual Libiao CLI

Command-line interface for Airtime OS + Handover Orchestrator.
"""

import argparse
import asyncio
import logging
import sys
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_train(args):
    """Train PPO-LNN policy."""
    from .sim.fleet_world import FleetWorld, FleetWorldConfig
    from .rl.ppo_lnn import PPOLNNPolicy, PPOTrainer
    import torch
    import numpy as np

    logger.info("Starting PPO-LNN training...")

    # Create environment
    config = FleetWorldConfig(
        num_robots=args.num_robots,
        num_aps=args.num_aps,
        max_episode_seconds=args.episode_length,
    )
    env = FleetWorld(config)

    # Create policy
    policy = PPOLNNPolicy(
        global_obs_size=42,
        robot_obs_size=14,
        hidden_size=args.hidden_size,
        num_lnn_layers=args.num_layers,
    )

    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        lr=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
    )

    # Training loop
    total_steps = 0
    for episode in range(args.num_episodes):
        obs, info = env.reset()

        episode_reward = 0
        rollout = {
            "global_obs": [],
            "robot_obs": [],
            "actions": {"slice_budgets": [], "congestion_mode": [], "scan_quota": [], "robot_actions": []},
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }

        hidden = None

        for step in range(int(args.episode_length / 0.5)):
            # Parse observation
            global_obs = torch.tensor(obs[:42], dtype=torch.float32).unsqueeze(0)
            robot_obs = torch.tensor(obs[72:72+700].reshape(50, 14), dtype=torch.float32).unsqueeze(0)

            # Get action
            with torch.no_grad():
                actions, log_prob, value, hidden = policy.get_action(
                    global_obs, robot_obs, hidden=hidden
                )

            # Convert to environment action format
            env_action = {
                "slice_budgets": actions["slice_budgets"][0].numpy().tolist(),
                "congestion_mode": actions["congestion_mode"][0].item(),
                "switch_actions": [],
                "scan_robots": [],
            }

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(env_action)

            # Store transition
            rollout["global_obs"].append(global_obs.squeeze(0))
            rollout["robot_obs"].append(robot_obs.squeeze(0))
            rollout["actions"]["slice_budgets"].append(actions["slice_budgets"].squeeze(0))
            rollout["actions"]["congestion_mode"].append(actions["congestion_mode"].squeeze(0))
            rollout["actions"]["scan_quota"].append(actions["scan_quota"].squeeze(0))
            rollout["log_probs"].append(log_prob.squeeze(0))
            rollout["values"].append(value.squeeze(0))
            rollout["rewards"].append(torch.tensor(reward))
            rollout["dones"].append(torch.tensor(float(terminated or truncated)))

            episode_reward += reward
            total_steps += 1
            obs = next_obs

            if terminated or truncated:
                break

        # Convert rollout to tensors
        for k in ["global_obs", "robot_obs", "log_probs", "values", "rewards", "dones"]:
            rollout[k] = torch.stack(rollout[k])
        for k in ["slice_budgets", "congestion_mode", "scan_quota"]:
            rollout["actions"][k] = torch.stack(rollout["actions"][k])

        # PPO update
        metrics = trainer.update(rollout)

        logger.info(
            f"Episode {episode + 1}/{args.num_episodes}: "
            f"reward={episode_reward:.2f}, "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}"
        )

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_{episode + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "trainer_state_dict": trainer.optimizer.state_dict(),
                "episode": episode + 1,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training complete!")


def cmd_eval(args):
    """Evaluate policy against heuristics."""
    from .sim.fleet_world import FleetWorld, FleetWorldConfig

    logger.info("Running evaluation...")

    config = FleetWorldConfig(
        num_robots=args.num_robots,
        num_aps=args.num_aps,
    )
    env = FleetWorld(config)

    # Run episodes
    total_reward = 0
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        episode_reward = 0

        for step in range(600):  # 5 minute episode
            # Heuristic policy
            action = {
                "slice_budgets": [0.15, 0.25, 0.20, 0.15, 0.10, 0.15],
                "congestion_mode": 0,
                "switch_actions": [],
                "scan_robots": [],
            }

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_reward += episode_reward
        logger.info(f"Episode {episode + 1}: reward={episode_reward:.2f}")

    logger.info(f"Average reward: {total_reward / args.num_episodes:.2f}")


def cmd_simulate(args):
    """Run simulation with RCS mock."""
    from .sim.rcs_mock import RCSMock, RCSMockConfig, RCSMockServer

    logger.info(f"Starting RCS mock server on port {args.port}...")

    config = RCSMockConfig(
        num_robots=args.num_robots,
        num_aps=args.num_aps,
    )
    mock = RCSMock(config)
    server = RCSMockServer(mock, port=args.port)

    async def run():
        await server.start()
        logger.info("Server running. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
                mock.tick(1.0)
        except KeyboardInterrupt:
            await server.stop()

    asyncio.run(run())


def cmd_fl_server(args):
    """Start FL server."""
    from .fl.server import FLServer, FLServerConfig

    logger.info(f"Starting FL server on {args.address}...")

    config = FLServerConfig(
        server_address=args.address,
        num_rounds=args.num_rounds,
        min_fit_clients=args.min_clients,
        artifact_dir=args.artifact_dir,
    )
    server = FLServer(config)
    server.start()


def cmd_fl_client(args):
    """Start FL client."""
    from .fl.client import FLClient, FLClientConfig
    from .rl.ppo_lnn import PPOLNNPolicy
    import torch

    logger.info(f"Starting FL client {args.client_id}...")

    # Create model
    policy = PPOLNNPolicy()

    # Load weights if provided
    if args.weights:
        checkpoint = torch.load(args.weights)
        policy.load_state_dict(checkpoint["policy_state_dict"])

    config = FLClientConfig(
        server_address=args.server,
        client_id=args.client_id,
    )
    client = FLClient(policy, config)
    client.start()


def main():
    parser = argparse.ArgumentParser(
        description="Preceptual Libiao - Airtime OS + Handover Orchestrator"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train PPO-LNN policy")
    train_parser.add_argument("--num-robots", type=int, default=50)
    train_parser.add_argument("--num-aps", type=int, default=5)
    train_parser.add_argument("--num-episodes", type=int, default=100)
    train_parser.add_argument("--episode-length", type=float, default=300.0)
    train_parser.add_argument("--hidden-size", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--ppo-epochs", type=int, default=4)
    train_parser.add_argument("--save-interval", type=int, default=10)
    train_parser.add_argument("--output-dir", type=str, default="./checkpoints")
    train_parser.set_defaults(func=cmd_train)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate policy")
    eval_parser.add_argument("--num-robots", type=int, default=50)
    eval_parser.add_argument("--num-aps", type=int, default=5)
    eval_parser.add_argument("--num-episodes", type=int, default=10)
    eval_parser.add_argument("--weights", type=str, help="Path to model weights")
    eval_parser.set_defaults(func=cmd_eval)

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run RCS mock server")
    sim_parser.add_argument("--port", type=int, default=8080)
    sim_parser.add_argument("--num-robots", type=int, default=50)
    sim_parser.add_argument("--num-aps", type=int, default=5)
    sim_parser.set_defaults(func=cmd_simulate)

    # FL server command
    fl_server_parser = subparsers.add_parser("fl-server", help="Start FL server")
    fl_server_parser.add_argument("--address", type=str, default="0.0.0.0:8090")
    fl_server_parser.add_argument("--num-rounds", type=int, default=100)
    fl_server_parser.add_argument("--min-clients", type=int, default=2)
    fl_server_parser.add_argument("--artifact-dir", type=str, default="./fl_artifacts")
    fl_server_parser.set_defaults(func=cmd_fl_server)

    # FL client command
    fl_client_parser = subparsers.add_parser("fl-client", help="Start FL client")
    fl_client_parser.add_argument("--server", type=str, default="localhost:8090")
    fl_client_parser.add_argument("--client-id", type=str, default="site_001")
    fl_client_parser.add_argument("--weights", type=str, help="Initial weights path")
    fl_client_parser.set_defaults(func=cmd_fl_client)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
