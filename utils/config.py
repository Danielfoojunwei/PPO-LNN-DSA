"""
Configuration management for PPO-LFM training.
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 256
    num_lfm_layers: int = 2
    num_lstm_layers: int = 2
    num_heads: int = 4
    use_moe: bool = False


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    update_interval: int = 2048


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    num_channels: int = 10
    num_devices: int = 20
    seq_length: int = 10
    max_steps: int = 100
    interference_threshold: float = 0.5


@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    num_clients: int = 20
    num_edge_servers: int = 4
    clients_per_edge: int = 5
    num_rounds: int = 100
    episodes_per_round: int = 10
    aggregation_method: str = 'fedavg'


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)

    device: str = 'cpu'
    seed: int = 42
    save_interval: int = 10
    eval_interval: int = 5
    log_interval: int = 1
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    results_dir: str = 'results'


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        TrainingConfig instance
    """
    if config_path is None:
        return TrainingConfig()

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Parse nested configs
    model_config = ModelConfig(**config_dict.get('model', {}))
    ppo_config = PPOConfig(**config_dict.get('ppo', {}))
    env_config = EnvironmentConfig(**config_dict.get('env', {}))
    federated_config = FederatedConfig(**config_dict.get('federated', {}))

    # Create training config
    training_config = TrainingConfig(
        model=model_config,
        ppo=ppo_config,
        env=env_config,
        federated=federated_config
    )

    # Update top-level fields
    for key in ['device', 'seed', 'save_interval', 'eval_interval', 'log_interval', 'checkpoint_dir', 'log_dir', 'results_dir']:
        if key in config_dict:
            setattr(training_config, key, config_dict[key])

    return training_config


def save_config(config: TrainingConfig, path: str):
    """
    Save configuration to YAML file.

    Args:
        config: TrainingConfig instance
        path: Path to save YAML file
    """
    config_dict = {
        'model': {
            'hidden_dim': config.model.hidden_dim,
            'num_lfm_layers': config.model.num_lfm_layers,
            'num_heads': config.model.num_heads,
            'use_moe': config.model.use_moe
        },
        'ppo': {
            'lr': config.ppo.lr,
            'gamma': config.ppo.gamma,
            'gae_lambda': config.ppo.gae_lambda,
            'clip_epsilon': config.ppo.clip_epsilon,
            'value_coef': config.ppo.value_coef,
            'entropy_coef': config.ppo.entropy_coef,
            'max_grad_norm': config.ppo.max_grad_norm,
            'n_epochs': config.ppo.n_epochs,
            'batch_size': config.ppo.batch_size,
            'update_interval': config.ppo.update_interval
        },
        'env': {
            'num_channels': config.env.num_channels,
            'num_devices': config.env.num_devices,
            'seq_length': config.env.seq_length,
            'max_steps': config.env.max_steps,
            'interference_threshold': config.env.interference_threshold
        },
        'federated': {
            'num_clients': config.federated.num_clients,
            'num_edge_servers': config.federated.num_edge_servers,
            'clients_per_edge': config.federated.clients_per_edge,
            'num_rounds': config.federated.num_rounds,
            'episodes_per_round': config.federated.episodes_per_round,
            'aggregation_method': config.federated.aggregation_method
        },
        'device': config.device,
        'seed': config.seed,
        'save_interval': config.save_interval,
        'eval_interval': config.eval_interval,
        'log_interval': config.log_interval,
        'checkpoint_dir': config.checkpoint_dir,
        'log_dir': config.log_dir
    }

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
