"""
Edge Server for Hierarchical Federated Learning

Implements regional aggregation in the client-edge-cloud architecture.
Each edge server aggregates updates from a subset of clients.
"""

import torch
import numpy as np
from collections import defaultdict


class EdgeServer:
    """
    Edge server that aggregates updates from local clients.

    In hierarchical federated learning:
    - Clients train locally and send updates to edge server
    - Edge server aggregates client updates (regional model)
    - Edge server sends aggregated update to cloud server
    """
    def __init__(self, edge_id, device='cpu'):
        """
        Args:
            edge_id: Unique identifier for this edge server
            device: Device to run on
        """
        self.edge_id = edge_id
        self.device = device
        self.clients = []
        self.regional_model = None
        self.aggregation_history = []

    def register_client(self, client):
        """
        Register a client with this edge server.

        Args:
            client: FederatedClient instance
        """
        self.clients.append(client)

    def aggregate_clients(self, client_updates, aggregation_method='fedavg'):
        """
        Aggregate model updates from clients.

        Args:
            client_updates: List of (client_id, state_dict, num_samples) tuples
            aggregation_method: Aggregation method ('fedavg', 'fedprox', etc.)

        Returns:
            aggregated_model: Aggregated model state dictionary
        """
        if aggregation_method == 'fedavg':
            return self._fedavg(client_updates)
        elif aggregation_method == 'weighted':
            return self._weighted_average(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    def _fedavg(self, client_updates):
        """
        FedAvg: Simple average of client models.

        Args:
            client_updates: List of (client_id, state_dict, num_samples)

        Returns:
            averaged_state_dict: Averaged model parameters
        """
        if not client_updates:
            return self.regional_model

        # Initialize aggregated state dict
        aggregated_dict = {}

        # Get first client's state dict structure
        _, first_state_dict, _ = client_updates[0]

        # Average each parameter
        for key in first_state_dict.keys():
            # Stack parameters from all clients
            params = [state_dict[key].float() for _, state_dict, _ in client_updates]
            stacked = torch.stack(params)

            # Compute average
            aggregated_dict[key] = torch.mean(stacked, dim=0)

        self.regional_model = aggregated_dict
        return aggregated_dict

    def _weighted_average(self, client_updates):
        """
        Weighted average based on number of samples.

        Args:
            client_updates: List of (client_id, state_dict, num_samples)

        Returns:
            weighted_state_dict: Weighted averaged model parameters
        """
        if not client_updates:
            return self.regional_model

        # Calculate total samples
        total_samples = sum(num_samples for _, _, num_samples in client_updates)

        # Initialize aggregated state dict
        aggregated_dict = {}

        # Get first client's state dict structure
        _, first_state_dict, _ = client_updates[0]

        # Weighted average for each parameter
        for key in first_state_dict.keys():
            weighted_sum = None

            for client_id, state_dict, num_samples in client_updates:
                weight = num_samples / total_samples
                weighted_param = state_dict[key].float() * weight

                if weighted_sum is None:
                    weighted_sum = weighted_param
                else:
                    weighted_sum += weighted_param

            aggregated_dict[key] = weighted_sum

        self.regional_model = aggregated_dict
        return aggregated_dict

    def get_regional_model(self):
        """
        Get the current regional model.

        Returns:
            regional_model: Current aggregated model
        """
        return self.regional_model

    def set_global_model(self, global_state_dict):
        """
        Update regional model with global model from cloud.

        Args:
            global_state_dict: Global model parameters from cloud server
        """
        self.regional_model = global_state_dict

    def distribute_to_clients(self, state_dict=None):
        """
        Distribute model to registered clients.

        Args:
            state_dict: Model to distribute (defaults to regional model)
        """
        if state_dict is None:
            state_dict = self.regional_model

        if state_dict is None:
            return

        for client in self.clients:
            client.set_model_parameters(state_dict)

    def train_round(self, num_episodes_per_client=10):
        """
        Execute one round of federated training.

        Args:
            num_episodes_per_client: Episodes each client trains

        Returns:
            metrics: Training metrics from all clients
        """
        # Train all clients locally
        client_metrics = []
        client_updates = []

        for client in self.clients:
            # Local training
            metrics = client.local_train(num_episodes=num_episodes_per_client)
            client_metrics.append(metrics)

            # Collect model update
            state_dict = client.get_model_update()
            client_updates.append((
                client.client_id,
                state_dict,
                metrics['total_steps']
            ))

        # Aggregate updates
        aggregated_model = self.aggregate_clients(client_updates)

        # Distribute aggregated model back to clients
        self.distribute_to_clients(aggregated_model)

        # Store aggregation history
        self.aggregation_history.append({
            'num_clients': len(client_updates),
            'client_metrics': client_metrics
        })

        # Compute aggregate metrics
        aggregate_metrics = {
            'edge_id': self.edge_id,
            'num_clients': len(client_updates),
            'mean_client_reward': np.mean([m['mean_reward'] for m in client_metrics]),
            'std_client_reward': np.std([m['mean_reward'] for m in client_metrics])
        }

        return aggregate_metrics, aggregated_model
