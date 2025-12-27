"""
Cloud Server for Hierarchical Federated Learning

Coordinates global model aggregation across edge servers.
"""

import torch
import numpy as np
from collections import defaultdict


class CloudServer:
    """
    Cloud server that coordinates global federated learning.

    In hierarchical federated learning:
    - Edge servers aggregate client updates (regional models)
    - Cloud server aggregates edge server updates (global model)
    - Cloud server distributes global model back to edge servers
    """
    def __init__(self, device='cpu'):
        """
        Args:
            device: Device to run on
        """
        self.device = device
        self.edge_servers = []
        self.global_model = None
        self.training_history = []
        self.round_number = 0

    def register_edge_server(self, edge_server):
        """
        Register an edge server with the cloud.

        Args:
            edge_server: EdgeServer instance
        """
        self.edge_servers.append(edge_server)

    def aggregate_edge_servers(self, edge_updates, aggregation_method='fedavg'):
        """
        Aggregate model updates from edge servers.

        Args:
            edge_updates: List of (edge_id, state_dict, num_clients) tuples
            aggregation_method: Aggregation method

        Returns:
            global_model: Aggregated global model
        """
        if aggregation_method == 'fedavg':
            return self._fedavg(edge_updates)
        elif aggregation_method == 'weighted':
            return self._weighted_average(edge_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    def _fedavg(self, edge_updates):
        """
        FedAvg: Simple average of edge models.

        Args:
            edge_updates: List of (edge_id, state_dict, num_clients)

        Returns:
            averaged_state_dict: Averaged model parameters
        """
        if not edge_updates:
            return self.global_model

        # Initialize aggregated state dict
        aggregated_dict = {}

        # Get first edge's state dict structure
        _, first_state_dict, _ = edge_updates[0]

        # Average each parameter
        for key in first_state_dict.keys():
            # Stack parameters from all edges
            params = [state_dict[key].float() for _, state_dict, _ in edge_updates]
            stacked = torch.stack(params)

            # Compute average
            aggregated_dict[key] = torch.mean(stacked, dim=0)

        self.global_model = aggregated_dict
        return aggregated_dict

    def _weighted_average(self, edge_updates):
        """
        Weighted average based on number of clients.

        Args:
            edge_updates: List of (edge_id, state_dict, num_clients)

        Returns:
            weighted_state_dict: Weighted averaged model parameters
        """
        if not edge_updates:
            return self.global_model

        # Calculate total clients
        total_clients = sum(num_clients for _, _, num_clients in edge_updates)

        # Initialize aggregated state dict
        aggregated_dict = {}

        # Get first edge's state dict structure
        _, first_state_dict, _ = edge_updates[0]

        # Weighted average for each parameter
        for key in first_state_dict.keys():
            weighted_sum = None

            for edge_id, state_dict, num_clients in edge_updates:
                weight = num_clients / total_clients
                weighted_param = state_dict[key].float() * weight

                if weighted_sum is None:
                    weighted_sum = weighted_param
                else:
                    weighted_sum += weighted_param

            aggregated_dict[key] = weighted_sum

        self.global_model = aggregated_dict
        return aggregated_dict

    def get_global_model(self):
        """
        Get the current global model.

        Returns:
            global_model: Current global model
        """
        return self.global_model

    def distribute_to_edges(self, state_dict=None):
        """
        Distribute global model to edge servers.

        Args:
            state_dict: Model to distribute (defaults to global model)
        """
        if state_dict is None:
            state_dict = self.global_model

        if state_dict is None:
            return

        for edge_server in self.edge_servers:
            edge_server.set_global_model(state_dict)
            edge_server.distribute_to_clients(state_dict)

    def training_round(self, num_episodes_per_client=10):
        """
        Execute one global training round.

        Args:
            num_episodes_per_client: Episodes each client trains

        Returns:
            metrics: Global training metrics
        """
        self.round_number += 1

        # Each edge server runs a training round
        edge_metrics_list = []
        edge_updates = []

        for edge_server in self.edge_servers:
            # Edge training round
            edge_metrics, regional_model = edge_server.train_round(
                num_episodes_per_client=num_episodes_per_client
            )
            edge_metrics_list.append(edge_metrics)

            # Collect edge update
            edge_updates.append((
                edge_server.edge_id,
                regional_model,
                edge_metrics['num_clients']
            ))

        # Aggregate edge updates into global model
        global_model = self.aggregate_edge_servers(edge_updates)

        # Distribute global model to edges
        self.distribute_to_edges(global_model)

        # Compute global metrics
        global_metrics = {
            'round': self.round_number,
            'num_edges': len(edge_updates),
            'total_clients': sum(m['num_clients'] for m in edge_metrics_list),
            'mean_edge_reward': np.mean([m['mean_client_reward'] for m in edge_metrics_list]),
            'std_edge_reward': np.std([m['mean_client_reward'] for m in edge_metrics_list]),
            'edge_metrics': edge_metrics_list
        }

        # Store history
        self.training_history.append(global_metrics)

        return global_metrics

    def save_global_model(self, path):
        """
        Save the global model.

        Args:
            path: Path to save the model
        """
        if self.global_model is not None:
            torch.save({
                'global_model': self.global_model,
                'round_number': self.round_number,
                'training_history': self.training_history
            }, path)

    def load_global_model(self, path):
        """
        Load a global model.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.global_model = checkpoint['global_model']
        self.round_number = checkpoint.get('round_number', 0)
        self.training_history = checkpoint.get('training_history', [])

        # Distribute to edges
        self.distribute_to_edges()
