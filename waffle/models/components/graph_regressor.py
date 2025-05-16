import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch_geometric.nn import global_mean_pool


# Regression decoder
class GraphRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = global_mean_pool
        self.regressor = nn.LazyLinear(out_features=1)

    def forward(self, B_z: Tensor, G_z: Tensor, batch: PygBatch) -> Tensor:
        """
        Args:
            B_z: (Tensor) shape (Nb, C)
            G_z: (Tensor) shape (Ng, C)
            batch: (PygBatch) batched data returned by PyG DataLoader
        Returns:
            affinity_pred: (Tensor) shape (B, 1)
        """
        h_b = self.pooling(
            B_z, batch.x_b_batch
        )  # shape (B, C), doesn't change the embedding dimension
        h_g = self.pooling(
            G_z, batch.x_g_batch
        )  # shape (B, C), doesn't change the embedding dimension
        x = torch.cat([h_b, h_g], dim=1)  # shape (B, C*2)
        affinity_pred = self.regressor(x)  # shape (B, 1)
        return affinity_pred
