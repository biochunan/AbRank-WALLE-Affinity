from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as PyGSequential
from torchmetrics import PearsonCorrCoef

from waffle.models.components.graph_regressor import GraphRegressor

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("Set2")


# Regression model: from WALLE classification model with a regression decoder
class RegressionGCNAbAgIntLM(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: (DictConfig) configuration
        The configuration should contain the following keys:
        - encoder: (DictConfig) encoder configuration
        - regressor: (DictConfig) regressor configuration
        - loss_func: (DictConfig) loss function configuration
        """
        super().__init__()
        self.config = cfg

        # store the loss for each batch in an epoch
        self.training_loss_epoch = []
        self.training_pred_y_epoch = []
        self.training_true_y_epoch = []
        self.validation_loss_epoch = []
        self.validation_pred_y_epoch = []
        self.validation_true_y_epoch = []
        self.test_pred_y_epoch = {}  # e.g. {"generalization": [], "perturbation": []} for multiple test sets
        self.test_true_y_epoch = {}  # e.g. {"generalization": [], "perturbation": []} for multiple test sets
        self.test_loss_epoch = {}  # e.g. {"generalization": [], "perturbation": []} for multiple test sets

        logger.info("Instantiating encoder blocks ...")
        self.B_encoder_block = self.create_encoder_block(**cfg.encoder.ab)
        self.G_encoder_block = self.create_encoder_block(**cfg.encoder.ag)
        logger.info(self.B_encoder_block)
        logger.info(self.G_encoder_block)

        logger.info("Instantiating regression component ...")
        self.regressor = GraphRegressor()
        logger.info(self.regressor)

        logger.info("Instantiating losses...")
        self.loss_func_dict = self.configure_loss_func_dict()
        logger.info(f"Using losses: {self.loss_func_dict}")

        logger.info("Instantiating metrics...")
        self.metric_func_dict = self.configure_metric_func_dict()
        logger.info(f"Using metrics: {self.metric_func_dict}")

        self.save_hyperparameters()  # add config to self.hparams

    def create_encoder_block(
        self,
        node_feat_name: str,
        edge_index_name: str,
        input_dim: int,
        input_act: str,
        dim_list: List[int],
        act_list: List[str],
        gcn_kwargs: Dict[str, Any],
    ):
        def _create_gcn_layer(
            i: int, j: int, in_channels: int, out_channels: int
        ) -> Tuple[GCNConv, str]:
            """
            Generate a GCN layer

            Args:
                i (int): input layer index
                j (int): output layer index
                in_channels (int): input channels
                out_channels (int): output channels

            Returns:
                GCNConv: GCN layer
                str: a string to map the input args to the output, e.g. "x_b,
                edge_index_b -> x_b_1" means use node features `x_b` and edge_index
                `edge_index_b` to derive updated node features named `x_b_1`. This
                is used in Sequential module to map the input args to the output of
                the layer. See the reference for more details.
                Ref: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=sequential#torch_geometric.nn.sequential.Sequential
            """
            if i == 0:
                mapping = f"{node_feat_name}, {edge_index_name} -> {node_feat_name}_{j}"
            else:
                mapping = (
                    f"{node_feat_name}_{i}, {edge_index_name} -> {node_feat_name}_{j}"
                )
            # print(mapping)

            return (
                GCNConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **gcn_kwargs,
                ),
                mapping,
            )

        def _create_act_layer(act_name: Optional[str]) -> nn.Module:
            # assert act_name is either None or str
            assert act_name is None or isinstance(
                act_name, str
            ), f"act_name must be None or str, got {act_name}"

            if act_name is None:
                # return identity
                return (nn.Identity(),)
            elif act_name.lower() == "relu":
                return (nn.ReLU(inplace=True),)
            elif act_name.lower() == "leakyrelu":
                return (nn.LeakyReLU(inplace=True),)
            else:
                raise ValueError(
                    f"activation {act_name} not supported, please choose from ['relu', 'leakyrelu', None]"
                )

        modules = [
            _create_gcn_layer(0, 1, input_dim, dim_list[0]),
            _create_act_layer(input_act),
        ]

        for i in range(len(dim_list) - 1):
            modules.extend(
                [
                    _create_gcn_layer(
                        i + 1, i + 2, dim_list[i], dim_list[i + 1]
                    ),  # i+1 increment due to the input layer
                    _create_act_layer(act_list[i]),
                ]
            )

        return PyGSequential(
            input_args=f"{node_feat_name}, {edge_index_name}", modules=modules
        )

    def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch: (PygBatch) batched data returned by PyG DataLoader
        Returns:
            B_z: (Tensor) shape (Nb, C)
            G_z: (Tensor) shape (Ng, C)
        """
        B_z = self.B_encoder_block(
            batch.x_b, batch.edge_index_b
        )  # (batch.x_b.shape[0], C), e.g.  C=64 depends on the config
        G_z = self.G_encoder_block(
            batch.x_g, batch.edge_index_g
        )  # (batch.x_g.shape[0], C), e.g.  C=64 depends on the config

        return B_z, G_z

    def forward(self, batch: PygBatch) -> Tensor:
        # encode
        z_ab, z_ag = self.encode(batch)
        # regression
        affinity_pred = self.regressor(z_ab, z_ag, batch)
        return affinity_pred

    # --------------------------------------------------------------------------
    # Configure
    # --------------------------------------------------------------------------
    def configure_optimizers(self):
        logger.info(f"Optimizer config: {self.config.optimizer}")
        optimizer = hydra.utils.instantiate(self.config.optimizer)["optimizer"]
        logger.info(f"Optimizer: {optimizer}")
        optimizer = optimizer(self.parameters())

        if self.config.get("scheduler"):
            logger.info("Instantiating scheduler...")
            scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer)
            scheduler = OmegaConf.to_container(scheduler)
            scheduler["scheduler"] = scheduler["scheduler"](optimizer=optimizer)
            optimizer_config = {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
            logger.info(f"optimizer configuration: {optimizer_config}")
            return optimizer_config
        return optimizer

    def configure_loss_func_dict(self):
        """
        Configure the loss function dictionary

        rank loss nn.MarginRankingLoss(margin=0.5)
        Args:
            input1: (Tensor) predicted affinity
            input2: (Tensor) predicted affinity
            target: (Tensor) ranking labels

        """
        # rank loss nn.MarginRankingLoss(margin=0.5)
        return {"mse": nn.MSELoss()}

    def configure_metric_func_dict(self):
        """
        Configure the metric function dictionary
        """

        def _accuracy(pred_rank_label: Tensor, true_rank_label: Tensor) -> Tensor:
            return (pred_rank_label == true_rank_label).float().mean()

        return {"accuracy": _accuracy}

    # --------------------------------------------------------------------------
    # Custom methods
    # --------------------------------------------------------------------------
    def compute_loss(
        self,
        pred_y: Tensor,
        true_y: Tensor,
        stage: str,
    ) -> Dict[str, Tensor]:
        if pred_y.ndim == 2:
            pred_y = pred_y.squeeze()
        if true_y.ndim == 2:
            true_y = true_y.squeeze()

        # compute loss
        loss_dict = {
            f"{stage}/loss/{k}": v(
                pred_y.float(), true_y.float()
            )
            for k, v in self.loss_func_dict.items()
        }
        return loss_dict

    # --------------------------------------------------------------------------
    # Log
    # --------------------------------------------------------------------------
    def log_step(self, log_dict: Dict[str, float], sync_dist: bool = False):
        # log step
        for k, v in log_dict.items():
            self.log(
                name=k,
                value=v,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=sync_dist,
            )

    def log_epoch(self, log_dict: Dict[str, float], sync_dist: bool = True) -> None:
        # log epoch
        for k, v in log_dict.items():
            self.log(
                name=k,
                value=v,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=sync_dist,
            )

    # --------------------------------------------------------------------------
    # Step Hooks
    # --------------------------------------------------------------------------
    def forward(self, batch: PygBatch) -> Tensor:
        # encode
        z_ab, z_ag = self.encode(batch)
        # regression
        affinity_pred = self.regressor(z_ab, z_ag, batch)
        # clip to (-6, 6)
        affinity_pred = torch.clamp(affinity_pred, -12, 12)
        return affinity_pred

    def _one_step(
        self, batch: PygBatch, stage: str
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        # forward
        pred_y: Tensor = self.forward(batch)  # (B, 1)

        # compute loss
        loss_dict = self.compute_loss(
            pred_y=pred_y, true_y=batch.y, stage=stage
        )

        # NOTE: add key value pair `"loss": loss_value` the loss dict to return after each step
        # NOTE: required by Lightning
        # since we only use a single loss function, we only need to return the total
        loss_values = list(loss_dict.values())
        loss_dict["loss"] = loss_values[0]

        return loss_dict, pred_y

    def training_step(
        self, batch: PygBatch, batch_idx: int
    ) -> Tensor:
        # compute loss
        loss_dict, pred_y = self._one_step(
            batch=batch,
            stage="train",
        )
        # store loss for each batch in an epoch
        self.training_loss_epoch.append(loss_dict["loss"])
        # store pred labels for each batch in an epoch
        self.training_pred_y_epoch.append(pred_y.squeeze())  # (B,)
        # store true labels for each batch in an epoch
        self.training_true_y_epoch.append(batch.y.squeeze())  # (B,)
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        self.log_step(loss_dict, sync_dist=is_distributed)
        return loss_dict["loss"]

    def validation_step(
        self, batch: PygBatch, batch_idx: int
    ) -> Tensor:
        # compute loss
        loss_dict, pred_y = self._one_step(
            batch=batch,
            stage="val",
        )
        # store loss for each batch in an epoch
        self.validation_loss_epoch.append(loss_dict["loss"])
        # store pred labels for each batch in an epoch
        self.validation_pred_y_epoch.append(pred_y.squeeze())  # (B,)
        # store true labels for each batch in an epoch
        self.validation_true_y_epoch.append(batch.y.squeeze())  # (B,)
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        self.log_step(loss_dict, sync_dist=is_distributed)
        return loss_dict["loss"]

    def test_step(
        self,
        batch: PygBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """
        Test set is packed in a single batch. This function is called only once
        for the entire test set.
        NOTE: the test set doesn't have true y, we only run a forward pass
        """
        # NOTE: input test dataloader is a dict of two loaders
        # e.g. test_dataloader = {"generalization": dl1, "perturbation": dl2}
        test_name = list(self.test_dataloader.keys())[dataloader_idx]
        names = batch.name
        pred_y: Tensor = self.forward(batch)  # (B, 1)
        d = [(n, p) for n, p in zip(names, pred_y.squeeze())]
        if test_name not in self.test_pred_y_epoch:
            self.test_pred_y_epoch[test_name] = []
        # store pred y for each batch in an epoch
        self.test_pred_y_epoch[test_name].extend(d)

    def predict_step(
        self, batch: PygBatch, batch_idx: int
    ) -> Tensor:
        """
        This function is called during inference.
        """
        # forward
        y_pred = self.forward(batch)  # (B, 1)
        return y_pred

    # --------------------------------------------------------------------------
    # Epoch Hooks
    # --------------------------------------------------------------------------
    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack(self.training_loss_epoch).mean()

        # Check if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        pred_y = torch.cat(self.training_pred_y_epoch, dim=0)
        true_y = torch.cat(self.training_true_y_epoch, dim=0)

        # calculate accuracy
        mse = self.loss_func_dict["mse"](pred_y, true_y)

        # Log the average loss
        self.log_epoch(log_dict={"train/loss/avg": avg_loss}, sync_dist=sync_dist)
        self.log_epoch(log_dict={"train/mse": mse}, sync_dist=sync_dist)

        # Clear lists for the next epoch
        self.training_loss_epoch.clear()
        self.training_pred_y_epoch.clear()
        self.training_true_y_epoch.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.validation_loss_epoch).mean()

        # Check if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1

        # store pred y for each batch in an epoch
        pred_y = torch.cat(self.validation_pred_y_epoch, dim=0)
        true_y = torch.cat(self.validation_true_y_epoch, dim=0)

        # calculate accuracy
        mse = self.loss_func_dict["mse"](pred_y, true_y)

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        # Log the average loss
        self.log_epoch(log_dict={"val/loss/avg": avg_loss}, sync_dist=sync_dist)
        self.log_epoch(log_dict={"val/mse": mse}, sync_dist=sync_dist)

        # Clear lists for the next epoch
        self.validation_loss_epoch.clear()
        self.validation_pred_y_epoch.clear()
        self.validation_true_y_epoch.clear()

    def on_test_epoch_end(self) -> None:
        # NOTE: test set doesn't have true y, we only upload a csv file artifact
        # storing graph pair name and pred y for later usage
        for test_name in self.test_pred_y_epoch:
            # Check if we are in a distributed setting
            is_distributed = self.trainer.world_size > 1

            # store pred y for each batch in an epoch
            names, pred_y = zip(*self.test_pred_y_epoch[test_name])
            names = list(names)
            # move each item in pred_y to cpu and to float
            pred_y = [p.item() for p in pred_y]
            df = pd.DataFrame({"name": names, "pred_y": pred_y})

            # Save DataFrame to a temporary CSV file
            import os
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                df.to_csv(tmp.name, index=False)
                # create an artifact
                artifact = wandb.Artifact(
                    name=f"test_{test_name}",
                    type="test",
                    description=f"Test set for {test_name}",
                )
                artifact.add_file(tmp.name, name=f"{test_name}.csv")
                self.logger.experiment.log_artifact(artifact)
                # Clean up the temporary file
                os.unlink(tmp.name)

            # Clear lists for the next epoch
            self.test_pred_y_epoch[test_name].clear()

