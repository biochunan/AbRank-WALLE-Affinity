from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import seaborn as sns
import torch
import torch.nn as nn
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
        self.training_metric_epoch = []
        self.validation_loss_epoch = []
        self.validation_metric_epoch = []
        self.validation_pred_labels_epoch = []
        self.validation_true_labels_epoch = []
        self.test_pred_labels_epoch = []
        self.test_true_labels_epoch = []
        self.test_loss_epoch = []
        self.test_metric_epoch = []
        self.test_results = {}  # for storing test results

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
        return {
            "MarginRankingLoss": nn.MarginRankingLoss(
                margin=self.config.loss_func.margin_ranking_loss.margin
            )
        }

    def configure_metric_func_dict(self):
        """
        Configure the metric function dictionary
        """
        def _accuracy(pred_rank_label: Tensor, true_rank_label: Tensor) -> Tensor:
            return (pred_rank_label == true_rank_label).float().mean()
        return {
            "accuracy": _accuracy
        }

    # --------------------------------------------------------------------------
    # Custom methods
    # --------------------------------------------------------------------------
    def compute_loss(
        self,
        y_pred1: Tensor,  # e.g. affinity
        y_pred2: Tensor,  # e.g. affinity
        ranking_labels: Tensor,  # e.g. 1 or -1, if y_pred1 > y_pred2, then ranking_labels = 1, otherwise -1
        stage: str,
    ) -> Dict[str, Tensor]:
        if y_pred1.ndim == 2:
            y_pred1 = y_pred1.squeeze()
        if y_pred2.ndim == 2:
            y_pred2 = y_pred2.squeeze()
        if ranking_labels.ndim == 2:
            ranking_labels = ranking_labels.squeeze()

        # compute loss
        loss_dict = {
            f"{stage}/loss/{k}": v(
                y_pred1.float(), y_pred2.float(), ranking_labels.float()
            )
            for k, v in self.loss_func_dict.items()
        }
        return loss_dict

    def get_labels(self, batch: PygBatch) -> Tensor:
        return batch.y

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
        return affinity_pred

    def _one_step(
        self, batch1: PygBatch, batch2: PygBatch, ranking_labels: Tensor, stage: str
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        # forward
        y_pred1: Tensor = self.forward(batch1)  # (B, 1)
        y_pred2: Tensor = self.forward(batch2)  # (B, 1)
        label_pred: Tensor = (y_pred1 > y_pred2).float() * 2 - 1

        # compute loss
        loss_dict = self.compute_loss(
            y_pred1=y_pred1,
            y_pred2=y_pred2,
            ranking_labels=ranking_labels,
            stage=stage
        )

        # NOTE: add key value pair `"loss": loss_value` the loss dict to return after each step
        # NOTE: required by Lightning
        # since we only use a single loss function, we only need to return the total
        loss_values = list(loss_dict.values())
        loss_dict["loss"] = loss_values[0]

        return loss_dict, label_pred

    def training_step(self, batch: Tuple[PygBatch, PygBatch, Tensor], batch_idx: int) -> Tensor:
        # compute loss
        loss_dict, _ = self._one_step(
            batch1=batch[0],
            batch2=batch[1],
            ranking_labels=batch[2],
            stage="train",
        )
        # store loss for each batch in an epoch
        self.training_loss_epoch.append(loss_dict["loss"])
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        self.log_step(loss_dict, sync_dist=is_distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Tuple[PygBatch, PygBatch, Tensor], batch_idx: int) -> Tensor:
        # compute loss
        loss_dict, label_pred = self._one_step(
            batch1=batch[0],
            batch2=batch[1],
            ranking_labels=batch[2],
            stage="val",
        )
        # store loss for each batch in an epoch
        self.validation_loss_epoch.append(loss_dict["loss"])
        # store pred labels for each batch in an epoch
        self.validation_pred_labels_epoch.append(label_pred.squeeze())  # (B,)
        # store true labels for each batch in an epoch
        self.validation_true_labels_epoch.append(batch[2].squeeze())  # (B,)
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        self.log_step(loss_dict, sync_dist=is_distributed)
        return loss_dict["loss"]

    def test_step(self, batch: Tuple[PygBatch, PygBatch, Tensor], batch_idx: int) -> Tensor:
        """
        Test set is packed in a single batch. This function is called only once
        for the entire test set.
        """
        # compute loss
        loss_dict, label_pred = self._one_step(
            batch1=batch[0],
            batch2=batch[1],
            ranking_labels=batch[2],
            stage="test",
        )
        # append loss value to the list
        self.test_loss_epoch.append(loss_dict["loss"])
        # store pred labels for each batch in an epoch
        self.test_pred_labels_epoch.append(label_pred.squeeze())  # (B,)
        # store true labels for each batch in an epoch
        self.test_true_labels_epoch.append(batch[2].squeeze())  # (B,)
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        self.log_step(loss_dict, sync_dist=is_distributed)

        '''
        # TODO: 我们最后是否应该输出一个最终的排序，比如 test set 一共有 289 个 sample
        # 则最终输出这 289 个 sample 的 排序，这有点像 bobble sort
        # e.g. (sample 0, rank 1), (sample 1, rank 2), ..., (sample 288, rank 289)
        # 再计算下面的 correlation 指标
        '''

        # # calculate the correlation between predicted and true affinity
        # y_pred, y_true = y_pred.squeeze().cpu(), batch.y.squeeze().cpu()
        # corr = PearsonCorrCoef()(y_pred, y_true).item()
        # self.log_step(log_dict={"test/corr": corr}, sync_dist=is_distributed)
        # # add results to the test results for later artifact logging
        # self.test_results.update(
        #     {
        #         "loss": loss_dict["loss"],
        #         "y_pred": y_pred,
        #         "y_true": y_true,
        #         "pearson_corr": corr,
        #     }
        # )
        return loss_dict["loss"]

    def predict_step(self, batch: Tuple[PygBatch, PygBatch, Tensor], batch_idx: int) -> Tensor:
        """
        This function is called during inference.
        """
        # forward
        y_pred1 = self.forward(batch[0])  # (B, 1)
        y_pred2 = self.forward(batch[1])  # (B, 1)
        # predicted labels convert to 1 or -1
        label_pred = (y_pred1 > y_pred2).float() * 2 - 1
        return label_pred

    # --------------------------------------------------------------------------
    # Epoch Hooks
    # --------------------------------------------------------------------------
    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack(self.training_loss_epoch).mean()

        # Check if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        # Log the average loss
        self.log_epoch(log_dict={"train/loss/avg": avg_loss}, sync_dist=sync_dist)

        # # Log the average loss to console
        # logger.info(f"Epoch {self.current_epoch} loss: {avg_loss}")

        # Clear lists for the next epoch
        self.training_loss_epoch.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.validation_loss_epoch).mean()

        # Check if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1

        # store pred labels for each batch in an epoch
        pred_labels = torch.cat(self.validation_pred_labels_epoch, dim=0)
        true_labels = torch.cat(self.validation_true_labels_epoch, dim=0)

        # calculate accuracy
        acc = self.metric_func_dict["accuracy"](pred_labels, true_labels)

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        # Log the average loss
        self.log_epoch(log_dict={"val/loss/avg": avg_loss}, sync_dist=sync_dist)
        self.log_epoch(log_dict={"val/accuracy": acc}, sync_dist=sync_dist)

        # Clear lists for the next epoch
        self.validation_loss_epoch.clear()
        self.validation_pred_labels_epoch.clear()
        self.validation_true_labels_epoch.clear()

    def on_test_epoch_end(self) -> None:
        avg_loss = torch.stack(self.test_loss_epoch).mean()

        # Check if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1

        # store pred labels for each batch in an epoch
        pred_labels = torch.cat(self.test_pred_labels_epoch, dim=0)
        true_labels = torch.cat(self.test_true_labels_epoch, dim=0)

        # calculate accuracy
        acc = self.metric_func_dict["accuracy"](pred_labels, true_labels)

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        # Log the average loss
        self.log_epoch(log_dict={"test/loss/avg": avg_loss}, sync_dist=sync_dist)
        self.log_epoch(log_dict={"test/accuracy": acc}, sync_dist=sync_dist)

        # Clear lists for the next epoch
        self.test_loss_epoch.clear()
        self.test_pred_labels_epoch.clear()
        self.test_true_labels_epoch.clear()





