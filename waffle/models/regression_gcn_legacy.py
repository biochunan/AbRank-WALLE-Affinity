from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hydra
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as PyGSequential
from torch_geometric.nn import global_mean_pool
from torchmetrics import PearsonCorrCoef

from waffle.models.components.graph_regressor import GraphRegressor


# Base class
class BaseAbAgInt(nn.Module):
    def __init__(
        self,
        input_ab_dim: int,
        input_ag_dim: int,
        dim_list: List[int],
        act_list: List[str],
        decoder: Optional[Dict] = None,
        try_gpu: bool = True,
        input_ab_act: str = "relu",
        input_ag_act: str = "relu",
    ):
        """
        Base class for AbAgInt model

        NOTE: length of dim_list must equal to length of act_list + 1
        Because we don't apply an activate function to the last layer (i.e. the output layer)

        Args:
            input_ab_dim (int): input antibody graph node feature dimension
            input_ag_dim (int): input antigen graph node feature dimension
            dim_list (List[int]): a list of dimensions for the encoder layers, length is equal to `len(act_list) + 1`
            act_list (List[str]): a list of activation functions for the encoder layers
            decoder (Optional[Dict], optional): decoder layer type. Defaults to None.
                Choices: ['inner_prod', 'fc', 'bilinear']
            try_gpu (bool, optional): try to use GPU. Defaults to True.
                if True and GPU is available, use GPU
            input_ab_act (str, optional): input layer activation function for antibody graph. Defaults to "relu".
            input_ag_act (str, optional): input layer activation function for antigen graph. Defaults to "relu".
        """
        super().__init__()
        self.device = torch.device(
            "cuda" if try_gpu and torch.cuda.is_available() else "cpu"
        )

        # add to hparams
        self.hparams = {
            "input_ab_dim": input_ab_dim,
            "input_ag_dim": input_ag_dim,
            "dim_list": dim_list,
            "act_list": act_list,
            "decoder": decoder,
        }
        self._args_sanity_check()

    def _args_sanity_check(self):
        if self.hparams["dim_list"] is not None or self.hparams["act_list"] is not None:
            try:
                assert (
                    len(self.hparams["dim_list"]) == len(self.hparams["act_list"]) + 1
                ), (
                    f"dim_list length must be equal to act_list length + 1, "
                    f"got dim_list {self.hparams['dim_list']} and act_list {self.hparams['act_list']}"
                )
            except AssertionError as e:
                raise ValueError(
                    "dim_list length must be equal to act_list length + 1, "
                ) from e

        if self.hparams["decoder"] is not None:
            try:
                assert isinstance(self.hparams["decoder"], (dict, DictConfig))
            except AssertionError as e:
                raise TypeError(
                    f"decoder must be a dict, got {self.hparams['decoder']}"
                ) from e
            try:
                assert self.hparams["decoder"]["name"] in (
                    "inner_prod",
                    "fc",
                    "bilinear",
                )
            except AssertionError as e:
                raise ValueError(
                    f"decoder {self.hparams['decoder']['name']} not supported, "
                    "please choose from ['inner_prod', 'fc', 'bilinear']"
                ) from e

    def decoder_factory(
        self, decoder_dict: Dict[str, str]
    ) -> Union[nn.Module, nn.Parameter, None]:
        name = decoder_dict["name"]

        if name == "bilinear":
            init_method = decoder_dict.get("init_method", "kaiming_normal_")
            decoder = nn.Parameter(
                data=torch.empty(
                    self.hparams["dim_list"][-1], self.hparams["dim_list"][-1]
                ),
                requires_grad=True,
            )
            torch.nn.init.__dict__[init_method](decoder)
            return decoder

        elif name == "fc":
            return self.init_fc_decoder(decoder_dict)

        elif name == "inner_prod":
            return None

    def decoder_func_factory(self, decoder_dict: Dict[str, str]) -> Callable:
        name = decoder_dict["name"]

        if name == "bilinear":
            return lambda b_z, g_z: b_z @ self.decoder @ g_z.t()

        elif name == "fc":

            def _fc_runner(b_z: Tensor, g_z: Tensor) -> Tensor:
                h = torch.cat(
                    [
                        b_z.unsqueeze(1).expand(-1, g_z.size(0), -1),
                        g_z.unsqueeze(0).expand(b_z.size(0), -1, -1),
                    ],
                    dim=-1,
                )
                h = self.decoder(h)
                return h.squeeze(-1)

            return _fc_runner

        elif name == "inner_prod":
            return lambda b_z, g_z: b_z @ g_z.t()

    def init_fc_decoder(self, decoder) -> nn.Sequential:
        bias: bool = decoder.get("bias", True)
        dp: Optional[float] = decoder.get("dropout", None)

        dc = nn.ModuleList()

        if dp is not None:
            dc.append(nn.Dropout(dp))
        dc.append(
            nn.Linear(
                in_features=self.hparams["dim_list"][-1] * 2, out_features=1, bias=bias
            )
        )
        dc = nn.Sequential(*dc)

        return dc

    # subclasses must implement this method
    def create_encoder_block(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError("Subclasses should implement this method.")

    # subclasses must override encode func
    def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
        """
        Encoding function, Child class must override this function

        Args:
            batch (PygBatch): batched data returned by PyG DataLoader

        Raises:
            NotImplementedError: Subclasses should implement this method.

        Returns:
            Tuple[Tensor, Tensor]: B_z, G_z
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def decode(
        self, B_z: Tensor, G_z: Tensor, batch: PygBatch
    ) -> Tuple[List[Tensor], List[Tensor]]:
        batch = batch.to(self.device)

        edge_index_bg_pred = []
        edge_index_bg_true = []

        edge_index_bg_dense = torch.zeros(batch.x_b.shape[0], batch.x_g.shape[0]).to(
            self.device
        )
        edge_index_bg_dense[batch.edge_index_bg[0], batch.edge_index_bg[1]] = 1

        node2graph_idx = torch.stack(
            [
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_b_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_g_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),
            ],
            dim=0,
        )

        for i in range(batch.num_graphs):
            edge_index_bg_pred.append(
                F.sigmoid(
                    self._dc_func(
                        b_z=B_z[batch.x_b_batch == i], g_z=G_z[batch.x_g_batch == i]
                    )
                )
            )
            edge_index_bg_true.append(
                edge_index_bg_dense[
                    node2graph_idx[0, i] : node2graph_idx[0, i + 1],
                    node2graph_idx[1, i] : node2graph_idx[1, i + 1],
                ]
            )

        return edge_index_bg_pred, edge_index_bg_true

    def forward(self, batch: PygBatch) -> Dict[str, Union[int, Tensor]]:
        batch = batch.to(self.device)
        z_ab, z_ag = self.encode(batch)
        edge_index_bg_pred, edge_index_bg_true = self.decode(z_ab, z_ag, batch)

        return {
            "abdbid": batch.abdbid,
            "edge_index_bg_pred": edge_index_bg_pred,
            "edge_index_bg_true": edge_index_bg_true,
        }


# GCN version of the model
class GCNAbAgInt(BaseAbAgInt):
    def __init__(
        self,
        input_ab_dim: int,
        input_ag_dim: int,
        dim_list: List[int],
        act_list: List[str],
        decoder: Optional[Dict] = None,
        try_gpu: bool = True,
        input_ab_act: str = "relu",
        input_ag_act: str = "relu",
        **kwargs,
    ):
        """
        GCN version of the AbAgInt model

        NOTE: length of dim_list must equal to length of act_list + 1
        Because we don't apply an activate function to the last layer (i.e. the output layer)

        Args:
            input_ab_dim (int): input antibody graph node feature dimension (e.g. if use AntiBERTy, this is 512)
            input_ag_dim (int): input antigen graph node feature dimension (e.g. if use ESM2, this is 480)
            dim_list (List[int]): a list of dimensions for the encoder layers, length is equal to `len(act_list) + 1`
            act_list (List[str]): a list of activation functions for the encoder layers
            decoder (Optional[Dict], optional): The type of decoder layer. Defaults to None.
                if None (default), use inner product decoder
                choices: ['inner_prod', 'fc', 'bilinear']
            try_gpu (bool, optional): try to use GPU. Defaults to True.
                if True and GPU is available, use GPU
            input_ab_act (str, optional): input activation function for antibody graph. Defaults to "relu".
            input_ag_act (str, optional): input activation function for antigen graph. Defaults to "relu".
        """
        super().__init__(
            input_ab_dim=input_ab_dim,
            input_ag_dim=input_ag_dim,
            dim_list=dim_list,
            act_list=act_list,
            decoder=decoder,
            try_gpu=try_gpu,
            input_ab_act=input_ab_act,
            input_ag_act=input_ag_act,
        )
        decoder = (
            {
                "name": "inner_prod",
            }
            if decoder is None
            else decoder
        )

        # encoder
        _default_conv_kwargs = {"normalize": True}  # DO NOT set cache to True
        self.B_encoder_block = self.create_encoder_block(
            node_feat_name="x_b",
            edge_index_name="edge_index_b",
            input_dim=input_ab_dim,
            input_act=input_ab_act,
            dim_list=dim_list,
            act_list=act_list,
            gcn_kwargs=_default_conv_kwargs,
        ).to(self.device)
        self.G_encoder_block = self.create_encoder_block(
            node_feat_name="x_g",
            edge_index_name="edge_index_g",
            input_dim=input_ag_dim,
            input_act=input_ag_act,
            dim_list=dim_list,
            act_list=act_list,
            gcn_kwargs=_default_conv_kwargs,
        ).to(self.device)

        # Decoder attr placeholder
        self.decoder = self.decoder_factory(self.hparams["decoder"])
        # if not inner_prod, move to device
        if self.hparams["decoder"]["name"] != "inner_prod":
            self.decoder.to(self.device)
        self._dc_func: Callable = self.decoder_func_factory(self.hparams["decoder"])

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

    def decode(
        self, B_z: Tensor, G_z: Tensor, batch: PygBatch
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Inner Product Decoder

        Args:
            z_ab: (Tensor)  shape (Nb, dim_latent)
            z_ag: (Tensor)  shape (Ng, dim_latent)

        Returns:
            A_reconstruct: (Tensor) shape (B, G)
                reconstructed bipartite adjacency matrix
        """
        # move batch to device
        batch = batch.to(self.device)

        edge_index_bg_pred = []
        edge_index_bg_true = []

        # dense bipartite edge index
        edge_index_bg_dense = torch.zeros(batch.x_b.shape[0], batch.x_g.shape[0]).to(
            self.device
        )
        edge_index_bg_dense[batch.edge_index_bg[0], batch.edge_index_bg[1]] = 1

        # get graph sizes (number of nodes) in the batch, used to slice the dense bipartite edge index
        node2graph_idx = torch.stack(
            [
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_b_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),  # (Nb+1, ) CDR     nodes
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_g_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),  # (Ng+1, ) antigen nodes
            ],
            dim=0,
        )

        for i in range(batch.num_graphs):
            edge_index_bg_pred.append(
                F.sigmoid(
                    self._dc_func(
                        b_z=B_z[batch.x_b_batch == i], g_z=G_z[batch.x_g_batch == i]
                    )
                )
            )  # Tensor (Nb, Ng)
            edge_index_bg_true.append(
                edge_index_bg_dense[
                    node2graph_idx[0, i] : node2graph_idx[0, i + 1],
                    node2graph_idx[1, i] : node2graph_idx[1, i + 1],
                ]
            )  # Tensor (Nb, Ng)

        return edge_index_bg_pred, edge_index_bg_true

    def forward(self, batch: PygBatch) -> Dict[str, Union[int, Tensor]]:
        # device
        batch = batch.to(self.device)
        # encode
        z_ab, z_ag = self.encode(batch)  # (Nb, C), (Ng, C)
        # decode
        edge_index_bg_pred, edge_index_bg_true = self.decode(z_ab, z_ag, batch)

        return {
            "abdbid": batch.abdbid,  # List[str]
            "edge_index_bg_pred": edge_index_bg_pred,  # List[Tensor (Nb, Ng)]
            "edge_index_bg_true": edge_index_bg_true,  # List[Tensor (Nb, Ng)]
        }


# ----------------------------------------
# Regression
# ----------------------------------------
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


class RegressionGCNAbAgIntLM(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.config = cfg

        # store the loss for each batch in an epoch
        self.training_loss_epoch = []
        self.training_metric_epoch = []
        self.validation_loss_epoch = []
        self.validation_metric_epoch = []
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

        # logger.info("Configuring metrics...")
        # self.metric_func_dict = self.configure_metric_func_dict()
        # logger.info(self.metric_func_dict)

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
        return {"mse": nn.MSELoss()}

    # --------------------------------------------------------------------------
    # Custom methods
    # --------------------------------------------------------------------------
    def compute_loss(
        self, batch: PygBatch, affinity_pred: Tensor, stage: str
    ) -> Dict[str, Tensor]:
        affinity_true = batch.y  # (B,)
        if affinity_true.ndim == 2:
            affinity_true = affinity_true.squeeze()
        if affinity_pred.ndim == 2:
            affinity_pred = affinity_pred.squeeze()
        loss_dict = {
            f"{stage}/loss/{k}": v(affinity_pred.float(), affinity_true.float())
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

    def _one_step(self, batch: PygBatch, stage: str) -> None:
        # forward
        affinity_pred = self.forward(batch)  # (B, 1)

        # compute loss
        loss_dict = self.compute_loss(batch, affinity_pred, stage)

        # NOTE: add key value pair `"loss": loss_value` the loss dict to return after each step
        # NOTE: required by Lightning
        # since we only use a single loss function, we only need to return the total
        loss_values = list(loss_dict.values())
        loss_dict["loss"] = loss_values[0]

        # # compute metrics
        # metric_dict = self.compute_metrics(batch, affinity_pred, stage)

        return loss_dict

    def training_step(self, batch: PygBatch, batch_idx: int) -> Tensor:
        loss_dict = self._one_step(batch, "train")
        # store loss for each batch in an epoch
        self.training_loss_epoch.append(loss_dict["loss"])
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        # log with sync_dist=True if distributed
        sync_dist = is_distributed
        self.log_step(loss_dict, sync_dist=sync_dist)
        return loss_dict["loss"]

    def validation_step(self, batch: PygBatch, batch_idx: int) -> Tensor:
        loss_dict = self._one_step(batch, "val")
        # store loss for each batch in an epoch
        self.validation_loss_epoch.append(loss_dict["loss"])
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        # log with sync_dist=True if distributed
        sync_dist = is_distributed
        self.log_step(loss_dict, sync_dist=sync_dist)
        return loss_dict["loss"]

    def test_step(self, batch: PygBatch, batch_idx: int) -> Tensor:
        """
        Test set is packed in a single batch. This function is called only once
        for the entire test set.
        """
        y_pred = self.forward(batch)
        loss_dict = self.compute_loss(batch, y_pred, "test")
        loss_value = list(loss_dict.values())[0]  # only one value
        # append loss value to the list
        self.test_loss_epoch.append(loss_value)
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        # log with sync_dist=True if distributed
        sync_dist = is_distributed
        self.log_step(loss_dict, sync_dist=sync_dist)
        # calculate the correlation between predicted and true affinity
        y_pred, y_true = y_pred.squeeze().cpu(), batch.y.squeeze().cpu()
        corr = PearsonCorrCoef()(y_pred, y_true).item()
        self.log_step(log_dict={"test/corr": corr}, sync_dist=sync_dist)
        # add results to the test results for later artifact logging
        self.test_results.update(
            {
                "loss": loss_value,
                "y_pred": y_pred,
                "y_true": y_true,
                "pearson_corr": corr,
            }
        )
        return loss_value

    def predict_step(self, batch: PygBatch) -> Tensor:
        y_pred = self.forward(batch)
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

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        # Log the average loss
        self.log_epoch(log_dict={"val/loss/avg": avg_loss}, sync_dist=sync_dist)

        # # Log the average loss to console
        # logger.info(f"Epoch {self.current_epoch} loss: {avg_loss}")

        # Clear lists for the next epoch
        self.validation_loss_epoch.clear()

    def on_test_epoch_end(self) -> None:
        avg_loss = torch.stack(self.test_loss_epoch).mean()

        # Check if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        # Log the average loss
        self.log_epoch(log_dict={"test/loss/avg": avg_loss}, sync_dist=sync_dist)

        # # Log the average loss to console
        # logger.info(f"Epoch {self.current_epoch} loss: {avg_loss}")

        # Clear lists for the next epoch
        self.test_loss_epoch.clear()
