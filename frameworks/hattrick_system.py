import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv#, GINConv 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch_scatter
import os
import sys
from torch.utils.checkpoint import checkpoint
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

epsilon = 1e-4

# Set Transformer
class TransformerModel(nn.Module):
    def __init__(self, in_dim: int, nhead: int, dim_feedforward: int,
                 nlayers: int, dropout: float = 0.0, activation="gelu"):
        super().__init__()
                
        encoder_layers = TransformerEncoderLayer(d_model=in_dim, nhead=nhead,
                            dim_feedforward=dim_feedforward, dropout=dropout,
                        batch_first=True, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.in_dim = in_dim
        
    def forward(self, src: Tensor, src_key_padding_mask: Tensor = None) -> Tensor:
        """
        Forward pass of the Transformer model.
        
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_dim), 
                                representing the source sequences.
            src_key_padding_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len) 
                                                        indicating which positions should be ignored 
                                                        in the source sequence. Default is None.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input, after being passed through 
                        the Transformer encoder.
        """
        
        if src_key_padding_mask is not None:
            src_key_padding_mask = (~src_key_padding_mask)
            
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output

# GNN of HARP
class GNN(nn.Module):
    def __init__(self, num_features, num_gnn_layers):
        super(GNN, self).__init__()
        self.num_features = num_features
                
        self.gnns = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                self.gnns.append(GCNConv(num_features, num_features+1))
            elif i == 1:
                self.gnns.append(GCNConv(num_features+1, num_features+2))
            else:
                self.gnns.append(GCNConv(num_features+2, num_features+2))
        self.output_dim = num_gnn_layers*(self.num_features+2) - 1
        
        
        
    def forward(self, node_features, edge_index, capacities):
        """
        Forward pass of the GNN model.

        Args:
            node_features (torch.Tensor): Node features for each graph in the batch, 
                                        shape (batch_size, num_nodes, num_features).
            edge_index (torch.Tensor): Edge indices defining the graph connectivity, 
                                    shape (2, num_edges).
            capacities (torch.Tensor): Edge capacities for each graph in the batch, 
                                    shape (batch_size, num_edges).

        Returns:
            torch.Tensor: Edge embeddings for each edge in the graph, with capacities included, 
                        shape (batch_size, num_edges, output_dim).

        Process:
            1. Iterate over the batch of node features and capacities.
            2. For each graph in the batch:
                a. Pass the node features through each GNN layer (GCNConv), applying Leaky ReLU activation.
                b. Collect the intermediate node embeddings after each GNN layer.
                c. Concatenate node embeddings from all GNN layers if more than one GNN layer is used.
            3. Stack the node embeddings for all graphs in the batch.
            4. Expand edge_index to match the batch size, so it can be used for batch processing.
            5. Use the expanded edge_index to extract edge embeddings from the node embeddings.
            6. Sum the node embeddings corresponding to each edge and concatenate them with the edge capacities.
        """


        batch_size = node_features.shape[0]
        ne_list = []
        for i in range(batch_size):
            nf = node_features[i]
            caps = capacities[i]
            sample_ne_list = []
            for j, gnn in enumerate(self.gnns):
                if j == 0:
                    ne = gnn(nf, edge_index=edge_index, edge_weight=caps)
                else:
                    ne = gnn(ne, edge_index=edge_index, edge_weight=caps)
                ne = F.leaky_relu(ne, 0.02)
                # ne = F.silu(ne)
                sample_ne_list.append(ne)
            if len(self.gnns) > 1:
                node_embeddings = torch.cat((sample_ne_list), dim=-1)
                ne_list.append(node_embeddings)
            else:
                ne_list = sample_ne_list
        node_embeddings = torch.stack(ne_list).contiguous()
        edge_index_expanded = edge_index.t().expand(batch_size, -1, -1)
        batch_size, num_nodes, feature_size = node_embeddings.shape
        _, num_edges, _ = edge_index_expanded.shape
        
        # Create a batch index
        batch_index = torch.arange(batch_size).view(-1, 1, 1)
        batch_index = batch_index.expand(-1, num_edges, 2)  # Repeat the batch index for each edge
        edge_embeddings = node_embeddings[batch_index, edge_index_expanded]
        capacities = capacities.unsqueeze(-1)
        edge_embeddings = edge_embeddings.sum(dim=-2)
        edge_embeddings = torch.cat((edge_embeddings, capacities), dim=-1)
        
        return edge_embeddings
        
class Hattrick(nn.Module):
    
    def __init__(self, props):
        
        super(Hattrick, self).__init__()
        
        # Define the architecture of HARP
        self.num_gnn_layers = props.num_gnn_layers
        self.num_transformer_layers = props.num_transformer_layers
        self.dropout = props.dropout
        self.num_mlp1_hidden_layers = props.num_mlp1_hidden_layers
        self.num_mlp2_hidden_layers = props.num_mlp2_hidden_layers
        self.device = props.device
        self.topo = props.topo
        # Define the GNN
        self.gnn = GNN(2, self.num_gnn_layers)
        
        self.input_dim = self.gnn.output_dim + 1
                
        # CLS Token for the Set Transformer
        self.cls_token = nn.Parameter(torch.Tensor(1, self.input_dim))
        nn.init.kaiming_normal_(self.cls_token, nonlinearity='relu')
        
        if props.num_heads == 0:
            if props.num_gnn_layers == 1:
                num_heads = 2
            else:
                num_heads = self.input_dim//4
        else:
            num_heads = props.num_heads
        
        # Define the Set Transformer
        self.transformer = TransformerModel(in_dim = self.input_dim, nhead=num_heads,
                            dim_feedforward=self.input_dim, nlayers=self.num_transformer_layers, 
                            dropout=self.dropout, activation="gelu")
        
        ##########################################################################################
        # Define the 1st MLP
        self.mlp_11_dim = self.input_dim + 1
        self.mlp11 = nn.ModuleList()
        self.mlp11.append(nn.Linear(self.mlp_11_dim, self.mlp_11_dim))
        for i in range(self.num_mlp1_hidden_layers):
            self.mlp11.append(nn.Linear(self.mlp_11_dim, self.mlp_11_dim))
        self.mlp11.append(nn.Linear(self.mlp_11_dim, 1))
                
        
        self.mlp_12_dim = self.input_dim + 3 + 1
        self.mlp12 = nn.ModuleList()
        self.mlp12.append(nn.Linear(self.mlp_12_dim, self.mlp_12_dim))
        for i in range(self.num_mlp2_hidden_layers):
            self.mlp12.append(nn.Linear(self.mlp_12_dim, self.mlp_12_dim))
        self.mlp12.append(nn.Linear(self.mlp_12_dim, 1))
        ### HIGH CLASS MLPs ARE DONE
        
        ##########################################################################################
        self.mlp_21_dim = self.input_dim + 1 + 1 + 1
        self.mlp21 = nn.ModuleList()
        self.mlp21.append(nn.Linear(self.mlp_21_dim, self.mlp_21_dim))
        for i in range(self.num_mlp1_hidden_layers):
            self.mlp21.append(nn.Linear(self.mlp_21_dim, self.mlp_21_dim))
        self.mlp21.append(nn.Linear(self.mlp_21_dim, 2))
        
        
        self.mlp_22_dim = self.input_dim*2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 2
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        if props.violation:
            self.mlp_22_violation = \
            nn.Sequential(
                nn.Linear(1, 3),
                self.activation,
                nn.Linear(3, 6),
                self.activation,
                nn.Linear(6, 12),
                self.activation,
                nn.Linear(12, self.mlp_22_dim))
                
        
        self.mlp22 = nn.ModuleList()
        self.mlp22.append(nn.Linear(self.mlp_22_dim, self.mlp_22_dim))
        for i in range(self.num_mlp2_hidden_layers):
            self.mlp22.append(nn.Linear(self.mlp_22_dim, self.mlp_22_dim))
        self.mlp22.append(nn.Linear(self.mlp_22_dim, 2))
        ### MED CLASS MLPs ARE DONE
        
        ##########################################################################################
        self.mlp_31_dim = self.input_dim + 1 + 1 + 1 + 1 + 1 + 1
        self.mlp31 = nn.ModuleList()
        self.mlp31.append(nn.Linear(self.mlp_31_dim, self.mlp_31_dim))
        for i in range(self.num_mlp1_hidden_layers):
            self.mlp31.append(nn.Linear(self.mlp_31_dim, self.mlp_31_dim))
        self.mlp31.append(nn.Linear(self.mlp_31_dim, 3))
        self.mlp_32_dim = self.input_dim*3 + 3 + 1 + 1 + 1 + 1 + 3 + 2 + 3 + 3
                
        if props.violation:
            self.mlp_32_violation = \
                            nn.Sequential(
                nn.Linear(1, 3),
                self.activation,
                nn.Linear(3, 6),
                self.activation,
                nn.Linear(6, 12),
                self.activation,
                nn.Linear(12, self.mlp_32_dim))
        
        self.mlp32 = nn.ModuleList()
        self.mlp32.append(nn.Linear(self.mlp_32_dim, self.mlp_32_dim))
        for i in range(self.num_mlp2_hidden_layers):
            self.mlp32.append(nn.Linear(self.mlp_32_dim, self.mlp_32_dim))
        self.mlp32.append(nn.Linear(self.mlp_32_dim, 3))
        
        
        
    def forward(self, props, node_features, edge_index, capacities, padded_edge_ids_per_path,
                tm1, tm1_pred, tm2, tm2_pred, tm3, tm3_pred, paths_to_edges, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor):
        """
        Run the full Hattrick forward pass to predict path split ratios and edge utilizations
        for three traffic classes by combining GNN, set-transformer, and MLP-based RAU stages.

        High-level flow:
        - Encode nodes with a GNN (capacities as edge weights) to obtain per-edge embeddings.
        - Build per-path sequences of edge embeddings, prepend a CLS token, and encode with a
          set transformer to get path embeddings (CLS) and conditioned edge embeddings.
        - Stage 1 (class-1): predict initial path scores (gammas) and refine them through an
          iterative RAU loop using bottleneck-aware features and MLU.
        - Stage 2 (class-1+2): predict and refine class-2 gammas conditioned on stage-1.
        - Stage 3 (class-1+2+3): predict and refine class-3 gammas conditioned on stages 1–2.
        - Convert gammas to split ratios and simulate sequential class admission to respect link
          capacities. Compute per-edge utilizations per class and aggregate metrics.

        Args:
            props: Configuration object/namespace with fields including (not exhaustive):
                - device, dtype, mode in {"train","test"}, dynamic (bool), checkpoint (int),
                  violation (bool), rate_cap (bool), sim_mf_mlu (bool), num_paths_per_pair (int),
                  rau1/rau2/rau3 (ints for RAU iterations), and transformer/GNN sizes.
            node_features (Tensor): Node features [B, N, F_node].
            edge_index (LongTensor): Graph edges (COO) [2, E].
            capacities (Tensor): Link capacities per batch [B, E].
            padded_edge_ids_per_path (LongTensor): Padded edge IDs per path [B, P, Lmax].
            tm1, tm1_pred, tm2, tm2_pred, tm3, tm3_pred (Tensor): Actual and predicted
                traffic matrices per path group [B, P, 1].
            paths_to_edges (torch.sparse_coo_tensor): Sparse mapping path→edges [P, E].
            edge_ids_dict_tensor (Dict[int, LongTensor]): Groups paths by (length−1) → edge-id tensor
                for efficient transformer batching.
            original_pos_edge_ids_dict_tensor (Dict[int, LongTensor]): Indices to scatter transformer
                outputs back to original path order.

        Returns:
            If props.sim_mf_mlu is True:
                Tuple(Tensor, Tensor, Tensor): Admitted traffic per path for classes 1..3,
                each of shape [B, P].
            Else if props.mode == "train":
                Tuple(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
                - edges_util_tm1: Max-MLU-across-stages utilization for class-1 [B, E]
                - edges_util_tm2: Max-MLU-across-stages utilization for class-12 [B, E]
                - edges_util_tm3_3: Stage #3 MLU for class-123 [B, E]
                - edges_util_tm1_3: Stage #3 MLU for class-1 [B, E]
                - edges_util_tm2_3: Stage #3 MLU for class-12 [B, E]
                - all_traffic: Total admitted traffic per path (sum over classes) [B, P]
            Else (test mode):
                Tuple(Tensor, Tensor, Tensor): Final per-edge utilizations for classes 1..3,
                each [B, E].

        Notes:
        - Uses checkpointing as per props.checkpoint to save memory.
        - Caches transformer outputs in test mode when props.dynamic is False.
        - All computations honor props.dtype and may mix sparse/dense operations.
        """
        
                
        num_paths_per_pair = props.num_paths_per_pair
        if props.checkpoint >= 1:
            edge_embeddings_with_caps = checkpoint(self.gnn, node_features, edge_index, capacities, use_reentrant=False)
        else:
            edge_embeddings_with_caps = self.gnn(node_features, edge_index, capacities)
        batch_size = tm1.shape[0]
        if props.dynamic:
            batch_size_tf = batch_size
        else:
            batch_size_tf = 1
        total_number_of_paths = paths_to_edges.shape[0]
        cls_token = self.cls_token.unsqueeze(0)
        if props.mode == "train" or (props.mode == "test" and not hasattr(self, "transformer_output")):
            if props.checkpoint >= 1:
                transformer_output = checkpoint(self.compute_transformer_output, edge_embeddings_with_caps, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor, batch_size_tf, total_number_of_paths, props, cls_token, use_reentrant=False)
            else:
                transformer_output = self.compute_transformer_output(edge_embeddings_with_caps, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor, batch_size_tf, total_number_of_paths, props, cls_token)
        else:
            pass

        if props.mode == "train":
            if not props.dynamic:
                transformer_output = transformer_output.expand(batch_size, -1, -1, -1)
                capacities = capacities.expand(batch_size, -1)
        elif (props.mode == "test" and not hasattr(self, "transformer_output") and not props.dynamic):
            capacities = capacities.expand(batch_size, -1)
            self.transformer_output = transformer_output.expand(batch_size, -1, -1, -1)  
        if props.mode == "train":
            path_embeddings = transformer_output[:, :, 0, :]
            path_edge_embeddings = transformer_output[:, :, 1:, :]
        elif props.mode == "test":
            if not props.dynamic:
                path_embeddings = self.transformer_output[:, :, 0, :]
                path_edge_embeddings = self.transformer_output[:, :, 1:, :]
            else:
                path_embeddings = transformer_output[:, :, 0, :]
                path_edge_embeddings = transformer_output[:, :, 1:, :]
        paths_to_edges = paths_to_edges.coalesce()
        indices = paths_to_edges.indices()
        values = paths_to_edges.values()
        row_indices = indices[0]
        col_indices = indices[1]

        pte_info = [paths_to_edges, row_indices, col_indices, values]
        
        ############################################################################
        # Predicted matrix 
        if props.checkpoint:
            gammas_1 = checkpoint(self.forward_pass_mlp, torch.cat((path_embeddings, tm1_pred), dim=-1), self.mlp11, self.num_mlp1_hidden_layers, use_reentrant=False)
        else:
            gammas_1 = self.forward_pass_mlp(torch.cat((path_embeddings, tm1_pred), dim=-1), self.mlp11, self.num_mlp1_hidden_layers)
        for i in range(props.rau1):
            if i > 0:
                gammas_1 = new_gammas_1
                        
            if props.checkpoint >= 2:
                edges_util_tm1_pred_1, _, split_ratios_1 = checkpoint(self.compute_edge_utils, gammas_1, paths_to_edges, tm1_pred, capacities, props, 
                                                               batch_size, num_paths_per_pair, add_epsilon=True, use_reentrant=False)
                mlu11 = checkpoint(self.compute_mlu, edges_util_tm1_pred_1, batch_size, total_number_of_paths, subtract_epsilon=True, use_reentrant=False)
                c1_bottleneck_path_edge_embeddings, c1_max_utilization_per_path, _ = checkpoint(self.compute_bottleneck_link_mlu_per_path, edges_util_tm1_pred_1, padded_edge_ids_per_path,
                                                                                            path_edge_embeddings, batch_size, total_number_of_paths, pte_info, use_reentrant=False)
                
            else:
                edges_util_tm1_pred_1, _, split_ratios_1 = self.compute_edge_utils(gammas_1, paths_to_edges, tm1_pred, capacities, props, 
                                                               batch_size, num_paths_per_pair, add_epsilon=True)
                mlu11 = self.compute_mlu(edges_util_tm1_pred_1, batch_size, total_number_of_paths, subtract_epsilon=True)
                c1_bottleneck_path_edge_embeddings, c1_max_utilization_per_path, _ = self.compute_bottleneck_link_mlu_per_path(edges_util_tm1_pred_1, padded_edge_ids_per_path,
                                                                                            path_edge_embeddings, batch_size, total_number_of_paths, pte_info)
            
            c1_rau_inputs = torch.cat((c1_bottleneck_path_edge_embeddings, 
                                      c1_max_utilization_per_path,
                                      mlu11,
                                      tm1_pred,
                                      split_ratios_1), dim=-1).squeeze(0)
            if props.checkpoint >= 1:
                delta_gammas_1 = checkpoint(self.forward_pass_mlp, c1_rau_inputs, self.mlp12, self.num_mlp2_hidden_layers, use_reentrant=False)
            else:
                delta_gammas_1 = self.forward_pass_mlp(c1_rau_inputs, self.mlp12, self.num_mlp2_hidden_layers)
            gammas_1 = gammas_1.reshape(batch_size, -1, 1)
            new_gammas_1 = delta_gammas_1 + gammas_1

        if props.checkpoint >= 2:
            edges_util_tm1_1, _, __ = checkpoint(self.compute_edge_utils, new_gammas_1, paths_to_edges, tm1, capacities, props,
                                                                   batch_size, num_paths_per_pair, False, use_reentrant=False)
            edges_util_tm1_pred_1, _, __ = checkpoint(self.compute_edge_utils, new_gammas_1, paths_to_edges, tm1_pred, capacities, props,
                                                                   batch_size, num_paths_per_pair, False, use_reentrant=False)
            mlu11 = checkpoint(self.compute_mlu, edges_util_tm1_pred_1, batch_size, total_number_of_paths, subtract_epsilon=False, use_reentrant=False)
        else:
            edges_util_tm1_1, _, __ = self.compute_edge_utils(new_gammas_1, paths_to_edges, tm1, capacities, props,
                                                                    batch_size, num_paths_per_pair, False)
            edges_util_tm1_pred_1, _, __ = self.compute_edge_utils(new_gammas_1, paths_to_edges, tm1_pred, capacities, props,
                                                                    batch_size, num_paths_per_pair, False)
            mlu11 = self.compute_mlu(edges_util_tm1_pred_1, batch_size, total_number_of_paths, subtract_epsilon=False)

        ############################################################################
        path_embeddings_2 = torch.cat((path_embeddings, tm1_pred, tm2_pred, mlu11), dim=-1)
        if props.checkpoint >= 1:
            gammas_2 = checkpoint(self.forward_pass_mlp, path_embeddings_2, self.mlp21, self.num_mlp1_hidden_layers, use_reentrant=False)
        else:
            gammas_2 = self.forward_pass_mlp(path_embeddings_2, self.mlp21, self.num_mlp1_hidden_layers)
        gammas_2[:, :, :1] = gammas_2[:, :, :1] + new_gammas_1
        for i in range(props.rau2):
            if i > 0:
                gammas_2 = new_gammas_2
            gammas_12, gammas_22 = gammas_2[:, :, :1], gammas_2[:, :, 1:]
            
            if props.violation:
                if props.checkpoint >= 2:
                    edges_util_tm1_pred_2, data_on_tunnels_1, split_ratios_1 = checkpoint(self.compute_edge_utils, gammas_12, paths_to_edges, tm1_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, violation=True, use_reentrant=False)
                else:
                    edges_util_tm1_pred_2, data_on_tunnels_1, split_ratios_1 = self.compute_edge_utils(gammas_12, paths_to_edges, tm1_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, violation=True)
            else:
                if props.checkpoint >= 2:
                    edges_util_tm1_pred_2, _, split_ratios_1 = checkpoint(self.compute_edge_utils, gammas_12, paths_to_edges, tm1_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, use_reentrant=False)
                else:
                    edges_util_tm1_pred_2, _, split_ratios_1 = self.compute_edge_utils(gammas_12, paths_to_edges, tm1_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True)

    
            if props.checkpoint >= 2:
                edges_util_tm2_pred_2, _, split_ratios_2 = checkpoint(self.compute_edge_utils, gammas_22, paths_to_edges, tm2_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, use_reentrant=False)
            else:
                edges_util_tm2_pred_2, _, split_ratios_2 = self.compute_edge_utils(gammas_22, paths_to_edges, tm2_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True)                

            edges_util_tm2_pred_2 = edges_util_tm2_pred_2 + edges_util_tm1_pred_2
            if props.checkpoint >= 2:
                mlu12 = checkpoint(self.compute_mlu, edges_util_tm1_pred_2, batch_size, total_number_of_paths, subtract_epsilon=True, use_reentrant=False)
                mlu22 = checkpoint(self.compute_mlu, edges_util_tm2_pred_2, batch_size, total_number_of_paths, subtract_epsilon=True, use_reentrant=False)
                
                c1_bottleneck_path_edge_embeddings, c1_max_utilization_per_path, capacities_of_bottleneck_edges_1 = checkpoint(self.compute_bottleneck_link_mlu_per_path, edges_util_tm1_pred_2, padded_edge_ids_per_path,
                                                                                        path_edge_embeddings, batch_size, total_number_of_paths, pte_info, capacities, use_reentrant=False)
                c2_bottleneck_path_edge_embeddings, c2_max_utilization_per_path, _ = checkpoint(self.compute_bottleneck_link_mlu_per_path, edges_util_tm2_pred_2, padded_edge_ids_per_path,
                                                                                        path_edge_embeddings, batch_size, total_number_of_paths, pte_info, use_reentrant=False)

            else:
                mlu12 = self.compute_mlu(edges_util_tm1_pred_2, batch_size, total_number_of_paths, subtract_epsilon=True)
                mlu22 = self.compute_mlu(edges_util_tm2_pred_2, batch_size, total_number_of_paths, subtract_epsilon=True)
            
                c1_bottleneck_path_edge_embeddings, c1_max_utilization_per_path, capacities_of_bottleneck_edges_1 = self.compute_bottleneck_link_mlu_per_path(edges_util_tm1_pred_2, padded_edge_ids_per_path,
                                                                                            path_edge_embeddings, batch_size, total_number_of_paths, pte_info, capacities)
                
                c2_bottleneck_path_edge_embeddings, c2_max_utilization_per_path, _ = self.compute_bottleneck_link_mlu_per_path(edges_util_tm2_pred_2, padded_edge_ids_per_path,
                                                                                            path_edge_embeddings, batch_size, total_number_of_paths, pte_info)
            
            c2_rau_inputs = torch.cat((mlu11,
                                       tm1_pred,
                                       c1_bottleneck_path_edge_embeddings,
                                       c1_max_utilization_per_path,
                                       mlu12,
                                       split_ratios_1,
                                       tm2_pred,
                                       c2_bottleneck_path_edge_embeddings, 
                                       c2_max_utilization_per_path,
                                       mlu22,
                                       split_ratios_2,
                                      ), dim=-1).squeeze(0)
            
            if props.violation:
                violation_metric_mlu1_stage_2 = (data_on_tunnels_1.unsqueeze(-1)/capacities_of_bottleneck_edges_1.unsqueeze(-1))/ mlu11.detach()
                if props.checkpoint >= 1:
                    violation_metric_mlu1_stage_2 = checkpoint(self.mlp_22_violation, violation_metric_mlu1_stage_2, use_reentrant=False)
                else:
                    violation_metric_mlu1_stage_2 = self.mlp_22_violation(violation_metric_mlu1_stage_2)
                if props.checkpoint >= 1:
                    delta_gammas_2 = checkpoint(self.forward_pass_mlp, c2_rau_inputs, self.mlp22, self.num_mlp2_hidden_layers, violation_data=[violation_metric_mlu1_stage_2], use_reentrant=False)
                else:
                    delta_gammas_2 = self.forward_pass_mlp(c2_rau_inputs, self.mlp22, self.num_mlp2_hidden_layers, violation_data=[violation_metric_mlu1_stage_2])
            else:
                if props.checkpoint >= 1:
                    delta_gammas_2 = checkpoint(self.forward_pass_mlp, c2_rau_inputs, self.mlp22, self.num_mlp2_hidden_layers, use_reentrant=False)
                else:
                    delta_gammas_2 = self.forward_pass_mlp(c2_rau_inputs, self.mlp22, self.num_mlp2_hidden_layers)
            new_gammas_2 = delta_gammas_2 + gammas_2
            
        gammas_12, gammas_22 = new_gammas_2[:, :, :1], new_gammas_2[:, :, 1:]
        
        if props.checkpoint >= 2:
            edges_util_tm1_2, _, __ = checkpoint(self.compute_edge_utils, gammas_12, paths_to_edges, tm1, capacities, props,
                                                                    batch_size, num_paths_per_pair, False, use_reentrant=False)
            
            edges_util_tm2_2, _, __ = checkpoint(self.compute_edge_utils, gammas_22, paths_to_edges, tm2, capacities, props,
                                                                    batch_size, num_paths_per_pair, False, use_reentrant=False)
            
            
            edges_util_tm1_pred_2, _, __ = checkpoint(self.compute_edge_utils, gammas_12, paths_to_edges, tm1_pred, capacities, props, 
                                                                batch_size, num_paths_per_pair, add_epsilon=False, use_reentrant=False)
            
            edges_util_tm2_pred_2, _, __ = checkpoint(self.compute_edge_utils, gammas_22, paths_to_edges, tm2_pred, capacities, props,
                                                                batch_size, num_paths_per_pair, add_epsilon=False, use_reentrant=False)

        else:
            edges_util_tm1_2, _, __ = self.compute_edge_utils(gammas_12, paths_to_edges, tm1, capacities, props,
                                                                    batch_size, num_paths_per_pair, False)
            
            edges_util_tm2_2, _, __ = self.compute_edge_utils(gammas_22, paths_to_edges, tm2, capacities, props,
                                                                    batch_size, num_paths_per_pair, False)
            
            
            edges_util_tm1_pred_2, _, __ = self.compute_edge_utils(gammas_12, paths_to_edges, tm1_pred, capacities, props, 
                                                                batch_size, num_paths_per_pair, add_epsilon=False)
            
            edges_util_tm2_pred_2, _, __ = self.compute_edge_utils(gammas_22, paths_to_edges, tm2_pred, capacities, props,
                                                                batch_size, num_paths_per_pair, add_epsilon=False)            

        edges_util_tm2_2 = edges_util_tm2_2 + edges_util_tm1_2
        edges_util_tm2_pred_2 = edges_util_tm2_pred_2 + edges_util_tm1_pred_2
        if props.checkpoint >= 2:
            mlu12 = checkpoint(self.compute_mlu, edges_util_tm1_pred_2, batch_size, total_number_of_paths, subtract_epsilon=False, use_reentrant=False)
            mlu22 = checkpoint(self.compute_mlu, edges_util_tm2_pred_2, batch_size, total_number_of_paths, subtract_epsilon=False, use_reentrant=False)
        else:
            mlu12 = self.compute_mlu(edges_util_tm1_pred_2, batch_size, total_number_of_paths, subtract_epsilon=False)
            mlu22 = self.compute_mlu(edges_util_tm2_pred_2, batch_size, total_number_of_paths, subtract_epsilon=False)
            
        #########################################################################################
        
        path_embeddings_3 = torch.cat((path_embeddings, tm1_pred, tm2_pred, tm3_pred, mlu12, mlu22, mlu11), dim=-1)
        if props.checkpoint >= 1:
            gammas_3 = checkpoint(self.forward_pass_mlp, path_embeddings_3, self.mlp31, self.num_mlp1_hidden_layers, use_reentrant=False)
        else:
            gammas_3 = self.forward_pass_mlp(path_embeddings_3, self.mlp31, self.num_mlp1_hidden_layers)
        gammas_3[:, :, :1] = gammas_3[:, :, :1] + gammas_12
        gammas_3[:, :, 1:2] = gammas_3[:, :, 1:2] + gammas_22
        
        for i in range(props.rau3):
            if i > 0:
                gammas_3 = new_gammas_3
            gammas_13, gammas_23, gammas_33 = gammas_3[:, :, :1], gammas_3[:, :, 1:2], gammas_3[:, :, 2:]
            
            if props.violation:
                if props.checkpoint >= 2:
                    edges_util_tm1_pred_3, data_on_tunnels_1, split_ratios_1 = checkpoint(self.compute_edge_utils, gammas_13, paths_to_edges, tm1_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, violation=True, use_reentrant=False)        
                    edges_util_tm2_pred_3, data_on_tunnels_2, split_ratios_2 = checkpoint(self.compute_edge_utils, gammas_23, paths_to_edges, tm2_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, violation=True, use_reentrant=False)
                else:
                    edges_util_tm1_pred_3, data_on_tunnels_1, split_ratios_1 = self.compute_edge_utils(gammas_13, paths_to_edges, tm1_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, violation=True)        
                    edges_util_tm2_pred_3, data_on_tunnels_2, split_ratios_2 = self.compute_edge_utils(gammas_23, paths_to_edges, tm2_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, violation=True)
                data_on_tunnels_2 = data_on_tunnels_2 + data_on_tunnels_1
            else:
                if props.checkpoint >= 2:
                    edges_util_tm1_pred_3, _, split_ratios_1 = checkpoint(self.compute_edge_utils, gammas_13, paths_to_edges, tm1_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, use_reentrant=False)        
                    edges_util_tm2_pred_3, _, split_ratios_2 = checkpoint(self.compute_edge_utils, gammas_23, paths_to_edges, tm2_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, use_reentrant=False)
                else:
                    edges_util_tm1_pred_3, _, split_ratios_1 = self.compute_edge_utils(gammas_13, paths_to_edges, tm1_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True)        
                    edges_util_tm2_pred_3, _, split_ratios_2 = self.compute_edge_utils(gammas_23, paths_to_edges, tm2_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True)
            
            if props.checkpoint >= 2:
                edges_util_tm3_pred_3, _, split_ratios_3 = checkpoint(self.compute_edge_utils, gammas_33, paths_to_edges, tm3_pred, capacities, props,
                                                                   batch_size, num_paths_per_pair, add_epsilon=True, use_reentrant=False)
            else:
                edges_util_tm3_pred_3, _, split_ratios_3 = self.compute_edge_utils(gammas_33, paths_to_edges, tm3_pred, capacities, props,
                                                                   batch_size, num_paths_per_pair, add_epsilon=True)
            
            edges_util_tm2_pred_3 = edges_util_tm2_pred_3 + edges_util_tm1_pred_3
            edges_util_tm3_pred_3 = edges_util_tm3_pred_3 + edges_util_tm2_pred_3

            if props.checkpoint >= 2:
                mlu13 = checkpoint(self.compute_mlu, edges_util_tm1_pred_3, batch_size, total_number_of_paths, subtract_epsilon=True, use_reentrant=False)
                mlu23 = checkpoint(self.compute_mlu, edges_util_tm2_pred_3, batch_size, total_number_of_paths, subtract_epsilon=True, use_reentrant=False)
                mlu33 = checkpoint(self.compute_mlu, edges_util_tm3_pred_3, batch_size, total_number_of_paths, subtract_epsilon=True, use_reentrant=False)
            else:
                mlu13 = self.compute_mlu(edges_util_tm1_pred_3, batch_size, total_number_of_paths, subtract_epsilon=True)
                mlu23 = self.compute_mlu(edges_util_tm2_pred_3, batch_size, total_number_of_paths, subtract_epsilon=True)
                mlu33 = self.compute_mlu(edges_util_tm3_pred_3, batch_size, total_number_of_paths, subtract_epsilon=True)
            
            if props.checkpoint >= 2:
                c1_bottleneck_path_edge_embeddings, c1_max_utilization_per_path, capacities_of_bottleneck_edges_1 = checkpoint(self.compute_bottleneck_link_mlu_per_path, edges_util_tm1_pred_3, padded_edge_ids_per_path,
                                                                                        path_edge_embeddings, batch_size, total_number_of_paths, pte_info, capacities, use_reentrant=False)
                c2_bottleneck_path_edge_embeddings, c2_max_utilization_per_path, capacities_of_bottleneck_edges_2 = checkpoint(self.compute_bottleneck_link_mlu_per_path, edges_util_tm2_pred_3, padded_edge_ids_per_path,
                                                                                        path_edge_embeddings, batch_size, total_number_of_paths, pte_info, capacities, use_reentrant=False)
                c3_bottleneck_path_edge_embeddings, c3_max_utilization_per_path, _ = checkpoint(self.compute_bottleneck_link_mlu_per_path, edges_util_tm3_pred_3, padded_edge_ids_per_path,
                                                                                            path_edge_embeddings, batch_size, total_number_of_paths, pte_info, capacities, use_reentrant=False)
            else:
                c1_bottleneck_path_edge_embeddings, c1_max_utilization_per_path, capacities_of_bottleneck_edges_1 = self.compute_bottleneck_link_mlu_per_path(edges_util_tm1_pred_3, padded_edge_ids_per_path,
                                                                                        path_edge_embeddings, batch_size, total_number_of_paths, pte_info, capacities)
                c2_bottleneck_path_edge_embeddings, c2_max_utilization_per_path, capacities_of_bottleneck_edges_2 = self.compute_bottleneck_link_mlu_per_path(edges_util_tm2_pred_3, padded_edge_ids_per_path,
                                                                                        path_edge_embeddings, batch_size, total_number_of_paths, pte_info, capacities)
                c3_bottleneck_path_edge_embeddings, c3_max_utilization_per_path, _ = self.compute_bottleneck_link_mlu_per_path(edges_util_tm3_pred_3, padded_edge_ids_per_path,
                                                                                                path_edge_embeddings, batch_size, total_number_of_paths, pte_info, capacities)
            
            split_ratios = [split_ratios_1, split_ratios_2, split_ratios_3]
            tms = [tm1_pred, tm2_pred, tm3_pred]
            
            if props.checkpoint >= 2:
                c1_split_ratios, c2_split_ratios, c3_split_ratios, _ = checkpoint(self.simulate, split_ratios, tms, capacities, pte_info, batch_size, props, rate_cap=props.rate_cap, use_reentrant=False)
            else:
                c1_split_ratios, c2_split_ratios, c3_split_ratios, _ = self.simulate(split_ratios, tms, capacities, pte_info, batch_size, props, rate_cap=props.rate_cap)
            
            if props.dtype == torch.bfloat16:
                c1_split_ratios = c1_split_ratios.to(props.dtype)
                c2_split_ratios = c2_split_ratios.to(props.dtype)
                c3_split_ratios = c3_split_ratios.to(props.dtype)
            
            c3_rau_inputs = torch.cat((tm1_pred,
                                       c1_bottleneck_path_edge_embeddings,
                                       c1_max_utilization_per_path,
                                       mlu13,
                                       split_ratios_1,
                                       c1_split_ratios.unsqueeze(-1),
                                       tm2_pred,
                                       c2_bottleneck_path_edge_embeddings,
                                       c2_max_utilization_per_path,
                                       mlu23,
                                       split_ratios_2,
                                       c2_split_ratios.unsqueeze(-1),
                                       tm3_pred,                                     
                                       c3_bottleneck_path_edge_embeddings, 
                                       c3_max_utilization_per_path,
                                       mlu33,
                                       split_ratios_3,
                                       c3_split_ratios.unsqueeze(-1),
                                       mlu11,
                                       mlu12,
                                       mlu22,
                                      ), dim=-1).squeeze(0)                            
            c3_rau_inputs = c3_rau_inputs.to(dtype=props.dtype)
            if props.violation:
                violation_metric_mlu1_stage_3 = (data_on_tunnels_1.unsqueeze(-1)/capacities_of_bottleneck_edges_1.unsqueeze(-1))/mlu11.detach()
                violation_metric_mlu2_stage_3 = (data_on_tunnels_2.unsqueeze(-1)/capacities_of_bottleneck_edges_2.unsqueeze(-1))/mlu22.detach()
                if props.checkpoint >= 1:
                    violation_metric_mlu1_stage_3 = checkpoint(self.mlp_32_violation, violation_metric_mlu1_stage_3, use_reentrant=False)
                    violation_metric_mlu2_stage_3 = checkpoint(self.mlp_32_violation, violation_metric_mlu2_stage_3, use_reentrant=False)
                else:
                    violation_metric_mlu1_stage_3 = self.mlp_32_violation(violation_metric_mlu1_stage_3)
                    violation_metric_mlu2_stage_3 = self.mlp_32_violation(violation_metric_mlu2_stage_3)
                if props.checkpoint >= 1:
                    delta_gammas_3 = checkpoint(self.forward_pass_mlp, c3_rau_inputs, self.mlp32, self.num_mlp2_hidden_layers, [violation_metric_mlu1_stage_3, violation_metric_mlu2_stage_3], use_reentrant=False)
                else:
                    delta_gammas_3 = self.forward_pass_mlp(c3_rau_inputs, self.mlp32, self.num_mlp2_hidden_layers, [violation_metric_mlu1_stage_3, violation_metric_mlu2_stage_3])
            else:
                if props.checkpoint >= 1:
                    delta_gammas_3 = checkpoint(self.forward_pass_mlp, c3_rau_inputs, self.mlp32, self.num_mlp2_hidden_layers, use_reentrant=False)
                else:
                    delta_gammas_3 = self.forward_pass_mlp(c3_rau_inputs, self.mlp32, self.num_mlp2_hidden_layers)
            # delta_gammas_3 = self.forward_pass_mlp(c3_rau_inputs, self.mlp32, self.num_mlp2_hidden_layers)
            new_gammas_3 = delta_gammas_3 + gammas_3
        
        gammas_13, gammas_23, gammas_33 = new_gammas_3[:, :, :1], new_gammas_3[:, :, 1:2], new_gammas_3[:, :, 2:]

        if props.checkpoint >= 2:
            edges_util_tm1_3, _, split_ratios_1 = checkpoint(self.compute_edge_utils, gammas_13, paths_to_edges, tm1, capacities, props,
                                                                   batch_size, num_paths_per_pair, False, use_reentrant=False)
            edges_util_tm2_3, _, split_ratios_2 = checkpoint(self.compute_edge_utils, gammas_23, paths_to_edges, tm2, capacities, props,
                                                                   batch_size, num_paths_per_pair, False, use_reentrant=False)        
            edges_util_tm3_3, _, split_ratios_3 = checkpoint(self.compute_edge_utils, gammas_33, paths_to_edges, tm3, capacities, props,
                                                                   batch_size, num_paths_per_pair, False, use_reentrant=False)
        else:
            edges_util_tm1_3, _, split_ratios_1 = self.compute_edge_utils(gammas_13, paths_to_edges, tm1, capacities, props,
                                                                   batch_size, num_paths_per_pair, False)
            edges_util_tm2_3, _, split_ratios_2 = self.compute_edge_utils(gammas_23, paths_to_edges, tm2, capacities, props,
                                                                   batch_size, num_paths_per_pair, False)        
            edges_util_tm3_3, _, split_ratios_3 = self.compute_edge_utils(gammas_33, paths_to_edges, tm3, capacities, props,
                                                                   batch_size, num_paths_per_pair, False)

        split_ratios = [split_ratios_1, split_ratios_2, split_ratios_3]
        tms = [tm1, tm2, tm3]

        if props.checkpoint >= 2:
            c1_split_ratios, c2_split_ratios, c3_split_ratios, _ = checkpoint(self.simulate, split_ratios, tms, capacities, pte_info, batch_size, props, rate_cap=props.rate_cap, use_reentrant=False)
        else:
            c1_split_ratios, c2_split_ratios, c3_split_ratios, _ = self.simulate(split_ratios, tms, capacities, pte_info, batch_size, props, rate_cap=props.rate_cap)
        
        c1_split_ratios = c1_split_ratios*tm1.squeeze(-1)
        c2_split_ratios = c2_split_ratios*tm2.squeeze(-1)
        c3_split_ratios = c3_split_ratios*tm3.squeeze(-1)
        edges_util_tm2_3 = edges_util_tm2_3 + edges_util_tm1_3
        edges_util_tm3_3 = edges_util_tm3_3 + edges_util_tm2_3
        if props.sim_mf_mlu:
            return c1_split_ratios, c2_split_ratios, c3_split_ratios
        else:
            if props.mode == "train":
                edges_util_tm1 = torch.max(torch.max(edges_util_tm1_1, edges_util_tm1_2), edges_util_tm1_3)
                edges_util_tm2 = torch.max(edges_util_tm2_2, edges_util_tm2_3)
                all_traffic = c1_split_ratios + c2_split_ratios + c3_split_ratios
                return edges_util_tm1, edges_util_tm2, edges_util_tm3_3, edges_util_tm1_3, edges_util_tm2_3, all_traffic
            else:
                return edges_util_tm1_3, edges_util_tm2_3, edges_util_tm3_3
        
        
    def forward_pass_mlp(self, inputs, mlp: nn.ModuleList, num_hidden_layers, violation_data=None):
        """
        Apply a stack of Linear layers with LeakyReLU activations to produce raw path scores (gammas).

        - The first layer is applied to `inputs`, followed by LeakyReLU.
        - Each hidden layer (count = `num_hidden_layers`) is applied with LeakyReLU.
        - The final layer (index == num_hidden_layers + 1) is applied without activation.
          When `violation_data` is provided and the final layer's out_features is 2 or 3,
          the output is formed by independently applying the corresponding weight rows to
          either the raw activations or activations modulated by the violation tensors,
          then concatenating the results.

        Args:
            inputs (Tensor): Input tensor for the MLP.
            mlp (nn.ModuleList): Sequence of Linear layers defining the MLP.
            num_hidden_layers (int): Number of hidden layers (excludes input and output layers).
            violation_data (Optional[List[Tensor]]): Optional tensors used to modulate the final
                layer's computation to inject constraint/violation signals.

        Returns:
            Tensor: Output tensor shaped by the last Linear layer's out_features.
        """
        
        for index, layer in enumerate(mlp):
            if index == 0:
                gammas_1 = layer(inputs)
                gammas_1 = F.leaky_relu(gammas_1, 0.02)
            
            elif index == (num_hidden_layers + 1):
                if violation_data is not None:
                    if layer.weight.shape[0] == 2:
                        violation_data_1 = violation_data[0]
                        x = (violation_data_1 * gammas_1) @ layer.weight[0, :].unsqueeze(-1) + layer.bias[0]
                        y = (gammas_1) @ layer.weight[1, :].unsqueeze(-1) + layer.bias[1]
                        if len(x.shape) == 3 and len(y.shape) == 2:
                            x = x.squeeze(0)
                            y = y.squeeze(0)
                        gammas_1 = torch.cat((x, y), dim=-1)
                    elif layer.weight.shape[0] == 3:
                        violation_data_1, violation_data_2 = violation_data[0], violation_data[1]
                        x = (violation_data_1 * gammas_1) @ layer.weight[0, :].unsqueeze(-1) + layer.bias[0]
                        y = (violation_data_2 * gammas_1) @ layer.weight[1, :].unsqueeze(-1) + layer.bias[1]
                        z = (gammas_1) @ layer.weight[2, :].unsqueeze(-1) + layer.bias[2]
                        if len(x.shape) == 2 or len(y.shape) == 2 or len(z.shape) == 2:
                            x = x.squeeze(0)
                            y = y.squeeze(0)
                            z = z.squeeze(0)
                        gammas_1 = torch.cat((x, y, z), dim=-1)
                else:
                    gammas_1 = layer(gammas_1)
            else:
                gammas_1 = layer(gammas_1)
                gammas_1 = F.leaky_relu(gammas_1, 0.02)
                        
        return gammas_1
    
    def compute_mlu(self, edges_util, batch_size, total_number_of_paths, subtract_epsilon=True):
            """
            Compute per-batch Maximum Link Utilization (MLU) and broadcast over paths.

            - Reduces per-edge utilizations to a single MLU per batch element via max over edges.
            - Optionally subtracts a small epsilon for numerical stability.
            - Reshapes to [B, 1, 1] and expands to [B, P, 1] to align with path-wise tensors.

            Args:
                edges_util (Tensor): Per-edge utilizations [B, E].
                batch_size (int): Batch size B.
                total_number_of_paths (int): Number of paths P.
                subtract_epsilon (bool): Whether to subtract a small epsilon from the MLU.

            Returns:
                Tensor: Broadcast MLU of shape [B, P, 1].
            """

            mlu, mlu_indices = torch.max(edges_util, dim=-1)
            if subtract_epsilon:
                mlu = mlu -  epsilon
            mlu = mlu.view(batch_size, 1, 1).expand(-1, total_number_of_paths, -1)
            
            return mlu

    def compute_split_ratios(self, gammas, batch_size, num_paths_per_pair):
        gammas = gammas.reshape(batch_size, -1, num_paths_per_pair)
        split_ratios = torch.nn.functional.softmax(gammas, dim=-1)
        split_ratios = split_ratios.reshape(batch_size, -1)
        
        return split_ratios
        
    def compute_edge_utils(self, gammas, paths_to_edges, tm, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, violation=False, rate_cap=False):
        """
        Convert raw path scores to traffic allocations and compute per-edge utilizations.

        Behavior:
        - When rate_cap is False: reshape `gammas` to [B, P, K], apply log_softmax then exp to get
          numerically stable split ratios per path; multiply by tm to get data on tunnels.
        - When rate_cap is True: treat `gammas` as direct non-negative rates via ELU(+0.1).
        - Aggregate tunnel traffic to links via sparse matmul (paths_to_edges^T · tunnels).
        - Divide by capacities to obtain edge utilizations; optionally add epsilon.
        - If violation=True, also return raw tunnel traffic and split ratios for downstream use.

        Args:
            gammas (Tensor): Path scores; shape depends on rate_cap.
            paths_to_edges (torch.sparse_coo_tensor): Sparse map [P, E] from paths to edges.
            tm (Tensor): Traffic matrix per path group [B, P, 1].
            capacities (Tensor): Link capacities [B, E].
            props: Config namespace (uses props.dtype for dtype control).
            batch_size (int): B.
            num_paths_per_pair (int): K, paths per source-destination pair.
            add_epsilon (bool): Add small epsilon to edge utilizations.
            violation (bool): Return extra info for violation-aware stages.
            rate_cap (bool): Use rate-cap mode for `gammas`.

        Returns:
            Tuple[Tensor, Optional[Tensor], Tensor]:
            - edges_util: Edge utilizations [B, E].
            - data_on_tunnels: If violation=True, tunnel traffic [B, P]; else None.
            - split_ratios: Per-path split ratios [B, P, 1].
        """

        if not rate_cap:
            gammas = gammas.reshape(batch_size, -1, num_paths_per_pair)
            # split_ratios = torch.nn.functional.softmax(gammas, dim=-1)
            # Apply log_softmax and take exponential to get probabilities
            split_ratios = torch.exp(torch.nn.functional.log_softmax(gammas, dim=-1))
            split_ratios = split_ratios.reshape(batch_size, -1)
            data_on_tunnels = split_ratios*tm.squeeze(-1)
        else:
            gammas = gammas.reshape(batch_size, -1)
            data_on_tunnels = F.elu(gammas, 0.1) + 0.1
        
        # Actual matrix
        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        data_on_links = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), data_on_tunnels.to(dtype=torch.float32).t()).t()
        
        if props.dtype == torch.bfloat16:
            data_on_links = data_on_links.to(dtype=torch.bfloat16)
        
        if add_epsilon:
            edges_util = data_on_links/capacities + epsilon
        else:
            edges_util = data_on_links/capacities
        if violation:
            return edges_util, data_on_tunnels, split_ratios.unsqueeze(-1)
        else:
            return edges_util, None, split_ratios.unsqueeze(-1)
        
    
    def compute_bottleneck_util_per_path(self, edge_utils, paths_to_edges, row_indices, col_indices, values):
        """
        Compute, for each path, the bottleneck (max) link utilization and its edge index.

        Method:
        - Gather utilizations for the edges that belong to each path using the sparse layout
          defined by (row_indices, col_indices, values) of `paths_to_edges`.
        - Take a weighted max (utilization × value) over edges per path to find the bottleneck.
        - Map argmax positions back to edge IDs via `col_indices` and subtract a small epsilon
          from the max utilization for numerical stability.

        Args:
            edge_utils (Tensor): Per-edge utilization [B, E].
            paths_to_edges (torch.sparse_coo_tensor): Sparse map of paths to edges [P, E].
            row_indices (LongTensor): Row indices (path ids) of non-zero entries, shape [NNZ].
            col_indices (LongTensor): Column indices (edge ids) of non-zero entries, shape [NNZ].
            values (Tensor): Non-zero values of `paths_to_edges`, shape [NNZ].

        Returns:
            Tuple[Tensor, LongTensor]:
            - max_utilization_per_path: Bottleneck utilization per path [B, P].
            - bottleneck_edge_indices: Edge id of the bottleneck per path [B, P].
        """
        
        c1_max_utilization_per_path, c1_max_indices = torch_scatter.scatter_max((edge_utils[:, col_indices] * values),
                                                                            row_indices, dim=1, dim_size=paths_to_edges.shape[0])
        c1_max_indices = col_indices[c1_max_indices]
        c1_max_utilization_per_path = c1_max_utilization_per_path - epsilon
        
        return c1_max_utilization_per_path, c1_max_indices
        
    def simulate(self, split_ratios, tms, capacities, pte_info, batch_size, props, rate_cap=False):
        """
        Capacity-aware sequential admission simulation across three traffic classes.

        For each class (1 → 2 → 3):
        - Convert per-path split ratios and demands to link loads via sparse path→edge mapping.
        - Compute per-path bottleneck utilization and down-scale ratios that would exceed 1.0.
        - Recompute admitted loads and update residual capacities before the next class.

        Args:
            split_ratios (List[Tensor]): Class-wise per-path split ratios [B, P, 1] for classes 1..3.
            tms (List[Tensor]): Class-wise demands per path group [B, P, 1] for classes 1..3.
            capacities (Tensor): Link capacities [B, E].
            pte_info (Tuple): (paths_to_edges, row_indices, col_indices, values) describing sparse
                path→edge mapping.
            batch_size (int): B.
            rate_cap (bool): Present for API symmetry; not used in this routine.

        Returns:
            Tuple[Tensor, Tensor, Tensor, None]: Adjusted per-path split ratios for classes 1..3,
            each with shape [B, P]; last element is a placeholder (None).
        """
                
        sr_1, sr_2, sr_3 = split_ratios
        sr_1 = sr_1.reshape(batch_size, -1)
        sr_2 = sr_2.reshape(batch_size, -1)
        sr_3 = sr_3.reshape(batch_size, -1)
        
        tms_1, tms_2, tms_3 = tms
        split_ratios_1 = sr_1*tms_1.squeeze(-1)
        split_ratios_2 = sr_2*tms_2.squeeze(-1)
        split_ratios_3 = sr_3*tms_3.squeeze(-1)
        
        paths_to_edges, row_indices, col_indices, values = pte_info
        ## First class Simulation
        commodities_on_links_1 = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), split_ratios_1.to(dtype=torch.float32).t()).t()
        edge_utils_1 = commodities_on_links_1/capacities
        bottleneck_util_per_path_1, bottleneck_indices_1 = self.compute_bottleneck_util_per_path(edge_utils_1, paths_to_edges, row_indices, col_indices, values)
        sr_1 = sr_1/torch.where(bottleneck_util_per_path_1 < 1, 1, bottleneck_util_per_path_1)
        split_ratios_1 = sr_1*tms_1.squeeze(-1)
        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        commodities_on_links_1 = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), split_ratios_1.to(dtype=torch.float32).t()).t()
        edge_utils_1 = commodities_on_links_1/capacities
        capacities_2 = (capacities - commodities_on_links_1)
        capacities_2 = torch.where(capacities_2 <= 0, props.zero_cap_mask, capacities_2)
        # print(capacities_2.min().item(), c1_mlu.max().item())
        
        ## Second class Simulation
        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        commodities_on_links_2 = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), (split_ratios_2).to(dtype=torch.float32).t()).t()
        edge_utils_2 = commodities_on_links_2/capacities_2
        bottleneck_util_per_path_2, bottleneck_indices_2 = self.compute_bottleneck_util_per_path(edge_utils_2, paths_to_edges, row_indices, col_indices, values)
        sr_2 = sr_2/torch.where(bottleneck_util_per_path_2 < 1, 1, bottleneck_util_per_path_2)
        split_ratios_2 = sr_2*tms_2.squeeze(-1)
        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        commodities_on_links_2 = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), (split_ratios_2).to(dtype=torch.float32).t()).t()
        edge_utils_2 = commodities_on_links_2/capacities_2
        capacities_3 = (capacities_2 - commodities_on_links_2)
        capacities_3 = torch.where(capacities_3 <= 0, props.zero_cap_mask, capacities_3)
        
        ## Third class Simulation
        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        commodities_on_links_3 = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), (split_ratios_3).to(dtype=torch.float32).t()).t()
        edge_utils_3 = commodities_on_links_3/capacities_3
        bottleneck_util_per_path_3, bottleneck_indices_3 = self.compute_bottleneck_util_per_path(edge_utils_3, paths_to_edges, row_indices, col_indices, values)
        sr_3 = sr_3/torch.where(bottleneck_util_per_path_3 < 1, 1, bottleneck_util_per_path_3)

        return sr_1, sr_2, sr_3, None

    def compute_bottleneck_link_mlu_per_path(self, edge_utils, padded_edge_ids_per_path, path_edge_embeddings, batch_size, 
                                             total_number_of_paths, pte_info, capacities=None):
        """
        For each path, locate its bottleneck edge, fetch that edge's embedding, and return the
        bottleneck utilization with optional capacities.

        Steps:
        - Use `compute_bottleneck_util_per_path` logic to find the per-path max edge utilization
          and indices of the edges achieving that max.
        - Match those edge indices against `padded_edge_ids_per_path` to recover positions of the
          bottleneck edge within each path's padded edge sequence.
        - Index into `path_edge_embeddings` to extract the corresponding per-path bottleneck
          edge embeddings.
        - Optionally gather the capacities of the bottleneck edges (if `capacities` provided).

        Args:
            edge_utils (Tensor): Per-edge utilization [B, E].
            padded_edge_ids_per_path (LongTensor): Padded edge ids per path [B, P, Lmax].
            path_edge_embeddings (Tensor): Per-path per-edge embeddings [B, P, Lmax, D].
            batch_size (int): B.
            total_number_of_paths (int): P.
            pte_info (Tuple): (paths_to_edges, row_indices, col_indices, values) describing sparse
                path→edge mapping.
            capacities (Optional[Tensor]): Link capacities [B, E] to return bottleneck capacities.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]:
            - bottleneck_path_edge_embeddings: Embedding of bottleneck edge per path [B, P, D].
            - max_utilization_per_path: Bottleneck utilization per path [B, P, 1].
            - capacities_of_bottleneck_edges: If provided, capacities of those edges [B, P]; else None.
        """
        
        paths_to_edges, row_indices, col_indices, values = pte_info
        max_utilization_per_path, max_indices = torch_scatter.scatter_max((edge_utils[:, col_indices] * values),
                                                                        row_indices, dim=1, dim_size=paths_to_edges.shape[0])
        max_utilization_per_path = max_utilization_per_path - epsilon
        try:
            max_indices = col_indices[max_indices]
        except:
            print("max_indices.shape:", max_indices.shape)
            print("max_indices.device:", max_indices.device)
            print("max_indices.dtype:", max_indices.dtype)
            print("max_indices contains NaN:", torch.isnan(max_indices).any().item())
            print("max_indices contains Inf:", torch.isinf(max_indices).any().item())
            print(max_indices.max())
            print(col_indices.max())
            print("Out of bound indexing!!")
            exit(1)
        
        max_indices_expanded = max_indices.unsqueeze(2).expand(-1, -1,  padded_edge_ids_per_path.size(1))
        matches = (max_indices_expanded == padded_edge_ids_per_path)
        
        # positions = matches.nonzero()
        try:
            positions = torch.where(matches)
        except Exception as e:
            print(e)
            print(edge_utils.max())
            print("edge_utils contains NaN:", torch.isnan(edge_utils).any().item())
            print("edge_utils contains Inf:", torch.isinf(edge_utils).any().item())
            print(edge_utils.max())
            print(max_indices_expanded.shape, padded_edge_ids_per_path.shape)
            print(max_indices_expanded.max())
            print(padded_edge_ids_per_path.max())
            print(matches.max())
            print("Out of bound indexing!!")
            exit(1)
        positions = torch.stack(positions, dim=-1)
        positions = positions.view(batch_size, total_number_of_paths, -1)
                            
        dim0_range = positions[:, :, 0].view(batch_size, total_number_of_paths, -1)
        dim1_range = positions[:, :, 1].view(batch_size, total_number_of_paths, -1)
        positions = positions[:, :, -1].view(batch_size, total_number_of_paths, -1)
        
        bottleneck_path_edge_embeddings = (path_edge_embeddings[dim0_range, dim1_range, positions]).squeeze(-2)
        
        if capacities is not None:
            capacities_of_bottleneck_edges = torch.gather(capacities, dim=1, index=max_indices)
            return bottleneck_path_edge_embeddings, max_utilization_per_path.unsqueeze(-1), capacities_of_bottleneck_edges
        else:
            return bottleneck_path_edge_embeddings, max_utilization_per_path.unsqueeze(-1), None
        
    def compute_transformer_output(self, edge_embeddings_with_caps, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor, batch_size_tf, total_number_of_paths, props, cls_token):
            """
            Forward pass of the Set Transformer.
            """
            
            max_path_length = max(list(edge_ids_dict_tensor.keys())) + 1 # due to CLS token
            transformer_output = torch.empty((batch_size_tf, total_number_of_paths, max_path_length, self.input_dim),
                                                device=self.device, dtype=props.dtype)
            for i, key in enumerate(edge_ids_dict_tensor.keys()):
                temp_embds = edge_embeddings_with_caps[:, edge_ids_dict_tensor[key], :]
                # print(temp_embds.shape, temp_embds.min(), temp_embds.max())
                if torch.isnan(temp_embds).any():
                    print(f"NaN detected in temp_embds for key {key}")
                    print("temp_embds shape:", temp_embds.shape)
                    print("temp_embds min:", temp_embds.min().item())
                    print("temp_embds max:", temp_embds.max().item())
                    print("temp_embds dtype:", temp_embds.dtype)
                    print("temp_embds device:", temp_embds.device)
                    raise ValueError("NaN detected in temp_embds")
                if props.dynamic:
                    temp_cls_token = cls_token.expand(batch_size_tf, edge_ids_dict_tensor[key].shape[0], -1).unsqueeze(-2)
                else:
                    temp_cls_token = cls_token.expand(1, edge_ids_dict_tensor[key].shape[0], -1).unsqueeze(-2)
                temp_embds = torch.cat((temp_cls_token, edge_embeddings_with_caps[:, edge_ids_dict_tensor[key], :]), dim=-2)
                x1, x2, x3, x4 = temp_embds.shape
                temp_embds = temp_embds.reshape(x1*x2, x3, x4)
                temp_embds = self.transformer(temp_embds)
                temp_embds = temp_embds.reshape(x1, x2, x3, x4)
                temp_embds = F.pad(temp_embds, (0, 0, 0, max_path_length - temp_embds.shape[2]), value=0.0)
                transformer_output[:, original_pos_edge_ids_dict_tensor[key], :, :] = temp_embds
            
            return transformer_output