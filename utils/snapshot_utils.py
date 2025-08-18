import numpy as np
import networkx as nx
import torch
import json
import pickle
import os

class Read_Snapshot:
    def __init__(self, props, topology_filename, pairs_filename, tm_filename):
        
        self.props = props
        self.topo = self.props.topo
        self.parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.failure_id = self.props.failure_id
                
        # read graph from json
        self.pairs = self.read_pairs_from_pkl(pairs_filename)
        self.tm1 = self.read_tms(tm_filename, 1)
        self.tm2 = self.read_tms(tm_filename, 2)
        self.tm3 = self.read_tms(tm_filename, 3)
        self.graph, self.capacities = self.read_graph_from_json(self.topo, topology_filename)
        if self.props.pred:
            self.tm1_pred = self.read_tms_pred(tm_filename, 1, props.pred_type)
            self.tm2_pred = self.read_tms_pred(tm_filename, 2, props.pred_type)
            self.tm3_pred = self.read_tms_pred(tm_filename, 3, props.pred_type)
        else:
            self.tm1_pred = np.array([0])
            self.tm2_pred = np.array([0])
            self.tm3_pred = np.array([0])
        self.num_demands = len(self.pairs)
        
        self.node_ids_map = {node: i for i, node in enumerate(self.graph.nodes())}
        if not props.framework.lower() == "gurobi":
            self._node_features = self.get_node_features()
            
            
    def read_pairs_from_pkl(self, pairs_filename):
        file = open(f"{self.parent_dir_path}/pairs/{self.topo}/{pairs_filename}", "rb")
        pairs = pickle.load(file)
        file.close()
        
        return pairs
        
    def update_capacities(self, new_capacities):
        for i, (u, v) in enumerate(self.graph.edges()):
            self.graph[u][v]['capacity'] = new_capacities[i][0]
            
    def write_graph_to_json(self, filename, graph, priority=3):
        filename = filename.split(".")[0] + ".json"
        path = f"{self.parent_dir_path}/topologies/{self.topo}_{priority}/{filename}"
        graph_data = nx.node_link_data(graph)
        with open(path, 'w') as f:
            json.dump(graph_data, f, indent=4)

    def read_graph_from_json_c2(self, topo: str, tm_filename):
        topology_filename = tm_filename.split(".")[0] + ".json"
        with open(f"{self.parent_dir_path}/topologies/{self.topo}_2/{topology_filename}") as f:
            data = json.load(f)
        
                
        graph = nx.readwrite.json_graph.node_link_graph(data)
        capacities = [float(data['capacity']) for u, v, data in graph.edges(data=True)]
        capacities = torch.tensor(capacities, dtype=torch.float32)
        
        return capacities
        
        
    def read_graph_from_json(self, topo: str, topology_filename):
        with open(f"{self.parent_dir_path}/topologies/{self.topo}/{topology_filename}") as f:
            data = json.load(f)
        
        
        graph = nx.readwrite.json_graph.node_link_graph(data)
        
        capacities = [float(data['capacity']) for u, v, data in graph.edges(data=True)]
        if self.props.framework == "gurobi":
            capacities = torch.tensor(capacities, dtype=torch.float32) + 1e-6
        else:
            capacities = torch.tensor(capacities, dtype=torch.float32)
        if self.failure_id != None:
            undirected_graph = graph.to_undirected()
            undirected_edges = list(undirected_graph.edges())
            directed_edges = list(graph.edges())
            failed_edge = undirected_edges[self.failure_id]
            x, y = failed_edge
            idx1 = directed_edges.index((x, y))
            idx2 = directed_edges.index((y, x))
            capacities[idx1] = 0
            capacities[idx2] = 0
                    
        
        return graph, capacities
    
    def get_edge_index(self) -> torch.Tensor:        
        source_nodes = []
        target_nodes = []
        for i, j in self.graph.edges():
                x = self.node_ids_map[i]
                y = self.node_ids_map[j]
                source_nodes.append(x)
                target_nodes.append(y)
                
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.int64)
        
        return edge_index
        
    def get_node_features(self):
        degrees = dict(self.graph.in_degree())
        degrees = torch.tensor(list(degrees.values()), dtype=torch.float32)
        degrees = degrees.reshape(-1, 1)
        edge_index = self.get_edge_index()
        
        mask = torch.where(self.capacities == 0)
        # Make sure the capacities are very low compared to the minimum non-zero capacity.
        # Make the ratio between this epsilon and the minimum non-zero capacity at least 1000.
        self.capacities[mask] += self.props.zero_cap_mask
        
        cap_sum_list = []
        for node in self.graph.nodes():
            node_id = self.node_ids_map[node]
            indices = (edge_index[0] == node_id).nonzero()
            cap_sum = torch.sum(self.capacities[indices])
            cap_sum_list.append(cap_sum)
        cap_sum_list = torch.tensor(cap_sum_list, dtype=torch.float32)
        cap_sum_list = cap_sum_list.reshape(-1, 1)
        node_features = torch.cat((degrees, cap_sum_list), dim=1)
        node_features = node_features.to(dtype=torch.float32)
        
        return node_features
        
    def read_tms(self, tm_filename, priority):
        file = open(f"{self.parent_dir_path}/traffic_matrices/{self.topo}_{priority}/{tm_filename}", 'rb')
        tm = pickle.load(file)
        tm = tm.reshape(-1, 1)
        if self.props.framework == "hattrick":
            tm = tm.astype(np.float32)
        else:
            tm = tm.astype(np.float64)
        tm = np.repeat(tm, repeats=self.props.num_paths_per_pair, axis=0)
        file.close()
        assert(tm.shape[0] == len(self.pairs)*self.props.num_paths_per_pair)
        
        return tm
    
    def read_tms_pred(self, tm_filename, priority, pred_type):
        file = open(f"{self.parent_dir_path}/traffic_matrices/{self.topo}_{priority}_{pred_type}/{tm_filename}", 'rb')
        tm = pickle.load(file)
        tm = tm.reshape(-1, 1)
        tm = tm + 1e-7
        if self.props.framework == "hattrick":
            tm = tm.astype(np.float32)
        else:
            tm = tm.astype(np.float64)
        tm = np.repeat(tm, repeats=self.props.num_paths_per_pair, axis=0)
        file.close()
        assert(tm.shape[0] == len(self.pairs)*self.props.num_paths_per_pair)
        
        return tm
    
    
def update_node_feature(sps, residual_capacity_batch):
    batch_size = residual_capacity_batch.shape[0]
    new_nf = []
    for i in range(batch_size):
        sp = sps[i]
        residual_capacity = residual_capacity_batch[i]
        sp.capacities = residual_capacity 
        nf = sp.get_node_features()
        new_nf.append(nf)
    new_nf = torch.stack(new_nf)
    return new_nf
