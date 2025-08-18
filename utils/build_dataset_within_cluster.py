from utils.snapshot_utils import Read_Snapshot
from utils.cluster_utils import Cluster_Info
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import torch

class DM_Dataset_within_Cluster(Dataset):
    def __init__(self, props, cluster, start, end):
        self.props = props
        self.cluster = cluster
        self.list_snapshots = []
        self.list_tms1 = []
        self.list_tms1_pred = []
        self.list_tms2 = []
        self.list_tms2_pred = []
        self.list_tms3 = []
        self.list_tms3_pred = []
        self.list_capacities = []
        self.list_node_features = []
        self.list_optimal_values_1 = []
        self.list_optimal_values_2 = []
        self.list_optimal_values_3 = []
        self.list_optimal_values_1_mf = []
        self.list_optimal_values_2_mf = []
        self.list_optimal_values_3_mf = []
        self.parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.results_path = f"{self.parent_dir_path}/results/{self.props.topo}/{props.num_paths_per_pair}sp/{self.cluster}"
        filenames = np.loadtxt(f"{self.results_path}/filenames.txt", dtype="U", delimiter=",").reshape(-1, 3)
        filenames = filenames[start:end]
        
        if self.props.failure_id == None:
            # if props.metric == "mf" or props.sim_mf_mlu:
            #     file = open(f"{self.results_path}/gt_optimal_values_mf_1.txt")
            #     opts1 = np.loadtxt(file, dtype=np.float32).ravel()
            #     file.close()
                
            #     file = open(f"{self.results_path}/gt_optimal_values_mf_2.txt")
            #     opts2 = np.loadtxt(file, dtype=np.float32).ravel()
            #     file.close()


            #     file = open(f"{self.results_path}/gt_optimal_values_mf_3.txt")
            #     opts3 = np.loadtxt(file, dtype=np.float32).ravel()
            #     file.close()


            # elif props.metric == "mlu":
            file = open(f"{self.results_path}/gt_optimal_values_mf.txt")
            opts1_mf = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()
            
            file = open(f"{self.results_path}/gt_optimal_values_mf_mf.txt")
            opts2_mf = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()

            
            file = open(f"{self.results_path}/gt_optimal_values_mf_mf_mf.txt")
            opts3_mf = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()
                        
            file = open(f"{self.results_path}/gt_optimal_values_mlu.txt")
            opts1 = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()

            file = open(f"{self.results_path}/gt_optimal_values_mlu_mlu.txt")
            opts2 = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()
            
            if props.rate_cap:
                file = open(f"{self.results_path}/gt_optimal_values_mf_3.txt")
                opts3 = np.loadtxt(file, dtype=np.float32).ravel()
                file.close()
            else:
                file = open(f"{self.results_path}/gt_optimal_values_mlu_mlu_mlu.txt")
                opts3 = np.loadtxt(file, dtype=np.float32).ravel()
                file.close()



            opts1 = opts1[start:end]
            opts2 = opts2[start:end]
            opts3 = opts3[start:end]
            opts1_mf = opts1_mf[start:end]
            opts2_mf = opts2_mf[start:end]
            opts3_mf = opts3_mf[start:end]
        else:
            file = open(f"{self.results_path}/optimal_values_failure_id_{self.props.failure_id}.txt")
            opts = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()
            if len(opts) == 0:
                exit(1)
                    
        for snapshot_filename, opt_value_1, opt_value_2, opt_value_3, opt_value_1_mf, opt_value_2_mf, opt_value_3_mf \
            in zip(filenames, opts1, opts2, opts3, opts1_mf, opts2_mf, opts3_mf):
            topology_filename, pairs_filename, tm_filename = snapshot_filename
            if props.topo == "geant_org":
                if tm_filename == "t2188.pkl":
                    continue
            snapshot = Read_Snapshot(self.props, topology_filename, pairs_filename, tm_filename)
            self.list_snapshots.append(snapshot)
            self.list_tms1.append(snapshot.tm1)
            self.list_tms1_pred.append(snapshot.tm1_pred)
            self.list_tms2.append(snapshot.tm2)
            self.list_tms2_pred.append(snapshot.tm2_pred)
            self.list_tms3.append(snapshot.tm3)
            self.list_tms3_pred.append(snapshot.tm3_pred)
            self.list_optimal_values_1.append(opt_value_1)
            self.list_optimal_values_2.append(opt_value_2)
            self.list_optimal_values_3.append(opt_value_3)
            self.list_optimal_values_1_mf.append(opt_value_1_mf)
            self.list_optimal_values_2_mf.append(opt_value_2_mf)
            self.list_optimal_values_3_mf.append(opt_value_3_mf)
            self.list_capacities.append(snapshot.capacities)
            self.list_node_features.append(snapshot._node_features)
            
        cluster_info = Cluster_Info(snapshot, props, self.cluster)
        self.edge_index = cluster_info.sp.get_edge_index().to(props.device)
        self.pij = cluster_info.compute_ksp_paths(props.num_paths_per_pair, cluster_info.sp.pairs)
        self.pte = cluster_info.get_paths_to_edges_matrix(self.pij)
        self.padded_edge_ids_per_path, self.edge_ids_dict_tensor, self.original_pos_edge_ids_dict_tensor = cluster_info.get_padded_edge_ids_per_path(self.pij, cluster_info.edges_map)
        self.num_pairs = cluster_info.num_pairs
        
        
    def __len__(self):
        return len(self.list_tms1)
    
    def __getitem__(self, idx):
        if self.props.pred:
            return self.list_node_features[idx], self.list_capacities[idx], self.list_tms1[idx], \
                self.list_tms1_pred[idx], self.list_tms2[idx], self.list_tms2_pred[idx], \
                self.list_tms3[idx], self.list_tms3_pred[idx], \
                self.list_optimal_values_1[idx], self.list_optimal_values_2[idx], self.list_optimal_values_3[idx],\
                self.list_optimal_values_1_mf[idx], self.list_optimal_values_2_mf[idx], self.list_optimal_values_3_mf[idx], self.list_snapshots[idx]
        else:
            return self.list_node_features[idx], self.list_capacities[idx], self.list_tms1[idx], \
                self.list_tms1[idx], self.list_tms2[idx], self.list_tms2[idx], \
                self.list_tms3[idx], self.list_tms3[idx], \
                self.list_optimal_values_1[idx], self.list_optimal_values_2[idx], self.list_optimal_values_3[idx],\
                self.list_optimal_values_1_mf[idx], self.list_optimal_values_2_mf[idx], self.list_optimal_values_3_mf[idx], self.list_snapshots[idx]
        
def custom_collate(batch):
    """
    Custom collate function for DM_Dataset_within_Cluster without using dictionaries.

    Args:
        batch (list of tuples): Each element of the batch is a tuple of data elements from the dataset.

    Returns:
        tuple: A tuple containing all batched elements in the same order as in __getitem__.
    """
    # Unzip the batch
    (
        node_features,
        capacities,
        tms1,
        tms1_pred,
        tms2,
        tms2_pred,
        tms3,
        tms3_pred,
        optimal_values_1,
        optimal_values_2,
        optimal_values_3,
        optimal_values_1_mf,
        optimal_values_2_mf,
        optimal_values_3_mf,
        snapshots,

    ) = zip(*batch)
    
    # Ensure all elements are tensors
    to_tensor = lambda x: torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    
    return (
        torch.stack([to_tensor(item) for item in node_features]),
        torch.stack([to_tensor(item) for item in capacities]),
        torch.stack([to_tensor(item) for item in tms1]),
        torch.stack([to_tensor(item) for item in tms1_pred]),
        torch.stack([to_tensor(item) for item in tms2]),
        torch.stack([to_tensor(item) for item in tms2_pred]),
        torch.stack([to_tensor(item) for item in tms3]),
        torch.stack([to_tensor(item) for item in tms3_pred]),
        torch.tensor(optimal_values_1),
        torch.tensor(optimal_values_2),
        torch.tensor(optimal_values_3),
        torch.tensor(optimal_values_1_mf),
        torch.tensor(optimal_values_2_mf),
        torch.tensor(optimal_values_3_mf),
        list(snapshots),  # Keep snapshots as a list since they're custom objects
    )