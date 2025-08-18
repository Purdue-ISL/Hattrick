import numpy as np
from gurobipy import GRB, Model
import gurobipy as gp
import os
import sys
import argparse
import pickle
from collections import defaultdict
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.snapshot_utils import Read_Snapshot
from utils.cluster_utils import Cluster_Info
from scipy.sparse import csr_matrix

def check_paths_recomputation(previous_sp: Read_Snapshot, current_sp: Read_Snapshot):
    """
    Determine whether k-shortest paths must be recomputed by comparing snapshots.

    Returns True if node sets, node counts, edge counts, or demand pairs differ between
    the previous and current snapshots; otherwise returns False.
    """
    if (len(previous_sp.graph.nodes()) != len(current_sp.graph.nodes()))\
        or (set(previous_sp.graph.nodes()) != set(current_sp.graph.nodes()))\
        or (not np.array_equal(previous_sp.pairs, current_sp.pairs))\
        or(len(previous_sp.graph.edges()) != len(current_sp.graph.edges())):
            return True
    return False

def zero_the_nans(array):
    """
    Replace NaN values in a NumPy array with zeros and return the result.
    """
    return np.nan_to_num(array, nan=0.0)


class GurobiModel: 
    def __init__(self, props: argparse.ArgumentParser, current_sp: Read_Snapshot, mode="flexile"):
        """
        Initialize a multi-commodity flow Gurobi model for traffic engineering.

        Sets up model attributes, capacity arrays, traffic matrices (and predicted ones if
        enabled), bookkeeping dictionaries, and output directories based on props/mode.
        """
        self.model = Model("MCF")
        self.model.setParam(GRB.Param.OutputFlag, 0)
        self.props = props
        self.mode = mode
        self.current_sp = current_sp
        self.capacities = current_sp.capacities.numpy().astype(np.float64)
        # mask = np.where(self.capacities == 0)
        # self.capacities[mask] = 1e-6
        self.model_variables = dict()
        self.model_variables_by_tms = dict()
        self.mlu_variables = dict()
        self.num_tunnels = current_sp.num_demands * props.num_paths_per_pair
        self.tms = {0: current_sp.tm1, 1: current_sp.tm2, 2: current_sp.tm3}
        # self.x = sum([self.tms[i].sum() for i in range(self.props.priority)])/self.props.num_paths_per_pair
        # print(self.x)
        self.commodities_on_links = dict()
        self.mode = mode
        if self.props.pred:
            self.tms_pred = {0: current_sp.tm1_pred, 1: current_sp.tm2_pred, 2: current_sp.tm3_pred}
            self.pred_type = self.props.pred_type
        else:
            self.pred_type = "gt" # ground truth
        self.sr_path = f"{parent_dir}/../scratch/split_ratios/{self.props.topo}/{self.props.gur_mode}/{self.pred_type}"
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_path = f"{parent_dir}/results/{props.topo}/{props.num_paths_per_pair}sp"
        
        try:
            os.makedirs(self.sr_path)
            os.makedirs(f"{self.sr_path}/{self.props.priority - 1}")
        except:
            pass
        if self.mode == "flexile":
            for i in range(3):
                try:
                    os.mkdir(f"{self.sr_path}/{i}")
                except:
                    pass
        
    def add_variables(self, objs, index):
        """
        Add decision variables for path split ratios (or rates) per demand and path.

        In flexile mode, creates variables per priority stage with bounds [0, 1].
        In swan mode, creates variables only for the current priority; earlier stages
        are loaded from previously saved split ratios if needed.
        """
        
        if self.mode == "flexile":
            for i in range(len(objs)):
                self.model_variables[i] = self.model.addMVar((self.current_sp.num_demands,
                                        self.props.num_paths_per_pair),
                                        vtype=GRB.CONTINUOUS,
                                        lb=0.0, ub=1.0)
        elif self.mode == "swan":
            self.model_variables[self.props.priority - 1] = self.model.addMVar((self.current_sp.num_demands,
                                        self.props.num_paths_per_pair),
                                        vtype=GRB.CONTINUOUS,
                                        lb=0.0, ub=GRB.INFINITY)
            if self.props.priority == 2:
                self.model_variables[0] = self.read_split_ratios(index)[0]
            elif self.props.priority == 3:
                temp = self.read_split_ratios(index)
                self.model_variables[0] = temp[0]
                self.model_variables[1] = temp[1]
    
    def read_split_ratios(self, index):
        """
        Load previously saved split ratios for a given snapshot index (swan mode only).
        """
        if self.mode == "swan":
            file = open(f"{self.sr_path}/{index}.pkl", "rb")
            split_ratios = pickle.load(file)
            file.close()
            return split_ratios

    def add_demand_constraints(self, objs: list, *args):
        """
        Add per-demand sum constraints on split ratios based on the objective and stage.

        - For MLU, enforce sum == 1 per demand.
        - For MaxFlow, enforce sum <= 1 (or <= provided upper bounds for mid class).
        Behavior adapts across stages and between flexile and swan modes.
        """
        len_args = len(args)
        if self.mode == "flexile":
            for i, obj in enumerate(objs):
                if i == 0:
                    if obj.lower() == "mlu":
                        self.model.addConstrs((self.model_variables[i][k, :].sum() == 1)
                                            for k in range(self.current_sp.num_demands))
                    elif obj.lower() in ["max_flow", "mf"]:
                        self.model.addConstrs((self.model_variables[i][k, :].sum() <= 1)
                                            for k in range(self.current_sp.num_demands))
                        
                        
                if i == 1:
                    if obj.lower() in ["max_flow", "mf"]:
                        if len_args > 0:
                            split_ratio_constraints = args[0].reshape(-1, self.props.num_paths_per_pair)
                            split_ratio_constraints = split_ratio_constraints.sum(axis=1)
                            self.model.addConstrs((self.model_variables[i][k, :].sum() <= split_ratio_constraints[k])
                                                for k in range(self.current_sp.num_demands))
                        else:
                            self.model.addConstrs((self.model_variables[i][k, :].sum() <= 1)
                                                for k in range(self.current_sp.num_demands))
                                                                            
                    else: # Obj is MLU.
                        self.model.addConstrs((self.model_variables[i][k, :].sum() == 1)
                                            for k in range(self.current_sp.num_demands))
                    
                if i == 2:
                    if obj.lower() in ["max_flow", "mf"]:
                        self.model.addConstrs((self.model_variables[i][k, :].sum() <= 1)
                                        for k in range(self.current_sp.num_demands))
                    else:
                        self.model.addConstrs((self.model_variables[i][k, :].sum() == 1)
                                        for k in range(self.current_sp.num_demands))
        elif self.mode == "swan":
            if objs[-1].lower() in ["max_flow", "mf"]:
                self.model.addConstrs((self.model_variables[self.props.priority - 1][k, :].sum() <= 1)
                                    for k in range(self.current_sp.num_demands))
            else:
                self.model.addConstrs((self.model_variables[self.props.priority - 1][k, :].sum() == 1)
                                for k in range(self.current_sp.num_demands))
            
    def add_mlu_variables(self, objs: list):
        """
        Add scalar MLU variables for stages with MLU objective.

        Creates one continuous variable per stage (or only current priority in swan mode)
        when the corresponding objective is MLU.
        """
        if self.mode == "flexile":
            for i, obj in enumerate(objs):
                if obj.lower() == "mlu":
                    self.mlu_variables[i] = \
                        self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY)
        elif self.mode == "swan":
            if objs[-1].lower() == "mlu":
                self.mlu_variables[self.props.priority - 1] = \
                    self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY)
    
    def compute_mlu_on_gt_third_stage(self, paths_to_edges: csr_matrix):
        """
        Compute per-stage MLUs on ground-truth traffic for the third stage (priority==3).

        Accumulates admitted traffic from all earlier stages, projects to links using
        the sparse path→edge matrix, and returns a list of MLUs (one per stage).
        Requires predictive mode (props.pred==1) and priority==3.
        """
        assert self.props.pred == 1, "Not predictive TE"
        assert self.props.priority == 3, "Not final stage"
        demand_over_tunnels = defaultdict(int)
        list_mlu = []
        for i in range(self.props.priority):
            for j in range(i+1):
                demand_over_tunnels[i] += self.model_variables[j].X.reshape(-1, 1)*self.tms[j]
            commodities_on_links = paths_to_edges.T @ demand_over_tunnels[i]
            edge_utils = commodities_on_links / (self.capacities.reshape(-1, 1)+1e-8)
            mlu = edge_utils.max()
            list_mlu.append(mlu)
        return list_mlu
    
    def mul_vars_by_tms(self):
        """
        Multiply split-ratio variables by traffic matrices to form commodity demands.

        Handles both ground-truth and predictive modes, and optionally combines TMs
        when props.combine_tms==1. Updates `model_variables_by_tms` in place.
        """
        if self.props.pred == 0:
            if self.props.combine_tms == 1:
                assert self.props.priority == 1
                for i in range(self.props.priority):
                    self.model_variables_by_tms[i] = self.model_variables[i].reshape(-1, 1) * (self.tms[0] + self.tms[1] + self.tms[2])
            else:
                for i in range(self.props.priority):
                    self.model_variables_by_tms[i] = self.model_variables[i].reshape(-1, 1) * self.tms[i]
        elif self.props.pred == 1:
            for i in range(self.props.priority):
                self.model_variables_by_tms[i] = self.model_variables[i].reshape(-1, 1) * self.tms_pred[i]
        self.model.update()
                    

    def compute_commodities_on_links(self, paths_to_edges: csr_matrix, objs):
        """
        Project per-demand path allocations to per-link commodities via sparse matmul.

        Uses the CSR `paths_to_edges` matrix to compute link loads for each stage
        under either predictive or ground-truth TMs. Note: using chained multiplications
        of MVars may trigger a Gurobi warning and slow down MaxFlow solves.
        """
        if self.props.pred == 1:
            self.commodities_on_links[0] = paths_to_edges.T @ (self.model_variables[0].reshape(-1, 1) * self.tms_pred[0])
            if self.props.priority == 2:
                self.commodities_on_links[1] = paths_to_edges.T @ (self.model_variables[1].reshape(-1, 1) * self.tms_pred[1] + self.model_variables[0].reshape(-1, 1) * self.tms_pred[0])
            elif self.props.priority == 3:
                self.commodities_on_links[1] = paths_to_edges.T @ (self.model_variables[1].reshape(-1, 1) * self.tms_pred[1] + self.model_variables[0].reshape(-1, 1) * self.tms_pred[0])
                self.commodities_on_links[2] = paths_to_edges.T @ (self.model_variables[2].reshape(-1, 1) * self.tms_pred[2] + self.model_variables[1].reshape(-1, 1) * self.tms_pred[1] + self.model_variables[0].reshape(-1, 1) * self.tms_pred[0])
        else:
            self.commodities_on_links[0] = paths_to_edges.T @ (self.model_variables[0].reshape(-1, 1) * self.tms[0])
            if self.props.priority == 2:
                self.commodities_on_links[1] = paths_to_edges.T @ (self.model_variables[1].reshape(-1, 1) * self.tms[1] + self.model_variables[0].reshape(-1, 1) * self.tms[0])
            elif self.props.priority == 3:
                self.commodities_on_links[1] = paths_to_edges.T @ (self.model_variables[1].reshape(-1, 1) * self.tms[1] + self.model_variables[0].reshape(-1, 1) * self.tms[0])
                self.commodities_on_links[2] = paths_to_edges.T @ (self.model_variables[2].reshape(-1, 1) * self.tms[2] + self.model_variables[1].reshape(-1, 1) * self.tms[1] + self.model_variables[0].reshape(-1, 1) * self.tms[0])
            
    def add_optimality_constraints(self, high, mid, objs):
        """
        Add constraints that enforce optimality with tolerance for earlier stages.

        - For MLU, constrain stage MLU variable to be <= optimal value + tol.
        - For MaxFlow, constrain cumulative admitted traffic to be >= optimal*(1 - tol).
        Applies for high/mid stages depending on priority.
        """
        if objs[0] == "mlu":
            self.model.addConstr((self.mlu_variables[0] <= (high + self.props.tol)))
        elif objs[0] in ["max_flow", "mf"]:
            self.model.addConstr(((self.model_variables_by_tms[0].sum()) >= (high*(1-self.props.tol))))
        if len(objs) == 3:
            if objs[1] == "mlu":
                self.model.addConstr((self.mlu_variables[1] <= (mid + self.props.tol)))
            elif objs[1] in ["max_flow", "mf"]:
                self.model.addConstr(((self.model_variables_by_tms[0].sum() + self.model_variables_by_tms[1].sum()) >= (mid*(1-self.props.tol))))
    
    def add_capacity_constraints(self, objs: list, paths_to_edges: csr_matrix):
        """
        Add link capacity constraints based on computed commodities-on-links.

        - For MLU, bound link load by mlu_variable * capacity.
        - For MaxFlow, bound link load by capacity
        Applies per stage and adapts to flexile vs swan modes.
        """
        self.compute_commodities_on_links(paths_to_edges, objs)
        if self.mode == "flexile":
            for i, obj in enumerate(objs):
                if obj.lower() == "mlu":
                    self.model.addConstrs(self.commodities_on_links[i][j][0] <= 
                                        self.mlu_variables[i]*self.capacities[j]\
                                            for j in range(len(self.capacities)))
                    
                elif obj.lower() in ["max_flow", "mf"]:
                    if i != len(objs) - 1:
                        continue
                    else:
                        self.model.addConstrs(self.commodities_on_links[i][j][0] <=
                                            self.capacities[j] \
                                            for j in range(len(self.capacities)))
        
        elif self.mode == "swan":
            if objs[-1].lower() in ["max_flow", "mf"]:
                self.model.addConstrs(self.commodities_on_links[self.props.priority - 1][j][0] <=
                                    self.capacities[j] \
                                    for j in range(len(self.capacities)))
            elif objs[-1].lower() == "mlu":
                self.model.addConstrs(self.commodities_on_links[self.props.priority - 1][j][0] <=
                                    self.mlu_variables[self.props.priority - 1]*self.capacities[j] \
                                    for j in range(len(self.capacities)))

    
    def define_objective(self, objs):
        """
        Define the model objective for the current priority: minimize MLU or maximize flow.

        - MLU: minimize the corresponding mlu variable.
        - MaxFlow: maximize the sum of admitted traffic over all stages.
        """
        prio = self.props.priority
        if objs[prio-1].lower() == "mlu":
            obj = gp.LinExpr(self.mlu_variables[prio-1])
            self.model.setObjective(obj, GRB.MINIMIZE)
        elif objs[prio-1].lower() in ["max_flow", "mf"]:
            obj = 0
            for key in self.model_variables_by_tms.keys():
                obj += self.model_variables_by_tms[key].sum()
            self.model.setObjective(obj, GRB.MAXIMIZE)
            
    def picklize_model_variables(self, index, mode="mlu"):
        """
        Serialize and save split-ratio variables to disk for later stages or analysis.

        Saves per-priority arrays into `{sr_path}/{priority-1}/{index}[ _mf].pkl` in flexile mode,
        or `{sr_path}/{index}.pkl` in swan mode. Skips saving for certain MLU cases.
        """
        if self.mode == "flexile":
            array = []
            if self.props.priority == 1:
                for i in range(self.props.priority):
                    array.append(self.model_variables[i].X.reshape(-1, 1))
            elif self.props.priority == 2 and mode == "mf":
                array.append(self.model_variables[self.props.priority - 1].X.reshape(-1, 1))
            elif self.props.priority == 2 and mode == "mlu":
                return
            elif self.props.priority == 3:
                for i in range(self.props.priority):
                    array.append(self.model_variables[i].X.reshape(-1, 1))
            
            if self.props.priority != 3:
                if mode == "mf":
                    file = open(f"{self.sr_path}/{self.props.priority - 1}/{index}_mf.pkl", "wb")
                else:
                    file = open(f"{self.sr_path}/{self.props.priority - 1}/{index}.pkl", "wb")
            else:
                file = open(f"{self.sr_path}/{self.props.priority - 1}/{index}.pkl", "wb")
            
            pickle.dump(array, file)
            file.close()
        elif self.mode == "swan":
            file = open(f"{self.sr_path}/{index}.pkl", "wb")
            if self.props.priority == 1:
                split_ratios = [self.model_variables[self.props.priority - 1].X.reshape(-1, 1)]
            elif self.props.priority == 2:
                split_ratios = [self.model_variables[0].reshape(-1, 1), self.model_variables[1].X.reshape(-1, 1)]
            elif self.props.priority == 3:
                split_ratios = [self.model_variables[0].reshape(-1, 1), self.model_variables[1].reshape(-1, 1), self.model_variables[2].X.reshape(-1, 1)]
            pickle.dump(split_ratios, file)
            file.close()
    def dispose_model(self, index, mode="mlu", dispose=True):
        """
        Optionally persist variables and release the Gurobi model resources.

        In swan mode, persist split ratios before disposing the model.
        """
        # if self.props.priority == 3 and dispose:
        #     self.picklize_model_variables(index, mode)
        # self.picklize_model_variables(index, mode)
        if self.mode == "swan":
            self.picklize_model_variables(index, mode)
        self.model.dispose()
    def get_mf_split_ratios(self, index):
        """
        Load previously saved MaxFlow split ratios from the prior stage (priority==3).
        """
        assert self.props.priority == 3
        file = open(f"{self.sr_path}/{self.props.priority - 2}/{index}_mf.pkl", "rb")
        return pickle.load(file)
        
    def simulate(self, paths_to_edges, cluster, cluster_index, objs, split_ratios=None):
        """
        Offline simulation of sequential class admission to report MLUs and admitted traffic.

        Uses saved or provided split ratios to simulate classes 1→2→3 admission, accounting
        for residual capacities. Writes MLUs and cumulative admitted traffic to results files
        under `results_path` for the given cluster.
        """
        assert self.props.priority == 3, "Not the final stage"
        objs = "_".join(objs)
        if split_ratios is None:
            split_ratios_1 = self.model_variables[0].X.reshape(-1, 1)
            split_ratios_2 = self.model_variables[1].X.reshape(-1, 1)
            split_ratios_3 = self.model_variables[2].X.reshape(-1, 1)
        else:
            split_ratios_1 = split_ratios[0]
            split_ratios_2 = split_ratios[1]
            split_ratios_3 = split_ratios[2]
        if cluster_index == 0:
            sim_results_file = open(f"{self.results_path}/{cluster}/{self.mode}_sim_results_{self.pred_type}_{objs}.txt", "w")
        else:
            sim_results_file = open(f"{self.results_path}/{cluster}/{self.mode}_sim_results_{self.pred_type}_{objs}.txt", "a")
        
        vars_mlu_tm1 = split_ratios_1*self.current_sp.tm1
        vars_mlu_tm2 = split_ratios_2*self.current_sp.tm2
        vars_mlu_tm3 = split_ratios_3*self.current_sp.tm3
        
        ## First class Simulation
        commodities_on_links_1 = paths_to_edges.T @ (vars_mlu_tm1)
        edge_utils_1 = commodities_on_links_1/(self.capacities.reshape(-1, 1) + 1e-8)
        mlu_1 = edge_utils_1.max()
        sim_results_file.write(str(mlu_1)+"\n")
        if mlu_1 <= 1:
            sim_results_file.write(str(vars_mlu_tm1.sum())+"\n")
        else:
            bottleneck_util_per_path_1 = edge_utils_1.T * paths_to_edges.toarray()
            bottleneck_util_per_path_1 = bottleneck_util_per_path_1.max(axis=1)
            bottleneck_util_per_path_1 = np.where(bottleneck_util_per_path_1 < 1, 1, bottleneck_util_per_path_1).reshape(-1, 1)
            vars_mlu_tm1 = vars_mlu_tm1/bottleneck_util_per_path_1
            commodities_on_links_1 = paths_to_edges.T @ (vars_mlu_tm1)
            edge_utils_1 = commodities_on_links_1/(self.capacities.reshape(-1, 1) + 1e-8)
            sim_results_file.write(str(vars_mlu_tm1.sum())+"\n")
        
        ## Second class Simulation
        commodities_on_links_2 = paths_to_edges.T @ (split_ratios_1*self.current_sp.tm1 + vars_mlu_tm2) # Considering all traffic
        edge_utils_2 = commodities_on_links_2/(self.capacities.reshape(-1, 1) + 1e-8)
        mlu_2 = edge_utils_2.max()
        sim_results_file.write(str(mlu_2)+"\n")
        if mlu_2 <= 1:
            sim_results_file.write(str(vars_mlu_tm1.sum() + vars_mlu_tm2.sum())+"\n")
        else:
            commodities_on_links_1 = paths_to_edges.T @ (vars_mlu_tm1)
            residual_capacities = self.capacities - commodities_on_links_1.ravel()
            residual_capacities = np.where(residual_capacities <= self.props.zero_cap_mask, self.props.zero_cap_mask, residual_capacities)
            commodities_on_links_2_res = paths_to_edges.T @ (vars_mlu_tm2)
            edge_utils_2_res = commodities_on_links_2_res/(residual_capacities.reshape(-1, 1))
            # sim_results_file.write(str(edge_utils_2_res.max())+"\n")
            bottleneck_util_per_path_2 = edge_utils_2_res.T * paths_to_edges.toarray()
            bottleneck_util_per_path_2 = bottleneck_util_per_path_2.max(axis=1)
            bottleneck_util_per_path_2 = np.where(bottleneck_util_per_path_2 < 1, 1, bottleneck_util_per_path_2).reshape(-1, 1)
            vars_mlu_tm2 = vars_mlu_tm2/bottleneck_util_per_path_2
            sim_results_file.write(str(vars_mlu_tm1.sum() + vars_mlu_tm2.sum())+"\n")
        
        ## Third class Simulation
        
        commodities_on_links_3 = paths_to_edges.T @ (split_ratios_1*self.current_sp.tm1 + split_ratios_2*self.current_sp.tm2 + vars_mlu_tm3)
        # commodities_on_links_3 = filter_small_values(commodities_on_links_3)
        edge_utils_3 = commodities_on_links_3/(self.capacities.reshape(-1, 1)+1e-8)
        mlu_3 = edge_utils_3.max()
        sim_results_file.write(str(mlu_3)+"\n")
        if mlu_3 <= 1:
            sim_results_file.write(str(vars_mlu_tm1.sum() + vars_mlu_tm2.sum() + vars_mlu_tm3.sum())+"\n")
        else:
            commodities_on_links_1_2 = paths_to_edges.T @ (vars_mlu_tm1 + vars_mlu_tm2)
            # commodities_on_links_1_2 = filter_small_values(commodities_on_links_1_2)
            edge_utils_1_2 = commodities_on_links_1_2/(self.capacities.reshape(-1, 1) + 1e-8)
            residual_capacities_1_2 = self.capacities - commodities_on_links_1_2.ravel()
            residual_capacities_1_2 = np.where(residual_capacities_1_2 <= self.props.zero_cap_mask, self.props.zero_cap_mask, residual_capacities_1_2)
            commodities_on_links_3 = paths_to_edges.T @ (vars_mlu_tm3)
            edge_utils_3 = commodities_on_links_3/(residual_capacities_1_2.reshape(-1, 1))
            bottleneck_util_per_path_3 = edge_utils_3.T * paths_to_edges.toarray()
            bottleneck_util_per_path_3 = bottleneck_util_per_path_3.max(axis=1)
            bottleneck_util_per_path_3 = np.where(bottleneck_util_per_path_3 < 1, 1, bottleneck_util_per_path_3).reshape(-1, 1)
            vars_mlu_tm3 = vars_mlu_tm3/bottleneck_util_per_path_3
            sim_results_file.write(str(vars_mlu_tm1.sum() + vars_mlu_tm2.sum() + vars_mlu_tm3.sum())+"\n")
            # sim_results_file.write(str(edge_utils_3.max())+"\n")
            
        sim_results_file.close()

def open_relevant_files(props, results_path, num_cluster):
    """
    Open and return the appropriate result file handles for a cluster based on mode and priority.

    Returns a tuple of open file objects tailored to flexile or swan modes and whether
    predictive TMs are used, including optimal value files and runtime logs.
    """
    prio = props.priority
    objs = props.objs
    assert len(objs) == prio
    objs_str = "_".join(objs)
    
    if props.pred == 0:
        pred_type = "gt"
    else:
        pred_type = props.pred_type

    runtime_file = open(f"{results_path}/{num_cluster}/{props.gur_mode}_runtime_{prio}.txt", "w")
    if props.gur_mode == "flexile":
        optimal_path = f"{results_path}/{num_cluster}/{pred_type}_optimal_values_{objs_str}.txt"
        optimal_values = open(optimal_path, "w")
        if prio == 2:
            optimal_values_high_class = np.loadtxt(f"{results_path}/{num_cluster}/{pred_type}_optimal_values_{objs[0]}.txt").reshape(-1)
        if prio == 3:
            optimal_values_high_class = np.loadtxt(f"{results_path}/{num_cluster}/{pred_type}_optimal_values_{objs[0]}.txt").reshape(-1)
            optimal_values_mid_class = np.loadtxt(f"{results_path}/{num_cluster}/{pred_type}_optimal_values_{objs[0]}_{objs[1]}.txt").reshape(-1)
        
        
        if props.pred == 0:
            filenames_path = f"{results_path}/{num_cluster}/filenames.txt"
            filenames = open(filenames_path, "w")
            if prio == 1:
                return optimal_values, filenames, runtime_file
            elif prio == 2:
                return optimal_values, filenames, optimal_values_high_class, runtime_file
            elif prio == 3:
                return optimal_values, filenames, optimal_values_high_class, optimal_values_mid_class, runtime_file
        elif props.pred == 1:
            optimal_path_on_gt = f"{results_path}/{num_cluster}/{pred_type}_optimal_values_{objs_str}_on_gt.txt"
            optimal_values_on_gt = open(optimal_path_on_gt, "w")
            if prio == 1:
                return optimal_values, optimal_values_on_gt, runtime_file
            elif prio == 2:
                return optimal_values, optimal_values_high_class, optimal_values_on_gt, runtime_file
            elif prio == 3:
                return optimal_values, optimal_values_high_class, optimal_values_mid_class, optimal_values_on_gt, runtime_file

    elif props.gur_mode == "swan":
        swan_path = f"{results_path}/{num_cluster}/swan_{pred_type}_{props.priority}.txt"
        swan = open(swan_path, "w")
        
        return swan, runtime_file
    
def write_mlus(num_cluster, num_snapshots_in_cluster, prio, results_path):
    """
    Create a placeholder file of ones for ground-truth MLU optimal values for a cluster.

    Writes a vector of length `num_snapshots_in_cluster` with value 1.0 to the expected
    optimal-values path to bootstrap downstream workflows.
    """
    mlu = "_".join(["mlu"]*prio)
    optimal_path = f"{results_path}/{num_cluster}/gt_optimal_values_{mlu}.txt"
    # INSERT_YOUR_CODE
    ones_array = np.ones(num_snapshots_in_cluster)
    np.savetxt(optimal_path, ones_array, fmt='%f')


    

def check_empty_cluster(props, results_path, num_snapshots_in_cluster, num_cluster):
    """
    Finalize the current cluster directory and reset counters if the cluster is complete.

    - If the cluster has snapshots, increment cluster index and reset count.
    - If empty, remove generated path files and ensure the next cluster directory exists.
    Returns the updated `(num_cluster, num_snapshots_in_cluster)`.
    """
    if num_snapshots_in_cluster > 0:
        num_cluster += 1
        num_snapshots_in_cluster = 0
    elif num_snapshots_in_cluster == 0:
        path1 = f"{parent_dir}/topologies/paths"
        path2 = f"{parent_dir}/topologies/paths_dict"
        os.remove(f"{path1}/{props.topo}_{props.num_paths_per_pair}_paths_cluster_{num_cluster}.pkl")
        os.remove(f"{path2}/{props.topo}_{props.num_paths_per_pair}_paths_dict_cluster_{num_cluster}.pkl")
    try:
        os.mkdir(f"{results_path}/{num_cluster}")
    except:
        pass
    
    return num_cluster, num_snapshots_in_cluster
