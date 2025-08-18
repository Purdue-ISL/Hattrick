import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from gurobipy import GRB
import gurobipy as gp
gp.disposeDefaultEnv()
import numpy as np
import copy
import tqdm
from scipy.sparse import csr_matrix
from utils.snapshot_utils import Read_Snapshot
from utils.cluster_utils import Cluster_Info
from utils.args_parser import parse_args
from frameworks.gurobi_utils import *
import gc
gc.collect()

try:
    os.mkdir(f"{parent_dir}/results")
except:
    pass

args = sys.argv[1:]
props = parse_args(args)
topo = props.topo
num_paths_per_pair = props.num_paths_per_pair
start_index = props.opt_start_idx
end_index = props.opt_end_idx
prio = props.priority
num_cluster  = props.cluster

results_path = f"{parent_dir}/results/{topo}/{num_paths_per_pair}sp"
try:
    os.makedirs(results_path)
except:
    pass


file_manifest = f"{parent_dir}/manifest/{props.topo}_manifest.txt"
manifest = np.loadtxt(file_manifest, dtype="U", delimiter=",")

try:
    os.mkdir(f"{results_path}/{num_cluster}")
except:
    pass


topology_filename, pairs_filename, tm_filename = manifest[start_index]
topology_filename = topology_filename.strip()
pairs_filename = pairs_filename.strip()
tm_filename = tm_filename.strip()

previous_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename)
current_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename)


cluster_info = Cluster_Info(current_sp, props, num_cluster)
cluster_info.pij = cluster_info.compute_ksp_paths(num_paths_per_pair, cluster_info.sp.pairs)
cluster_info.paths_to_edges = csr_matrix(cluster_info.get_paths_to_edges_matrix(cluster_info.pij).to_dense().numpy())


if props.gur_mode == "flexile":
    if props.pred == 0:
        if prio == 1:
            optimal_values, filenames, runtime_file = open_relevant_files(props, results_path, num_cluster)
        elif prio == 2:
            optimal_values, filenames, optimal_values_high_class, runtime_file = open_relevant_files(props, results_path, num_cluster)
        elif prio == 3:
            optimal_values, filenames, optimal_values_high_class, optimal_values_mid_class, runtime_file = open_relevant_files(props, results_path, num_cluster)
    elif props.pred == 1:
        if prio == 1:
            optimal_values, optimal_values_on_gt, runtime_file = open_relevant_files(props, results_path, num_cluster)
        elif prio == 2:
            optimal_values, optimal_values_high_class, optimal_values_on_gt, runtime_file = open_relevant_files(props, results_path, num_cluster)
        elif prio == 3:
            optimal_values, optimal_values_high_class, optimal_values_mid_class, optimal_values_on_gt, runtime_file = open_relevant_files(props, results_path, num_cluster)
elif props.gur_mode == "swan":
    assert props.pred == 1, "SWAN only supports prediction"
    swan, runtime_file = open_relevant_files(props, results_path, num_cluster)

num_snapshots_in_cluster = 0
for i, snapshot in tqdm.tqdm(enumerate(manifest[start_index:end_index]), total=len(manifest[start_index:end_index])):
    
    index = start_index + i
    objs = props.objs
    topology_filename, pairs_filename, tm_filename = snapshot
    topology_filename = topology_filename.strip()
    pairs_filename = pairs_filename.strip()
    tm_filename = tm_filename.strip()
    previous_sp = copy.deepcopy(current_sp)
    current_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename)
    
    if check_paths_recomputation(previous_sp, current_sp):
        if props.gur_mode == "flexile":
            optimal_values.close()
            if props.pred == 1:
                optimal_values_on_gt.close()
        elif props.gur_mode == "swan":
            swan.close()
        
        if props.pred == 0 and objs[-1] == "mf":
            write_mlus(num_cluster, num_snapshots_in_cluster, prio, results_path)

        num_cluster, num_snapshots_in_cluster = check_empty_cluster(props, results_path, num_snapshots_in_cluster, num_cluster)                

        optimal_values.close()
        runtime_file.close()

        if props.gur_mode == "flexile":
            if props.pred == 0:
                filenames.close()
                if prio == 1:
                    optimal_values, filenames, runtime_file = open_relevant_files(props, results_path, num_cluster)
                elif prio == 2:
                    optimal_values, filenames, optimal_values_high_class, runtime_file = open_relevant_files(props, results_path, num_cluster)
                elif prio == 3:
                    optimal_values, filenames, optimal_values_high_class, optimal_values_mid_class, runtime_file = open_relevant_files(props, results_path, num_cluster)
            elif props.pred == 1:
                optimal_values_on_gt.close()
                if prio == 1:
                    optimal_values, optimal_values_on_gt, runtime_file = open_relevant_files(props, results_path, num_cluster)
                elif prio == 2:
                    optimal_values, optimal_values_high_class, optimal_values_on_gt, runtime_file = open_relevant_files(props, results_path, num_cluster)
                elif prio == 3:
                    optimal_values, optimal_values_high_class, optimal_values_mid_class, optimal_values_on_gt, runtime_file = open_relevant_files(props, results_path, num_cluster)
        elif props.gur_mode == "swan":
            swan.close()
            swan, runtime_file = open_relevant_files(props, results_path, num_cluster)
        
        
        num_pairs = current_sp.num_demands
        cluster_info = Cluster_Info(current_sp, props, num_cluster)
        cluster_info.pij = cluster_info.compute_ksp_paths(num_paths_per_pair, cluster_info.sp.pairs)
        cluster_info.paths_to_edges = csr_matrix(cluster_info.get_paths_to_edges_matrix(cluster_info.pij).to_dense().numpy())
        
    #### Prepare the Gurobi model
    solver = GurobiModel(props, current_sp, props.gur_mode)
    solver.model.setParam("OutputFlag", 0)
    solver.model.setParam("NumericFocus", 3)
    solver.model.setParam("FeasibilityTol", 1e-6)  # Default is 1e-9
    # solver.model.setParam('Method', 2)       # Interior-point
    # solver.model.setParam('Presolve', 2)     # Aggressive presolve
    # solver.model.setParam('Crossover', 0)    # Disable crossover
            
    solver.add_variables(objs, index)
    solver.add_mlu_variables(objs)
    solver.mul_vars_by_tms()
    solver.add_demand_constraints(objs)
    if props.gur_mode == "flexile":
        if prio == 2:
            solver.add_optimality_constraints(optimal_values_high_class[num_snapshots_in_cluster], 0, objs)
        elif prio == 3:
            solver.add_optimality_constraints(optimal_values_high_class[num_snapshots_in_cluster], optimal_values_mid_class[num_snapshots_in_cluster], objs)
    solver.add_capacity_constraints(objs, cluster_info.paths_to_edges)
    solver.define_objective(objs)
    solver.model.optimize()
    if solver.model.status == GRB.status.OPTIMAL or solver.model.status == GRB.OPTIMAL:
        if props.pred == 0:
            optimal_values.write(str(solver.model.ObjVal)+"\n")
        else:
            if props.gur_mode == "flexile":
                if objs[-1].lower() == "mlu":
                    if prio != 3:
                        demand_over_tunnels = 0
                        for i in range(prio):
                            demand_over_tunnels += solver.model_variables[i].X.reshape(-1, 1)*solver.tms[i]
                        
                        commodities_on_links = cluster_info.paths_to_edges.T @ (demand_over_tunnels)
                        edge_utils = commodities_on_links/(solver.capacities.reshape(-1, 1) + 1e-6)
                        mask = np.isnan(edge_utils)
                        edge_utils[mask] = 0
                        # edge_utils = zero_the_nans(edge_utils)
                        optimal_values_on_gt.write(str(edge_utils.max())+"\n")
                        optimal_values.write(str(solver.model.ObjVal)+"\n")
                    else:
                        if objs[-1] == "mlu":
                            list_mlu = solver.compute_mlu_on_gt_third_stage(cluster_info.paths_to_edges)
                            optimal_values_on_gt.write(str(list_mlu[0])+"\n")
                            optimal_values_on_gt.write(str(list_mlu[1])+"\n")
                            optimal_values_on_gt.write(str(list_mlu[2])+"\n")
                            optimal_values.write(str(solver.model.ObjVal)+"\n")
                elif objs[-1].lower() == "mf":
                    optimal_values.write(str(solver.model.ObjVal)+"\n")
                if prio == 3:
                    solver.simulate(cluster_info.paths_to_edges, num_cluster, num_snapshots_in_cluster, objs)
                        
            elif props.gur_mode == "swan":
                if prio == 3:
                    x1 = solver.model_variables[0].reshape(-1, 1)
                    x2 = solver.model_variables[1].reshape(-1, 1)
                    x3 = solver.model_variables[2].X.reshape(-1, 1)
                    split_ratios = [x1, x2, x3]
                    solver.simulate(cluster_info.paths_to_edges, num_cluster, num_snapshots_in_cluster, objs, split_ratios)
                
        num_snapshots_in_cluster += 1
        runtime_file.write(str(solver.model.Runtime)+"\n")
        # print(solver.model.Runtime, )
        solver.dispose_model(index, objs[prio-1])
        del solver
        gc.collect()
        if props.pred == 0:
            filenames.write(str(topology_filename) + "," + str(pairs_filename) + "," + str(tm_filename) + "\n")
    else:
        print("Error! Likely a numerical issue or machine precision issue", tm_filename, index)
        print(solver.model.status)
        solver.model.computeIIS()
        solver.model.write("model.ilp")
        exit(1)

    
if props.pred == 0 and objs[-1] == "mf":
    write_mlus(num_cluster, num_snapshots_in_cluster, prio, results_path)

if props.gur_mode == "flexile":
    optimal_values.close()
    runtime_file.close()
    if props.pred == 1:
        optimal_values_on_gt.close()

elif props.gur_mode == "swan":
    swan.close()
if props.pred == 0:
    filenames.close()