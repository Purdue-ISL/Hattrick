#!/bin/bash

# Modify the parameters as needed.
# Optimally, the --tol should be zero. However, this will cause infeasibility due to numerical errors. Thus, it should be a small value.

### To generate ground truth MaxFlow values
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 0 --pred_type esm --cluster 0 --priority 1 --objs mf --gur_mode flexile --tol 0.000001
wait
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 0 --pred_type esm --cluster 0 --priority 2 --objs mf mf --gur_mode flexile --tol 0.000001
wait
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 0 --pred_type esm --cluster 0 --priority 3 --objs mf mf mf --gur_mode flexile --tol 0.000001
wait

### To generate ground truth MLU values
# If you care about the MLU values, run MLU computation after MaxFlow computation. Otherwise, MaxFlow will overwrite the MLU values with ones.
# Generally speaking, you don't need any optimal value to train Hattrick. However, we recommend computing them to make sure the model converges well.
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 0 --pred_type esm --cluster 0 --priority 1 --objs mlu --gur_mode flexile --tol 0.000001
wait
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 0 --pred_type esm --cluster 0 --priority 2 --objs mlu mlu --gur_mode flexile --tol 0.000001
wait
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 0 --pred_type esm --cluster 0 --priority 3 --objs mlu mlu mlu --gur_mode flexile --tol 0.000001
wait

# # To evaluate Best_MC using predicted TMs
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 1 --pred_type esm --cluster 0 --priority 1 --objs mf --gur_mode flexile --tol 0.000001
wait
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 1 --pred_type esm --cluster 0 --priority 2 --objs mf mf --gur_mode flexile --tol 0.000001
wait
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 1 --pred_type esm --cluster 0 --priority 3 --objs mf mf mf --gur_mode flexile --tol 0.000001
wait

# To evaluate SWAN using predicted TMs
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 1 --pred_type esm --cluster 0 --priority 1 --objs mf --gur_mode swan --tol 0.000001
wait
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 1 --pred_type esm --cluster 0 --priority 2 --objs mf mf --gur_mode swan --tol 0.000001
wait
python3 frameworks/gurobi_refactored.py --num_paths_per_pair 4 --opt_start_idx 0 --opt_end_idx 16128 --topo abilene --framework gurobi --pred 1 --pred_type esm --cluster 0 --priority 3 --objs mf mf mf --gur_mode swan --tol 0.000001
wait

