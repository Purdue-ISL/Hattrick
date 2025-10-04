# Hattrick: Solving Multi-Class TE using Neural Models
[Hattrick](https://doi.org/10.1145/3718958.3750470) is a transferable neural network for WAN **Multi-Class Traffic Engineering** that is designed to co-optimize multiple classes of traffic considering prediction error. It was published at ACM SIGCOMM 2025.

If you use this code, please cite:
```
@inproceedings{10.1145/3718958.3750470,
author = {AlQiam, Abd AlRhman and Li, Zhuocong and Ahuja, Satyajeet Singh and Wang, Zhaodong and Zhang, Ying and Rao, Sanjay G. and Ribeiro, Bruno and Tawarmalani, Mohit},
title = {Hattrick: Solving Multi-Class TE using Neural Models},
year = {2025},
isbn = {9798400715242},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3718958.3750470},
doi = {10.1145/3718958.3750470},
booktitle = {Proceedings of the ACM SIGCOMM 2025 Conference},
pages = {264–278},
numpages = {15},
keywords = {traffic engineering, wide-area networks, network optimization, machine learning},
location = {S\~{a}o Francisco Convent, Coimbra, Portugal},
series = {SIGCOMM '25}}
```

Please contact `aalqiam@purdue.edu` for any questions.
### Environment Used
Hattrick was tested using the following setup:
- RockyLinux 9.5 machine
- Python 3.12.5
- `torch==2.5.1+cu124`
- `torch-scatter==2.1.2+pt25cu124`
- Check the rest in requirements.

### Required Libraries
1. Install the required Python packages as listed in the requirements.txt. Use:
   `pip install -r requirements.txt`
2. Please follow this [link](https://pytorch.org/get-started/locally/) to install a version of PyTorch that fits your environment (CPU/GPU).
3. Identify and copy the link of a suitable [URL](https://data.pyg.org/whl/) depending on PyTorch and CUDA/CPU versions installed in the previous step. Then, run:
   - `pip install --no-index torch-scatter -f [URL]`
4. Follow [Gurobi Website](https://www.gurobi.com/) to install and setup Gurobi Optimizer.
   - Please check the `Setup_Gurobi.md` file to see the steps needed to setup Gurobi.

## Steps to run Hattrick
1. Prepare your data in the format Hattrick accepts. Please check [Data Format](#data-format). 
2. Generate predicted traffic matrices. Please check [Predicted Matrices Format and Generation](#predicted-matrices-format-and-generation).
3. Compute optimal MaxFlow and MLU values. Please check [How to use Gurobi](#how-to-compute-optimal-values-using-gurobi).
4. Train and test a `Hattrick` model. Please check [Training Hattrick](#how-to-train-and-test-hattrick).

### How to compute optimal values using Gurobi
- To compute optimal values and cluster your dataset:
   - Please refer to the `run_gur.sh` file, and carefully read the comments there.
   - Please refer to our [HARP](https://dl.acm.org/doi/10.1145/3651890.3672237) paper and check the definition of a "cluster" in this context.

### How to train and test `Hattrick`:
 - To train `Hattrick`, run (for example):
   - ``python3 run_hattrick.py --topo TopoName --mode train --epochs 60 --batch_size 64 --num_paths_per_pair 4 --num_transformer_layers 3 --num_gnn_layers 3 --num_mlp1_hidden_layers 2 --num_mlp2_hidden_layers 2 --rau1 3 --rau2 3 --rau3 3 --train_clusters 0 --train_start_indices 0 --train_end_indices 12096 --val_clusters 0 --val_start_indices 12096 --val_end_indices 14122 --pred 1 --dynamic 0 --lr 0.0005 --pred_type esm --initial_training 1 --violation 1``
   - ``python3 run_hattrick.py --topo TopoName --mode train --epochs 60 --batch_size 64 --num_paths_per_pair 15 --num_transformer_layers 3 --num_gnn_layers 4 --num_mlp1_hidden_layers 4 --num_mlp2_hidden_layers 4 --rau1 20 --rau2 15 --rau3 15 --train_clusters 0 1 2 --train_start_indices 0 0 0 --train_end_indices 3000 3000 3000 --val_clusters 3 4 --val_start_indices 0 0 --val_end_indices 1000 1000 --pred 1 --dynamic 1 --lr 0.0005 --pred_type esm --initial_training 1 --violation 1``

- To test `Hattrick`, run (for example):
  - ``python3 run_hattrick.py --topo abilene --test_start_idx 14112 --test_end_idx 16128 --test_cluster 0 --mode test --num_paths_per_pair 4 --rau1 3 --rau2 3 --rau3 3 --pred 1 --dynamic 0 --pred_type esm --sim_mf_mlu 1``
  - Testing can only be done one cluster at a time. Employ a for loop to test various clusters of a dataset.

- For further explanation on command line arguments, see [Command Line Arguments Explanation](#command-line-arguments-explanation)

### Working with Public Datasets:
- **NOTE**: We did not use Hattrick with Abilene.
- Download `AbileneTM-all.tar` from this [link](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/) and decompress it (twice) inside ``prepare_abilene`` folder.
   - `cd prepare_abilene`
   - `wget https://www.cs.utexas.edu/~yzhang/research/AbileneTM/AbileneTM-all.tar`
   - `tar -xvf AbileneTM-all.tar`
   - `gunzip *.gz`
   - Then, run ``python3 prepare_abilene_hattrick.py``
  - This example should serve as a reference on how to prepare any dataset.
  - You do not need to use all traffic matrices (~48K).
- Execute `wget --content-disposition "https://app.box.com/shared/static/8r9w2txtnjxj1g09yhru5oxf676ygulm?dl=1" -P traffic_matrices/` to download the **KDL** matrices we used.
  - The **KDL** TMs are also found on this [link](https://app.box.com/s/8r9w2txtnjxj1g09yhru5oxf676ygulm).
- Execute `wget --content-disposition "https://app.box.com/shared/static/c1m97gblyglwj7qrftgqrbttwz3h6mns?dl=1" -P traffic_matrices/` to download the **UsCarrier** matrices we used.
  - The **UsCarrier** matrices are also found on this [link](https://app.box.com/s/c1m97gblyglwj7qrftgqrbttwz3h6mns).
- A preprocessed copy of the **GEANT** matrices in the format needed by Hattrick is available on this [link](https://app.box.com/s/shzgaxnt36org6dmu9q228kzk28numue).
- The GEANT matrices contain a **single class** of traffic. You need to split them. Look at the Abilene preparation script for one way of doing it.

### Data Format:
- In the `manifest` folder, The user should provide a `txt` file that holds the topology name and describes at every time step the **topology_file.json**,**set_of_pairs_file.pkl**,**traffic_matrix.pkl** file that will be read at that time step. For every timestep, a corresponding file of these three should exist in the `topologies`, `pairs`, and `traffic_matrices` folders inside a directory with the topology name.
- **Traffic matrices**: Numpy array of shape (num_pairs, 1)
  - high-class traffic goes into `TopoName_1`
  - medium-class traffic goes into `TopoName_2`
  - low-class traffic goes into `TopoName_3`
- **Pairs**: Numpy array of shape (num_pairs, 2)
- Note: the kth demand in the traffic matrix must correspond to the kth pair in the set of pairs file. This relation must be preserved for all snapshots. **We suggest sorting the hash map (pairs/keys and values/demands) before separating**.
- **Paths**: By default, Hattrick computes K shortest paths and automatically puts them in the correct folders and format. If you wish to use your paths:
  - create a Python dictionary where the keys are the pairs and the values are a list of $K$ lists, where the inner lists are a sequence of edges.
  - For example: {(s, t): [[(s, a), (a, t)], [(s, a), (a, b), (b, t)]]}.
  - Put it inside `topologies/paths_dict` and name it: `TopoName_K_paths_dict_cluster_num_cluster.pkl`. For example: `abilene_8_paths_dict_cluster_0.pkl`.
  - Make sure all pairs have the same number of paths (replicate if needed).

### Predicted Matrices Format and Generation
- By default, Hattrick trains over predicted matrices.
- Running Hattrick with `--pred 0` trains it over ground truth matrices rather than predicted matrices.
- An ESM (Exponential Smoothing) predictor is provided in `traffic_matrices` directory
  - To use it, run: `python3 esm_predictor.py TopoName`
- Provide the predicted traffic matrices using a predictor of your choice, then put them inside the `traffic_matrices` directory inside a folder named `TopoName_x_PredictorName`.
  - For example, for the GEANT dataset, ground truth matrices will be under the `geant_1`, `geant_2`, and `geant_3` directories, whereas predicted matrices will be under the `geant_1_esm`, `geant_2_esm`, and `geant_3_esm` directories, assuming an exponential smoothing predictor is used.
  - Make sure that at every time step, the predicted matrix corresponds to the ground truth matrix at that time step.
     - For example: t100.pkl in the `geant_1` and the `geant_1_esm` folders correspond to each other, and so on.

### Command Line Arguments Explanation
| Flag | Meaning | Notes |
| --- | --- | --- |
| framework | Determines the framework that solves the problem [hattrick, gurobi]. | Default is Hattrick. |
| num_heads | Number of transformer attention heads [int]. | By default, it is equal to the number of GNN layers. |
| rau1 | Determines Hattrick's Number of Recurrent Adjustment Unit (RAUs) of the first optimization stage. | |
| rau2 | Determines Hattrick's Number of RAUs of the second optimization stage. | |
| rau3 | Determines Hattrick's Number of RAUs of the third optimization stage | |
| dynamic | If your topology varies across snapshots, set it to `1`. If it is static, set it to `0`. | <br>- In our paper, the PrivateWAN network is dynamic. <br>- GEANT, UsCarrier, and KDL networks are static. <br>- **This CLA is useful to save GPU memory when training for a (static) topology that does not change across snapshots**. |
| dtype | Determines the `dtype` of Hattrick and its data [float32, float16] corresponding to [torch.float32, torch.bfloat16]. | The default is float32. |
| violation | Turns ViolationMLPs on/off. | Default value is 1 (on). |
| gur_mode | Gurobi mode for multi-class TE [flexile, swan] | Flexile is Best_MC |
| zero_cap_mask | The value used to encode zero capacities (full-link failure). | Pick it to be 1000-10000x times less than the minimum non-zero capacity. |
| meta_learning [0, 1] | Turn on/off meta learning. This is used to train Hattrick on geant dataset for a couple of epochs before training it on the desired dataset. | This is generally useful, but specifically when the network/dataset is highly dynamic with lots of failures, where Hattrick might struggle to/not converge. Meta-learning significantly improves the convergence. |
| priority [1, 2, 3] | Used to indicate stage of optimization or class of traffic. | |
| checkpoint [0, 1, 2] | Used to control gradient checkpointing (GC) for Hattrick. | - **0: no GC**. <br>- **1: light GC**. <br>- **2: aggressive GC**. <br>- Gradient checkpointing basically trades-off time for memory on the level of the mini-batch. That is, the mini-batch takes more time to compute. However, you can serve more examples in the mini-batch, or train larger models.<br>- We find out that, for the KDL topology, using GC=2 saves around 70% of GPU memory but takes 45% more time per mini-batch, assuming fixed amount of examples per mini-batch. However, with GC, you can feed more examples in the mini-batch, which might end up to be faster on the level of the epoch. |
| initial_training [0, 1] | Turns on/off model checkpointing for Hattrick. Hattrick checkpoints the model and optimizer after every epoch. Set it to 1 to continue training from the last epoch. | This is useful if you use a shared cluster of GPUs where you have a maximum number of hours per job. |

### General Notes:
- The learning rate is a very important hyperparameter. For the KDL topology, we had to use a high learning rate (0.05)
- The repo has infrastructure for manual hyperparameter search.
- A small number of models may generate NaN gradients and terminate training. This has only occurred with a very dynamic topology (PrivateWAN). We are investigating and will update the repo once it is fixed.