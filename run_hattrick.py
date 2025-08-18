import sys
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'
import numpy as np
import torch
import random
torch.autograd.set_detect_anomaly(True)
seed = 490
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)

cwd = os.getcwd()
sys.path.append(cwd + "/utils")
from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster, custom_collate
from utils.AdamOptimizer import *
from utils.training_utils import *
import time

device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader
from tqdm import tqdm
from frameworks.hattrick_system import Hattrick
import copy

from datetime import datetime

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("Current Date and Time: ", formatted_date_time)
props = parse_args(sys.argv[1:])
props.device = device
props_geant = copy.deepcopy(props)

if props.dtype.lower() == "float32":
    props.dtype = torch.float32
elif props.dtype.lower() == "float16":
    props.dtype = torch.bfloat16
else:
    print("Only float32 and float16 are allowed")
    exit(1)


# Retrieve batch size and number of epochs
batch_size = props.batch_size
n_epochs = props.epochs

if props.pred:
    pred_type = props.pred_type
else:
    pred_type = "gt"

string = f"{props.num_paths_per_pair}sp_{props.batch_size}_{props.lr}_{props.num_gnn_layers}_{props.num_transformer_layers}_{props.num_mlp1_hidden_layers}_{props.num_mlp2_hidden_layers}_{props.rau1}_{props.rau2}_{props.rau3}_{pred_type}_viol_{props.violation}_AL_{props.additive_loss}_meta_{props.meta_learning}"

hp_path = f"hp_search/{props.topo}/{props.model}/{pred_type}/"

try:
    os.makedirs(hp_path)
except:
    pass


print(string)
if props.mode.lower() == "train":
    
    models_path = f"models/{props.topo}"
    optimizer_path = f"optimizers/{props.topo}"
    try:
        os.makedirs(models_path)
    except:
        pass
    try:
        os.makedirs(optimizer_path)
    except:
        pass
    
    models_path = f"models/{props.topo}"
    if props.initial_training == 1:
        model = Hattrick(props)
    else:
        try:
            model = torch.load(f'{models_path}/{string}.pkl', map_location=device)
            print("I'M LOADING THE MODEL")
        except Exception as e:
            print(e)
            model = Hattrick(props)
            print("Model not Found! Creating a new model")
    
    model = model.to(device=device, dtype=props.dtype)
    model.train()
    # create the training and validation DataLoaders
    ds_list, dl_list = create_dataloaders(props, batch_size, training = True, shuffle = True)
    val_ds_list, val_dl_list = create_dataloaders(props, 1, training = False, shuffle = False)
    
    
    if props.initial_training == 1:
        val1_loss = 1e9
        val2_loss = 1e9
        val3_loss = 1e9
    else:
        try:
            x = [i for i in os.listdir(f"{hp_path}") if string in i]
            x = x[0].split("_")
            val1_loss = float(x[0])
            val2_loss = float(x[1])
            val3_loss = float(x[2])
            print(val1_loss, val2_loss, val3_loss)
        except:
                val1_loss = 1e9
                val2_loss = 1e9
                val3_loss = 1e9
        
    if props.initial_training == 1:
        if props.additive_loss:
            c1_optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)
        else:
            c1_optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    else:
        try:
            if props.additive_loss:
                c1_optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)
            else:
                c1_optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
            c1_optimizer.load_state_dict(torch.load(f'{optimizer_path}/{string}.pkl'))
        except:
            if props.additive_loss:
                c1_optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)
            else:
                c1_optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    
    optimizers = [c1_optimizer]
    
    if props.meta_learning and props.initial_training == 1:
        print("Meta-Training Hattrick")
        rau = 7
        props_geant.topo = "geant"
        props_geant.num_paths_per_pair = 8
        props_geant.train_clusters = [0]
        props_geant.train_start_indices = [0]
        props_geant.train_end_indices = [6464]
        props_geant.epochs = 10
        props_geant.batch_size = 128
        props_geant.dynamic = 0
        props_geant.pred = 1
        props_geant.pred_type = "esm"
        props_geant.rau1 = rau
        props_geant.rau2 = rau
        props_geant.rau3 = rau
        props_geant.device = device
        props_geant.dtype = props.dtype
        geant_optimizer = ADAMOptimizer(model.parameters(), lr=0.001)
        
        geant_ds_list, geant_dl_list = create_dataloaders(props_geant, props_geant.batch_size, training = True, shuffle = True)
        for i in range(props_geant.epochs):
            train(i, props_geant.epochs, model, props_geant, geant_ds_list, geant_dl_list, [geant_optimizer])
        print("Meta-Training Done")
        
    for epoch in range(n_epochs):
        model.train()
        ds_list, dl_list = create_dataloaders(props, batch_size, training = True, shuffle = True)
        train(epoch, n_epochs, model, props, ds_list, dl_list, optimizers)
        # Iterate over validation clusters
        model.eval()
        val_avg_loss1, val_avg_loss2, val_avg_loss3, all_traffic = validate(model, props, val_ds_list, val_dl_list)
        if (val_avg_loss1 < (val1_loss + 0.003) and val_avg_loss2 < (val2_loss + 0.007) and val_avg_loss3 < (val3_loss)) \
            or (val_avg_loss1 < (val1_loss) and val_avg_loss2 < (val2_loss + 0.05) and val_avg_loss3 < (val3_loss + 0.1)) \
                or (val_avg_loss1 < (val1_loss + 0.005) and val_avg_loss2 < (val2_loss) and val_avg_loss3 < (val3_loss + 0.01)):
            val1_loss = val_avg_loss1
            val2_loss = val_avg_loss2
            val3_loss = val_avg_loss3
            try:
                x = [k for k in os.listdir(f"{hp_path}") if string in k]
                os.remove(f"{hp_path}/{x[0]}")
            except:
                print("What is wrong with you bro!?")
                pass
            torch.save(model, f'{hp_path}/{round(val_avg_loss1, 7)}_{round(val_avg_loss2, 7)}_{round(val_avg_loss3, 7)}_{round(all_traffic, 7)}_{string}_.pkl')
        torch.save(model, f"hattrick_{props.topo}_{props.num_paths_per_pair}sp.pkl")
        torch.save(model, f'{models_path}/{string}.pkl')
        torch.save(c1_optimizer.state_dict(), f'{optimizer_path}/{string}.pkl')
        
elif props.mode.lower() == "test": #test
    cluster = props.test_cluster
    start = props.test_start_idx
    end = props.test_end_idx
    test_dataset = DM_Dataset_within_Cluster(props, cluster, start, end)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    test_dataset.pte = (test_dataset.pte).to(props.device, dtype=props.dtype)
    test_dataset.padded_edge_ids_per_path = test_dataset.padded_edge_ids_per_path.to(device)
    move_to_device(test_dataset.edge_ids_dict_tensor, props.device)
    move_to_device(test_dataset.original_pos_edge_ids_dict_tensor, props.device)

    model = torch.load(f"abilene_7_7_7.pkl", map_location=device)
    
    # model = torch.load(f"hattrick_{props.topo}_{props.num_paths_per_pair}sp.pkl", map_location=device)
    model = model.to(dtype=props.dtype)
    model.eval()
    runtimes = open(f"results/{props.topo}/{props.num_paths_per_pair}sp/{cluster}/hattrick_runtime.txt", "w")
    with torch.no_grad():
        with tqdm(test_dl) as tests:
            test_losses1 = []
            test_losses2 = []
            test_losses3 = []
            file = open(f"results/{props.topo}/{props.num_paths_per_pair}sp/{cluster}/{props.model}_values_{pred_type}_sim_mlu_{props.sim_mf_mlu}.txt", "w")
            for inputs in tests:
                node_features, capacities, tms1, tms1_pred, tms2, tms2_pred, tms3, tms3_pred, opt1, opt2, opt3, opt1_mf, opt2_mf, opt3_mf, sp = inputs
                node_features = node_features.to(device=device, dtype=props.dtype)
                capacities = capacities.to(device=device, dtype=props.dtype)
                tms1 = tms1.to(device=device, dtype=props.dtype)
                tms2 = tms2.to(device=device, dtype=props.dtype)
                tms3 = tms3.to(device=device, dtype=props.dtype)
                opt1 = opt1.to(device=device, dtype=props.dtype)
                opt2 = opt2.to(device=device, dtype=props.dtype)
                opt3 = opt3.to(device=device, dtype=props.dtype)
                opt1_mf = opt1_mf.to(device=device, dtype=props.dtype)
                opt2_mf = opt2_mf.to(device=device, dtype=props.dtype)
                opt3_mf = opt3_mf.to(device=device, dtype=props.dtype)

                start = time.time()
                # If prediction is on, feed the predicted matrix
                if props.pred:
                    tms1_pred = tms1_pred.to(device=props.device, dtype=props.dtype)
                    tms2_pred = tms2_pred.to(device=props.device, dtype=props.dtype)
                    tms_3pred = tms3_pred.to(device=props.device, dtype=props.dtype)
                    output = model(props, node_features, test_dataset.edge_index, capacities,
                            test_dataset.padded_edge_ids_per_path,
                            tms1, tms1_pred, tms2, tms2_pred, tms3, tms_3pred, test_dataset.pte,
                            test_dataset.edge_ids_dict_tensor, test_dataset.original_pos_edge_ids_dict_tensor)

                # If prediction is off, feed the actual matrix as predicted matrix
                else:
                    output = model(props, node_features, test_dataset.edge_index, capacities,
                            test_dataset.padded_edge_ids_per_path,
                            tms1, tms1, tms2, tms2, tms3, tms3, test_dataset.pte,
                            test_dataset.edge_ids_dict_tensor, test_dataset.original_pos_edge_ids_dict_tensor)
                end = time.time()
                runtimes.write(str(end - start) + "\n")
                if props.metric == "mlu" and not props.sim_mf_mlu:
                    c1_edge_utils, c2_edge_utils, c3_edge_utils = output
                    loss1, loss1_val = loss_mlu(c1_edge_utils, opt1)
                    loss2, loss2_val = loss_mlu(c2_edge_utils, opt2)
                    loss3, loss3_val = loss_mlu(c3_edge_utils, opt3)
                else:
                    sr_1, sr_2, sr_3 = output
                    loss1, loss1_val = loss_mf(sr_1, opt1_mf)
                    loss2, loss2_val = loss_mf(sr_2, opt2_mf - opt1_mf)
                    loss3, loss3_val = loss_mf(sr_3, opt3_mf - opt2_mf)
                    
                test_losses1.append(loss1_val)
                test_losses2.append(loss2_val)
                test_losses3.append(loss3_val)
                
                if props.topo in ["uscarrier", "kdl"]:
                    file.write(str((sr_1.sum()/(tms1.sum()/props.num_paths_per_pair)).item()) + "\n")
                    file.write(str((sr_2.sum()/(tms2.sum()/props.num_paths_per_pair)).item()) + "\n")
                    file.write(str((sr_3.sum()/(tms3.sum()/props.num_paths_per_pair)).item()) + "\n")
                else:
                    file.write(str(loss1_val) + "\n")
                    file.write(str(loss2_val) + "\n")
                    file.write(str(loss3_val) + "\n")

            avg_loss1 = sum(test_losses1) / len(test_losses1)
            avg_loss2 = sum(test_losses2) / len(test_losses2)
            avg_loss3 = sum(test_losses3) / len(test_losses3)
            print(f"Test Error: \nAvg loss: {avg_loss1:>8f} | {avg_loss2:>8f} | {avg_loss3:>8f} \n")
            file.close()
            
            with open(f"results/{props.topo}/{props.num_paths_per_pair}sp/{cluster}/{props.model}_stats_{pred_type}_sim_mlu_{props.sim_mf_mlu}.txt", 'w') as f:
                import statistics
                dists = [float(v) for v in test_losses1]
                dists.sort()
                f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                f.write('MINIMUM: ' + str(dists[0]) + '\n')
                f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
                f.write('100TH: ' + str(dists[int(len(dists)-1)]) + '\n')
                f.write("___________________________________\n")
                dists = [float(v) for v in test_losses2]
                dists.sort()
                f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                f.write('MINIMUM: ' + str(dists[0]) + '\n')
                f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
                f.write('100TH: ' + str(dists[int(len(dists)-1)]) + '\n')
                
                f.write("___________________________________\n")
                dists = [float(v) for v in test_losses3]
                dists.sort()
                f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                f.write('MINIMUM: ' + str(dists[0]) + '\n')
                f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
                f.write('100TH: ' + str(dists[int(len(dists)-1)]) + '\n')

            runtimes.close()