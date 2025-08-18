import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .build_dataset_within_cluster import DM_Dataset_within_Cluster, custom_collate
# from .proj_utils import *
from .robust_proj_utils import project_gradients_one_optimizer_robust, assign_gradients_and_step
import time

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(epoch, n_epochs, model, props, ds_list, dl_list, optimizers):
    num_params = count_parameters(model)
    if epoch == 0:
        print("Hattrick's Number of parameters: ", num_params)
    
    model = model.to(device=props.device, dtype=props.dtype)
    final_grads = torch.zeros(num_params, dtype=props.dtype, device=props.device)
    c1_optimizer = optimizers[0]
    # Iterate over training clusters
    for j in range(len(ds_list)):
        train_dataset = ds_list[j]
        train_dl = dl_list[j]
        train_dataset.pte = (train_dataset.pte).to(device=props.device, dtype=props.dtype)
        train_dataset.pte = train_dataset.pte.coalesce()
        train_dataset.padded_edge_ids_per_path = train_dataset.padded_edge_ids_per_path.to(props.device)
        move_to_device(train_dataset.edge_ids_dict_tensor, props.device)
        move_to_device(train_dataset.original_pos_edge_ids_dict_tensor, props.device)
        model.train()
        with tqdm(train_dl) as tepoch:
            torch.cuda.empty_cache()
            loss1_sum = loss1_count = loss2_sum = loss2_count = loss3_sum = loss3_count = loss4_sum = loss4_count = loss5_sum = loss5_count = loss6_sum = loss6_count = 0
            for i, inputs in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}/{n_epochs}")
                # Retrieve inputs to HARP
                node_features, capacities, tms1, tms1_pred, tms2, tms2_pred, tms3, tms3_pred, opt1, opt2, opt3, opt1_mf, opt2_mf, opt3_mf, snapshots  = inputs
                
                # If the topology does not change across examples/snapshots (static topology), just take the first example
                if not props.dynamic:
                    node_features = node_features[:1]
                    capacities = capacities[:1]
                    
                node_features = node_features.to(device=props.device, dtype=props.dtype)
                capacities = capacities.to(device=props.device, dtype=props.dtype)
                tms1 = tms1.to(device=props.device, dtype=props.dtype)
                tms2 = tms2.to(device=props.device, dtype=props.dtype)
                tms3 = tms3.to(device=props.device, dtype=props.dtype)
                opt1 = opt1.to(device=props.device, dtype=props.dtype)
                opt2 = opt2.to(device=props.device, dtype=props.dtype)
                opt3 = opt3.to(device=props.device, dtype=props.dtype)
                opt1_mf = opt1_mf.to(device=props.device, dtype=props.dtype)
                opt2_mf = opt2_mf.to(device=props.device, dtype=props.dtype)
                opt3_mf = opt3_mf.to(device=props.device, dtype=props.dtype)
                # If prediction is on, feed the predicted matrix
                if props.pred:
                    tms1_pred = tms1_pred.to(device=props.device, dtype=props.dtype)
                    tms2_pred = tms2_pred.to(device=props.device, dtype=props.dtype)
                    tms3_pred = tms3_pred.to(device=props.device, dtype=props.dtype)
                    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                    start_total = time.time()
                    start = time.time()
                    output = model(props, node_features, train_dataset.edge_index, capacities,
                        train_dataset.padded_edge_ids_per_path,
                        tms1, tms1_pred, tms2, tms2_pred, tms3, tms3_pred, train_dataset.pte,
                        train_dataset.edge_ids_dict_tensor, train_dataset.original_pos_edge_ids_dict_tensor)
                    end = time.time()
                    # print("Time taken for forward pass: ", end - start)
                # If prediction is off, feed the actual matrix as predicted matrix
                else:
                    output = model(props, node_features, train_dataset.edge_index, capacities,
                            train_dataset.padded_edge_ids_per_path,
                            tms1, tms1, tms2, tms2, tms3, tms3, train_dataset.pte,
                            train_dataset.edge_ids_dict_tensor, train_dataset.original_pos_edge_ids_dict_tensor)
                
                edges_util_tm1_1, edges_util_tm2_2, edges_util_tm3_3, edges_util_tm1_3, edges_util_tm2_3, all_traffic = output
                # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                start = time.time()
                loss1, loss1_val = loss_mlu(edges_util_tm1_1, opt1)
                loss2, loss2_val = loss_mlu(edges_util_tm2_2, opt2)
                loss3, loss3_val = loss_mf(all_traffic, opt3_mf)
                loss4, loss4_val = loss_mlu(edges_util_tm3_3, opt3)
                loss5, loss5_val = loss_mlu(edges_util_tm1_3, opt1)
                loss6, loss6_val = loss_mlu(edges_util_tm2_3, opt2)

                end = time.time()
                # print("Time taken for loss calculation: ", end - start)
                if props.additive_loss:
                    loss = 10*loss1 + 0.1*loss2 + 0.01*loss3 + 0.001*loss4
                    loss.backward()
                    c1_optimizer.step()
                    model.zero_grad(set_to_none=True)
                else:
                    start = time.time()
                    final_grads_mb, shapes2, flattened_grads1, flattened_grads_c2, flattened_grads_c3, c1_optimizer = \
                                    project_gradients_one_optimizer_robust(model, loss1, loss2, loss3, loss4, c1_optimizer)
                    end = time.time()
                    end_total = time.time()
                    # print("Time taken for backward pass: ", end - start)
                    # print("Time taken for total: ", end_total - start_total)
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    # print(f"\nPeak GPU memory usage: {peak_memory:.4f} GB")
                    
                    if (i+1)%props.round == 0 or (i+1) == len(train_dl):
                        final_grads += final_grads_mb
                        assign_gradients_and_step(model, final_grads, c1_optimizer, shapes2)
                        torch.cuda.empty_cache()
                        final_grads[:] = 0
                    else:
                        final_grads += final_grads_mb
                        model.zero_grad(set_to_none=True)

                torch.cuda.empty_cache()
                loss1_sum += loss1_val
                loss1_count += 1
                loss2_sum += loss2_val
                loss2_count += 1
                loss3_sum += loss3_val
                loss3_count += 1
                loss1_avg = loss1_sum / loss1_count
                loss2_avg = loss2_sum / loss2_count
                loss3_avg = loss3_sum / loss3_count
                loss4_sum += loss4_val
                loss4_count += 1
                loss4_avg = loss4_sum / loss4_count
                loss5_sum += loss5_val
                loss5_count += 1
                loss5_avg = loss5_sum / loss5_count
                loss6_sum += loss6_val
                loss6_count += 1
                loss6_avg = loss6_sum / loss6_count
                tepoch.set_postfix(loss1=loss1_avg, loss2=loss2_avg, loss3=loss3_avg, loss4=loss4_avg, loss5=loss5_avg, loss6=loss6_avg)
        train_dataset.pte = (train_dataset.pte).to(device="cpu")
        train_dataset.padded_edge_ids_per_path = train_dataset.padded_edge_ids_per_path.to(device="cpu")
        move_to_device(train_dataset.edge_ids_dict_tensor, "cpu")
        move_to_device(train_dataset.original_pos_edge_ids_dict_tensor, "cpu")
        
# Define the loss
def loss_mlu(y_pred_batch, y_true_batch):
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        opt = y_true_batch[[i]]
        max_cong = torch.max(y_pred)
        
        loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong/max_cong.item()
        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt.item()
        losses.append(loss)
        loss_vals.append(loss_val)
    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val


def create_dataloaders(props, batch_size, training = True, shuffle = True):
    ds_list = []
    dl_list = []
    if training:
        clusters = props.train_clusters
        start_indices = props.train_start_indices
        end_indices = props.train_end_indices
    else:
        clusters = props.val_clusters
        start_indices = props.val_start_indices
        end_indices = props.val_end_indices
    for clstr, start, end in zip(clusters, start_indices, end_indices):
        dataset = DM_Dataset_within_Cluster(props, clstr, start, end)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle, collate_fn=custom_collate)
        ds_list.append(dataset)
        dl_list.append(dl)
    
    return ds_list, dl_list

def move_to_device(dictionary, device="cpu"):
    for key in dictionary.keys():
        dictionary[key] = dictionary[key].to(device)
    return dictionary


def loss_mf(y_pred_batch, y_true_batch):
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        opt = y_true_batch[[i]]
        total_flow = torch.sum(y_pred)
        loss = -total_flow if total_flow.item() == 0.0 else -total_flow/total_flow.item()
        loss_val = 1.0 if opt == 0.0 else total_flow.item() / opt.item()
        losses.append(loss)
        loss_vals.append(loss_val)
    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    return ret, ret_val
    

def validate(model, props, val_ds_list, val_dl_list):
        val_avg_loss1 = []
        val_avg_loss2 = []
        val_avg_loss3 = []
        all_traffic_list = []
        for k in range(len(val_ds_list)):
            val_dataset = val_ds_list[k]
            val_dl = val_dl_list[k]
            val_dataset.pte = (val_dataset.pte).to(device=props.device, dtype=props.dtype)
            val_dataset.padded_edge_ids_per_path = val_dataset.padded_edge_ids_per_path.to(device=props.device)
            move_to_device(val_dataset.edge_ids_dict_tensor, props.device)
            move_to_device(val_dataset.original_pos_edge_ids_dict_tensor, props.device)
            
            val_norm_mlu = _validate(model, props, val_dataset, val_dl)
            val_norm_mlu1, val_norm_mlu2, val_norm_mlu3, all_traffic = val_norm_mlu
            val_avg_loss1.append(sum(val_norm_mlu1)/len(val_norm_mlu1))
            val_avg_loss2.append(sum(val_norm_mlu2)/len(val_norm_mlu2))
            val_avg_loss3.append(sum(val_norm_mlu3)/len(val_norm_mlu3))
            all_traffic_list.append(sum(all_traffic)/len(all_traffic))
            print(f"Validation Avg loss: {round(val_avg_loss1[-1], 5), round(val_avg_loss2[-1], 5), round(val_avg_loss3[-1], 5), round(all_traffic_list[-1], 5)}")
            
            val_dataset.pte = (val_dataset.pte).to(device="cpu")
            val_dataset.padded_edge_ids_per_path = val_dataset.padded_edge_ids_per_path.to(device="cpu")
            move_to_device(val_dataset.edge_ids_dict_tensor, "cpu")
            move_to_device(val_dataset.original_pos_edge_ids_dict_tensor, "cpu")
        
        val_avg_loss1 = sum(val_avg_loss1) / len(val_avg_loss1)
        val_avg_loss2 = sum(val_avg_loss2) / len(val_avg_loss2)
        val_avg_loss3 = sum(val_avg_loss3) / len(val_avg_loss3)
        
        all_traffic = sum(all_traffic) / len(all_traffic)
                        
            
        return val_avg_loss1, val_avg_loss2, val_avg_loss3, all_traffic
        
# Function for validation set
def _validate(model, props, val_ds, val_dl):
    val_norm_mlu1 = []
    val_norm_mlu2 = []
    val_norm_mlu3 = []
    all_traffic_list = []
    with torch.no_grad():
        with tqdm(val_dl) as vals:
            for i, inputs in enumerate(vals):
                node_features, capacities, tms1, tms1_pred, tms2, tms2_pred, tms3, tms3_pred, opt1, opt2, opt3, opt1_mf, opt2_mf, opt3_mf, snapshots  = inputs
                
                node_features = node_features.to(device=props.device, dtype=props.dtype)
                capacities = capacities.to(device=props.device, dtype=props.dtype)
                tms1 = tms1.to(device=props.device, dtype=props.dtype)
                tms2 = tms2.to(device=props.device, dtype=props.dtype)
                tms3 = tms3.to(device=props.device, dtype=props.dtype)
                opt1 = opt1.to(device=props.device, dtype=props.dtype)
                opt2 = opt2.to(device=props.device, dtype=props.dtype)
                opt3 = opt3.to(device=props.device, dtype=props.dtype)
                opt1_mf = opt1_mf.to(device=props.device, dtype=props.dtype)
                opt2_mf = opt2_mf.to(device=props.device, dtype=props.dtype)
                opt3_mf = opt3_mf.to(device=props.device, dtype=props.dtype)
                # If prediction is on, feed the predicted matrix
                if props.pred:
                    tms1_pred = tms1_pred.to(device=props.device, dtype=props.dtype)
                    tms2_pred = tms2_pred.to(device=props.device, dtype=props.dtype)
                    tms3_pred = tms3_pred.to(device=props.device, dtype=props.dtype)
                    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(props, node_features, val_ds.edge_index, capacities,
                            val_ds.padded_edge_ids_per_path,
                            tms1, tms1_pred, tms2, tms2_pred, tms3, tms3_pred, val_ds.pte,
                            val_ds.edge_ids_dict_tensor, val_ds.original_pos_edge_ids_dict_tensor)
                # If prediction is off, feed the actual matrix as predicted matrix
                else:
                    output = model(props, node_features, val_ds.edge_index, capacities,
                            val_ds.padded_edge_ids_per_path,
                            tms1, tms1, tms2, tms2, tms3, tms3, val_ds.pte,
                            val_ds.edge_ids_dict_tensor, val_ds.original_pos_edge_ids_dict_tensor)
                    
                edges_util_tm1_1, edges_util_tm2_2, edges_util_tm3_3, edges_util_tm1_3, edges_util_tm2_3, all_traffic = output
                loss1, loss1_val = loss_mlu(edges_util_tm1_1, opt1)
                loss2, loss2_val = loss_mlu(edges_util_tm2_2, opt2)
                loss3, loss3_val = loss_mlu(edges_util_tm3_3, opt3)
                loss4, loss4_val = loss_mf(all_traffic, opt3_mf)
                loss5, loss5_val = loss_mlu(edges_util_tm1_3, opt1)
                loss6, loss6_val = loss_mlu(edges_util_tm2_3, opt2)
                val_norm_mlu1.append(loss5_val)
                val_norm_mlu2.append(loss6_val)
                val_norm_mlu3.append(loss3_val)
                all_traffic_list.append(loss4_val)

    return val_norm_mlu1, val_norm_mlu2, val_norm_mlu3, all_traffic_list
    
