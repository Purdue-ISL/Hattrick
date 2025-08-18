import torch
import numpy as np
eps = 1e-9  # Small epsilon value for numerical stability

def capture_non_zero_grads(flattened_grads_list):
    valid_grads = []
    for i, grad in enumerate(flattened_grads_list):
        if not torch.all(grad == 0):
            valid_grads.append(grad)
    return valid_grads

def project_two_grads(flattened_grads1, flattened_grads2):
    g1_g2_proj = torch.dot(flattened_grads1, flattened_grads2) / (torch.dot(flattened_grads1, flattened_grads1) + eps)
    if torch.isnan(g1_g2_proj):
        print("g1_g2_proj", g1_g2_proj)
        print("Dot Products",torch.dot(flattened_grads1, flattened_grads1), torch.dot(flattened_grads2, flattened_grads1))
        print("norms", torch.norm(flattened_grads1), torch.norm(flattened_grads2))
        print("Number of Nans", torch.isnan(flattened_grads1).sum(), torch.isnan(flattened_grads2).sum())
        
    flattened_grads2 = flattened_grads2 - g1_g2_proj * flattened_grads1
    return flattened_grads2

def project_three_grads(flattened_grads1, flattened_grads2, flattened_grads3):
    
    try:
        A = torch.stack([flattened_grads1, flattened_grads2], dim=1)
        A_T = A.T
        A_T_A = A_T @ A
        # Add small diagonal term for numerical stability
        A_T_A = A_T_A + eps * torch.eye(A_T_A.shape[0], device=A_T_A.device)
        A_T_b = A_T @ flattened_grads3
                
        x_hat = torch.linalg.solve(A_T_A, A_T_b)
        p = A @ x_hat
        flag = False
        for i, x in enumerate([flattened_grads1, flattened_grads2, flattened_grads3, A, A_T, A_T_A, A_T_b, x_hat, p]):
            mask = torch.isnan(x)
            if torch.sum(mask) > 0:
                flag = True
                print("3rd Proj: Hmmmm", i, torch.sum(mask))
        if flag:
            exit(1)
        
        # Update flattened_grads3
        flattened_grads3 = flattened_grads3 - p
    except:
        # Add eps to denominator for numerical stability
        g1_g3_proj = torch.dot(flattened_grads1, flattened_grads3) / (torch.dot(flattened_grads1, flattened_grads1) + eps)
        if torch.isnan(g1_g3_proj):
            print("g1_g3_proj", g1_g3_proj)
            print("Dot Products",torch.dot(flattened_grads1, flattened_grads1), torch.dot(flattened_grads3, flattened_grads1))
            print("norms", torch.norm(flattened_grads1), torch.norm(flattened_grads3))
            print("Number of Nans", torch.isnan(flattened_grads1).sum(), torch.isnan(flattened_grads3).sum())
            
        flattened_grads3 = flattened_grads3 - g1_g3_proj * flattened_grads1
        
    return flattened_grads3


def project_four_grads(flattened_grads1, flattened_grads2, flattened_grads3, flattened_grads4):
    try:
        A = torch.stack([flattened_grads1, flattened_grads2, flattened_grads3], dim=1)
        indep_cols = find_linearly_independent_vectors(A)
        A = A[:, indep_cols]
        A_T = A.T
        A_T_A = A_T @ A
        # Add small diagonal term for numerical stability
        A_T_A = A_T_A + eps * torch.eye(A_T_A.shape[0], device=A_T_A.device)
        A_T_b = A_T @ flattened_grads4
        
        x_hat = torch.linalg.solve(A_T_A, A_T_b)
        p = A @ x_hat
        flag = False
        for i, x in enumerate([A, A_T, A_T_A, A_T_b, x_hat, p]):
            mask = torch.isnan(x)
            if torch.sum(mask) > 0:
                flag = True
                print("Hmmmm", i, torch.sum(mask))
        if flag:
            exit(1)
        flattened_grads4 = flattened_grads4 - p
    except Exception as e:
        print(e, "HAHAHA")
        if torch.isnan(flattened_grads1).any():
            print(1, "flattened_grads1 is faulty", torch.isnan(flattened_grads1).sum())
            if torch.isnan(flattened_grads2).any():
                print(1, "flattened_grads2 is faulty", torch.isnan(flattened_grads2).sum())
            if torch.isnan(flattened_grads3).any():
                print(1, "flattened_grads3 is faulty", torch.isnan(flattened_grads3).sum())
            exit(1)
        if torch.isnan(flattened_grads2).any():
            print(2, "flattened_grads2 is faulty", torch.isnan(flattened_grads2).sum())
            if torch.isnan(flattened_grads3).any():
                print(2, "flattened_grads3 is faulty", torch.isnan(flattened_grads3).sum())
            exit(1)
        if torch.isnan(flattened_grads3).any():
            print(3, "flattened_grads3 is faulty", torch.isnan(flattened_grads3).sum())
            exit(1)
        exit(1)
    
    return flattened_grads4

def assign_gradients_and_step(model, final_grads, c1_optimizer, shapes):
    offset = 0
    mask = torch.isnan(final_grads)
    if torch.sum(mask) > 0:
        print(torch.sum(mask))
        print("HELLO! FAULTY GRADIENTS")
        exit(1)
    for params2, temp_shapes2 in zip(c1_optimizer.param_groups[0]['params'], shapes):
        temp_shapes2_prod = np.prod(temp_shapes2)
        temp_grads2 = final_grads[offset:offset + temp_shapes2_prod]
        offset += temp_shapes2_prod
        temp_grads2 = temp_grads2.view(temp_shapes2)
        params2.grad.data = temp_grads2
    
    c1_optimizer.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()


def flatten_grads(model, loss, retain_graph=True, zero_grad=True):
    loss.backward(retain_graph=retain_graph)
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25.0)
    flattened_grads = []
    shapes = []
    for k, param in enumerate(model.parameters()):
        try:
            shapes.append(param.grad.data.shape)
        except:
            shapes.append(param.shape)
        try:
            flattened_grads.append(torch.flatten(param.grad.data.clone()))
        except:
            flattened_grads.append(torch.flatten(torch.zeros_like(param)))
    flattened_grads = torch.concat(flattened_grads)
    if zero_grad:
        model.zero_grad()
    return flattened_grads, shapes


def project_gradients_one_optimizer_robust(model, loss1, loss2, loss3, loss4, c1_optimizer):
        flattened_grads1, shapes1 = flatten_grads(model, loss1, retain_graph=True, zero_grad=True)
        flattened_grads2, shapes2 = flatten_grads(model, loss2, retain_graph=True, zero_grad=True)
        flattened_grads3, shapes3 = flatten_grads(model, loss3, retain_graph=True, zero_grad=True)
        flattened_grads4, shapes4 = flatten_grads(model, loss4, retain_graph=False, zero_grad=False)
        original_type = flattened_grads1.dtype
        
        flattened_grads1 = flattened_grads1.to(dtype=torch.float32)
        flattened_grads2 = flattened_grads2.to(dtype=torch.float32)
        flattened_grads3 = flattened_grads3.to(dtype=torch.float32)
        flattened_grads4 = flattened_grads4.to(dtype=torch.float32)
        
        
        # Capture only non-zero gradients
        valid_grads = capture_non_zero_grads([flattened_grads1, flattened_grads2, flattened_grads3, flattened_grads4])
        
        if len(valid_grads) == 0:
            final_grads = torch.zeros_like(flattened_grads1)
        
        elif len(valid_grads) == 1:
            final_grads = valid_grads[0]

        elif len(valid_grads) == 2:
            valid_grads[1] = project_two_grads(valid_grads[0], valid_grads[1])
            final_grads = valid_grads[0] + valid_grads[1]
        
        elif len(valid_grads) == 3:
            valid_grads[1] = project_two_grads(valid_grads[0], valid_grads[1])
            if torch.all(valid_grads[1] == 0):
                valid_grads.pop(1)
                valid_grads[1] = project_two_grads(valid_grads[0], valid_grads[1])
                final_grads = valid_grads[0] + valid_grads[1]
            else:
                valid_grads[2] = project_three_grads(valid_grads[0], valid_grads[1], valid_grads[2])
                final_grads = valid_grads[0] + valid_grads[1] + valid_grads[2]

        elif len(valid_grads) == 4:
            valid_grads[1] = project_two_grads(valid_grads[0], valid_grads[1])
            if torch.all(valid_grads[1] == 0):
                valid_grads.pop(1)
                valid_grads[1] = project_two_grads(valid_grads[0], valid_grads[1])
                if torch.all(valid_grads[1] == 0):
                    valid_grads.pop(1)
                    valid_grads[1] = project_two_grads(valid_grads[0], valid_grads[1])
                    final_grads = valid_grads[0] + valid_grads[1]
                else:
                    valid_grads[2] = project_three_grads(valid_grads[0], valid_grads[1], valid_grads[2])
                    final_grads = valid_grads[0] + valid_grads[1] + valid_grads[2]
            else:
                valid_grads[2] = project_three_grads(valid_grads[0], valid_grads[1], valid_grads[2])
                if torch.all(valid_grads[2] == 0):
                    valid_grads.pop(2)
                    valid_grads[2] = project_three_grads(valid_grads[0], valid_grads[1], valid_grads[2])
                    final_grads = valid_grads[0] + valid_grads[1] + valid_grads[2]
                else:
                    valid_grads[3] = project_four_grads(valid_grads[0], valid_grads[1], valid_grads[2], valid_grads[3])
                    final_grads = valid_grads[0] + valid_grads[1] + valid_grads[2] + valid_grads[3]
        
        final_grads = final_grads.to(dtype=original_type)
        flattened_grads1 = flattened_grads1.to(dtype=original_type)
        flattened_grads2 = flattened_grads2.to(dtype=original_type)
        flattened_grads3 = flattened_grads3.to(dtype=original_type)
        flattened_grads4 = flattened_grads4.to(dtype=original_type)
        
        return final_grads, shapes2, flattened_grads1, flattened_grads2, flattened_grads3, c1_optimizer

def find_linearly_independent_vectors(vectors):
    vectors = vectors.double()
    q, r = torch.linalg.qr(vectors, mode='reduced')
    threshold = 1e-6  # Numerical threshold for linear independence
    independent_columns = torch.abs(torch.diag(r)) > threshold  # Non-zero diagonal entries indicate independence
    
    return independent_columns
