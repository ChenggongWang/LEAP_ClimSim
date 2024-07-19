import numpy as np 
import torch 
from torcheval.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tqdm import tqdm
def test_model_ensemble(dataloader, model_list, loss_fn, output_mask, device, output_selection=None):
    MSEmetric = MeanSquaredError(multioutput="raw_values").to(device) 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = []
            for name in model_list:
                model = model_list[name]
                model.eval()
                X, y = X.to(device), y.to(device)
                pred.append(model(X))
            pred = torch.stack(pred).mean(dim=0)
            test_loss += loss_fn(pred, y).item() 
            MSEmetric.update(pred, y) 
            
    test_loss /= num_batches 
    R2value  = 1-MSEmetric.compute().cpu().numpy() # target std =1 
    R2mean = np.mean(R2value)
    if output_selection is None:
        print(f"Test Error:  Avg loss: {test_loss:>.5f} R2score: {R2mean:>0.3f}  ")
    else:
        R2value_all = np.ones(368) # put it back to 368 dimension for comparison
        R2value_all[output_selection] = R2value 
        R2mean_all = np.mean(R2value_all)
        R2value = R2value_all
        print(f"Test Error:  Avg loss: {test_loss:>.5f} R2score_sele: {R2mean:>0.3f} \n R2score(R=1 for mask=0): {R2mean_all:>0.3f}  ")
        
    return test_loss, R2value 

def test_model_ensemble_rm_maxmin(dataloader, model_list, loss_fn, output_mask, device, output_selection=None):
    MSEmetric = MeanSquaredError(multioutput="raw_values").to(device)  
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = []
            for name in model_list:
                model = model_list[name]
                model.eval()
                X, y = X.to(device), y.to(device)
                pred.append(model(X))
            pred = torch.stack(pred)
            pred_max = pred.max(dim=0).values
            pred_min = pred.min(dim=0).values
            pred = (pred.sum(dim=0)-pred_max-pred_min)/(len(model_list)-2) # mean exclude max and min
            test_loss += loss_fn(pred, y).item() 
            MSEmetric.update(pred, y) 
            
    test_loss /= num_batches 
    R2value  = 1-MSEmetric.compute().cpu().numpy() # target std =1 
    R2mean = np.mean(R2value)
    if output_selection is None:
        print(f"Test Error:  Avg loss: {test_loss:>.5f} R2score: {R2mean:>0.3f}  ")
    else:
        R2value_all = np.ones(368) # put it back to 368 dimension for comparison
        R2value_all[output_selection] = R2value 
        R2mean_all = np.mean(R2value_all)
        R2value = R2value_all
        print(f"Test Error:  Avg loss: {test_loss:>.5f} R2score_sele: {R2mean:>0.3f} \n R2score(R=1 for mask=0): {R2mean_all:>0.3f}  ")
        
    return test_loss, R2value 
    
def test_model_ensemble_R2weighted(dataloader, model_list, R2ranks, loss_fn, output_mask, device, output_selection ):
    ds = xr.open_dataset('/tigress/cw55/work/2024_leap_climsim/data/output_maxmin.nc') 
    output_max = ((ds['max']-ds['mean'])/ds['std']).values[output_selection]
    output_min = ((ds['min']-ds['mean'])/ds['std']).values[output_selection]
    output_max = torch.tensor(output_max).to(device) 
    output_min = torch.tensor(output_min).to(device) 
    MSEmetric = MeanSquaredError(multioutput="raw_values").to(device)  
    num_batches = len(dataloader)
    test_loss = 0
    R2ranks = torch.tensor(R2ranks[:,:,output_selection]).to(device)  
    R2ranks_sum =  R2ranks.sum(dim=0) 
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = []
            for mi, name in enumerate(model_list):
                model = model_list[name]
                model.eval()
                X, y = X.to(device), y.to(device)
                pred_tmp = model(X)
                pred_tmp = torch.where(pred_tmp>output_max,output_max,pred_tmp)
                pred_tmp = torch.where(pred_tmp<output_min,output_min,pred_tmp)
                pred.append(pred_tmp)
            pred = torch.stack(pred) 
            pred = (pred*R2ranks).sum(dim=0)/R2ranks_sum  # weight = rank 
            test_loss += loss_fn(pred, y).item() 
            MSEmetric.update(pred, y) 
            
    test_loss /= num_batches 
    R2value  = 1-MSEmetric.compute().cpu().numpy() # target std =1 
    R2mean = np.mean(R2value)
    if output_selection is None:
        print(f"Test Error:  Avg loss: {test_loss:>.5f} R2score: {R2mean:>0.3f}  ")
    else:
        R2value_all = np.ones(368) # put it back to 368 dimension for comparison
        R2value_all[output_selection] = R2value 
        R2mean_all = np.mean(R2value_all)
        R2value = R2value_all
        print(f"Test Error:  Avg loss: {test_loss:>.5f} R2score_sele: {R2mean:>0.3f} \n R2score(R=1 for mask=0): {R2mean_all:>0.3f}  ")
        
    return test_loss, R2value
def test_model_ensemble_R2weighted(dataloader, model_list, R2ranks, loss_fn, output_mask, device, output_selection ):
    # ds = xr.open_dataset('/tigress/cw55/work/2024_leap_climsim/data/output_maxmin.nc') 
    # output_max = ((ds['max']-ds['mean'])/ds['std']).values[output_selection]
    # output_min = ((ds['min']-ds['mean'])/ds['std']).values[output_selection]
    # output_max = torch.tensor(output_max).to(device) 
    # output_min = torch.tensor(output_min).to(device) 
    MSEmetric = MeanSquaredError(multioutput="raw_values").to(device)  
    num_batches = len(dataloader)
    test_loss = 0
    R2ranks = torch.tensor(R2ranks[:,:,output_selection]).to(device)  
    R2ranks_sum =  R2ranks.sum(dim=0) 
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = []
            for mi, name in enumerate(model_list):
                model = model_list[name]
                model.eval()
                X, y = X.to(device), y.to(device)
                pred_tmp = model(X)
                # pred_tmp = torch.where(pred_tmp>output_max,output_max,pred_tmp)
                # pred_tmp = torch.where(pred_tmp<output_min,output_min,pred_tmp)
                pred.append(pred_tmp)
            pred = torch.stack(pred) 
            pred = (pred*R2ranks).sum(dim=0)/R2ranks_sum  # weight = rank 
            test_loss += loss_fn(pred, y).item() 
            MSEmetric.update(pred, y) 
            
    test_loss /= num_batches 
    R2value  = 1-MSEmetric.compute().cpu().numpy() # target std =1 
    R2mean = np.mean(R2value)
    if output_selection is None:
        print(f"Test Error:  Avg loss: {test_loss:>.5f} R2score: {R2mean:>0.3f}  ")
    else:
        R2value_all = np.ones(368) # put it back to 368 dimension for comparison
        R2value_all[output_selection] = R2value 
        R2mean_all = np.mean(R2value_all)
        R2value = R2value_all
        print(f"Test Error:  Avg loss: {test_loss:>.5f} R2score_sele: {R2mean:>0.3f} \n R2score(R=1 for mask=0): {R2mean_all:>0.3f}  ")
        
    return test_loss, R2value

def create_submit_output_ens(model_list, R2ranks, dataloader, device, output_selection=np.ones(368)): 
    R2ranks = torch.tensor(R2ranks[:,:,output_selection]).to(device)  
    R2ranks_sum =  R2ranks.sum(dim=0) 
    output = np.zeros((dataloader.dataset.total_sample,output_selection.size), dtype=np.float32)
    sta_i = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = []
            X, y = X.to(device), y.to(device)
            isize = X.shape[0]
            for mi, name in enumerate(model_list):
                model = model_list[name]
                model.eval()
                pred_tmp = model(X)
                pred.append(pred_tmp)
            pred = torch.stack(pred) 
            pred = (pred*R2ranks).sum(dim=0)/R2ranks_sum  # weight = rank 
            pred = pred.cpu().numpy()
            output[sta_i:sta_i+isize] = pred
            sta_i = sta_i+isize
    return output.astype('float64') 
    
def save_ens_csv(output, model_name, model_list, R2ranks, output_scale, output_offset, dataloader, r2_values, device, output_selection): 
    R2mean = np.mean(r2_values) 
    R2mean_sel = np.mean(r2_values[output_selection]) 
    filename = f"/tigress/cw55/work/2024_leap_climsim/results/{model_name}_r2all_{R2mean:0.3f}_r2sel_{R2mean_sel:0.3f}.csv"  
    
    # output = create_submit_output_ens(model_list, R2ranks, dataloader, device, output_selection)
    output_denorm = output*output_scale+output_offset  
    output_data = np.zeros((625000,368))
    output_data[:,output_selection] = output_denorm
    # csv
    submit_input = xr.open_dataset('/tigress/cw55/work/2024_leap_climsim/data/submit_input.nc')['data'][:,132:148]
    output_data[:,132:148] = -submit_input/1200
    data_submit = pd.read_csv('/tigress/cw55/work/2024_leap_climsim/data/sample_submission.csv')
    output_pd = pd.DataFrame(output_data, columns=data_submit.columns[1:])
    output_pd = output_pd*data_submit.values[:,1:]
    output_pd.insert(0, data_submit.columns[0], data_submit.iloc[:,0].values)    
    output_pd.to_csv(filename, index=False)
    print(f'Submission file saved! {filename}')
    return  output_pd
     
    