import numpy as np 
import torch 
from torcheval.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

def min_max_clip (x, output_min, output_max):
    x = torch.where(x<output_min,output_min,x)
    x = torch.where(x>output_max,output_max,x)
    return x
    
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) 
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        if batch % 100 == 0:
            loss, current = loss, (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"loss: {loss:>7f} | last batch")
    return loss
     
def test(dataloader, model, loss_fn, output_mask, device, output_selection=None):
    MSEmetric = MeanSquaredError(multioutput="raw_values").to(device) 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
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
    
# def create_submit_output(model, dataloader, device):
#     output = np.zeros((dataloader.dataset.total_sample,368), dtype=np.float32)
#     sta_i = 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             isize = X.shape[0]
#             pred = model(X).cpu().numpy()
#             output[sta_i:sta_i+isize] = pred
#             sta_i = sta_i+isize
#     return output.astype('float64') 
    
def save_submit_file(filename, model, output_scale, output_offset, dataloader, epoch, r2, device): 
    output = create_submit_output(model, dataloader, device)
    output_denorm = output*output_scale+output_offset 
    data_submit = pd.read_csv('./data/sample_submission.csv')
    output_pd = pd.DataFrame(output_denorm, columns=data_submit.columns[1:])
    output_pd = output_pd*data_submit.values[:,1:]
    output_pd.insert(0, data_submit.columns[0], data_submit.iloc[:,0].values) 
    filename = f"./results/{filename}_e{epoch}_r2_{r2}.csv"
    # output_pd.to_csv(filename, index=False, float_format='%.4e')
    output_pd.to_csv(filename, index=False)
    print(f'Submission file saved! {filename}')
    return  

def create_submit_output(model, dataloader, device, output_selection=np.ones(368)): 
    output = np.zeros((dataloader.dataset.total_sample,output_selection.size), dtype=np.float32)
    sta_i = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            isize = X.shape[0]
            pred = model(X).cpu().numpy()
            output[sta_i:sta_i+isize] = pred
            sta_i = sta_i+isize
    return output.astype('float64') 
    
def save_submit_nc_sel(model_name, model, output_scale, output_offset, dataloader, epoch, r2_values, device, output_selection, output_name): 
    output = create_submit_output(model, dataloader, device, output_selection)
    output_denorm = output*output_scale+output_offset  
    R2mean = np.mean(r2_values) 
    R2mean_sel = np.mean(r2_values[output_selection]) 
    filename = f"./results/{model_name}_outi_{output_selection[0]}_e{epoch}_r2all_{R2mean:0.3f}_r2sel_{R2mean_sel:0.3f}.nc"  
    ds = xr.Dataset(
        {
            "data": (["id", "var"], output_denorm),
            "R2": (["var"], r2_values[output_selection])
        },
        coords={"id": (["id"], np.arange(output_denorm.shape[0])), 
                "var": (["var"], output_name.values) }  
        )
    ds.to_netcdf(filename)
    print(f'Submission file saved! {filename}')
    # save model checkpoint
    model_ckpt_path = f"./results/{model_name}_outi_{output_selection[0]}_e{epoch}_r2all_{R2mean:0.3f}_r2sel_{R2mean_sel:0.3f}.ckpt"  
    torch.save(model, model_ckpt_path)
    # torch.save(model.state_dict(), model_ckpt_path) # to be updated
    return  
    
def plot_loss(loss_train, loss_test):
    plt.subplots(figsize=(6, 3))
    plt.plot(loss_train)
    plt.plot(loss_test)

def plot_R2(R2value_list):
    # plot R2
    plt.subplots(figsize=(10, 3))
    # for R2value in [R2value_list[5],R2value_list[15],R2value_list[30], R2value_list[-1]]: 
    # for R2value in [R2value_list[5],R2value_list[-1]]: 
    for R2value in [R2value_list[0],R2value_list[-1]]:  
        plt.plot(R2value)
    # plt.plot(R2value,linewidth = 2)
    plt.plot([0,368],[0.8,0.8],c='grey')
    plt.plot([0,368],[0.5,0.5],c='grey')
    plt.plot([0,368],[0.2,0.2],c='grey')
    plt.ylim(0,1.01)
    plt.xlim(-1,368)
    plt.ylabel('R2score')
