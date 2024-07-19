import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import R2Score
from torch.optim  import  lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# self defined func
from data_process_v2 import norm_mean_std, create_train_test_dataset_selectout, ClimSimDataset
from train_util_new import min_max_clip, test, train, save_submit_nc_sel
##########################################################################################################
# define model
##########################################################################################################
class ln_block(nn.Module):
    def __init__(self, dim_in, dim_out ):
        super().__init__()  
        self.block = nn.Sequential( 
            nn.Linear(dim_in, dim_out),
            nn.BatchNorm1d(dim_out),
            # nn.ReLU(),
            # nn.Linear(dim_out, dim_out),
            # nn.BatchNorm1d(dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
            nn.BatchNorm1d(dim_out) 
        )  
        self.acti = nn.ReLU()
    def forward(self, x):  
        out = self.block(x) + x
        out = self.acti(out)
        return out
class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, output_min, output_max, output_mask, device):
        super().__init__() 
        self.output_mask = torch.tensor(output_mask).to(device)
        self.output_min  = output_min.to(device)
        self.output_max  = output_max.to(device)
        self.device = device
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.ReLU(), 
        )  
        self.linear_relu_stack2 = nn.Sequential(
            ln_block(input_dim*4, input_dim*4),
            ln_block(input_dim*4, input_dim*4)
        )   
        self.linear_relu_stack3 =  nn.Linear(input_dim*4, output_dim)  
        
    def forward(self, x): 
        y = self.linear_relu_stack1(x) 
        y = self.linear_relu_stack2(y) 
        y = self.linear_relu_stack3(y) 
        # y = min_max_clip(y, self.output_min, self.output_max)
        # y[:,132:148] =  -x[:,132:148] # replace top q0002 
        # y = torch.where(self.output_mask!=0,y,0)
        return y
        
if __name__ == "__main__":  
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
          
    batch_size = 10000
    submit = True
    model_name = 'FNN_v2_all'
    
    data_source = 'large' # or small for quick test
    
    R_crit = 0.65 # threshold to save output 
    T_0 = 20 # CosineAnnealingWarmRestarts
    epochs = 2*T_0 # total epoch NxT_0 N = 1,2,3,4
    
    ##########################################################################################################
    # output_selection
    ##########################################################################################################
    output_mask = np.ones(368)
    output_mask[132:148] = 0 # remove q0002 132-147
    
    ##########################################################################################################
    # load data
    ##########################################################################################################
    if submit:
        train_dataset, test_dataset, output_mask, input_offset, input_scale, output_offset, output_scale, submit_input, output_selection, output_name \
        = create_train_test_dataset_selectout(dataset = data_source, norm='std', test_size=0.15, output_selection = output_mask, submit=submit)
        # submit data
        submit_dataloader = DataLoader(submit_input, batch_size=1024, shuffle=False)
    else:
        train_dataset, test_dataset, output_mask, input_offset, input_scale, output_offset, output_scale, output_selection, output_name \
        = create_train_test_dataset_selectout(dataset = data_source, norm='std', test_size=0.15, output_selection = output_mask, submit=submit)
          
    output_min = torch.tensor(train_dataset.output.min(axis=0))
    output_max = torch.tensor(train_dataset.output.max(axis=0))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
    
    
     
    input_dim = input_offset.size
    output_dim = output_offset.size
    model_FNN = FNN(input_dim, output_dim, 
                    output_min, output_max,
                    output_mask,device).to(device)
    print(model_FNN)
    pytorch_total_params = sum(p.numel() for p in model_FNN.parameters())
    print(f'model parameters: {pytorch_total_params}')
    
    ##########################################################################################################
    # train model
    ##########################################################################################################
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model_FNN.parameters(), lr=1e-3, weight_decay = 1e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=T_0)
    R2value_list = []
    loss_train = []
    loss_test = []
    for t in tqdm(range(epochs)):
        print(f"\n------------ Epoch {t+1} ------------{model_name}-------------")
        print('LR:',scheduler.get_last_lr())
        l_train = train(train_dataloader, model_FNN, loss_fn, optimizer, device)
        l_test, R2value = test(test_dataloader, model_FNN, loss_fn, output_mask, 
                               device, output_selection=output_selection)
        scheduler.step()
        R2value_list.append(R2value)
        loss_train.append(l_train)
        loss_test.append(l_test)
        R2mean = np.mean(R2value) 
        R2mean_sel = np.mean(R2value[output_selection]) 
        if( t==0 and R2mean<0.50) or (t==5 and R2mean<0.60): 
            print('---------------------------------------------------------------------------------')
            print('early stop for bad models')
            print(f'Epoch: {t+1} R2: {R2mean} too small')
            print('---------------------------------------------------------------------------------')
            break 
        if (t+1)%T_0 == 0: 
            print(R2value) 
            if submit and R2mean > R_crit+0.005 or t+1 == epochs:
                R_crit = R2mean
                save_submit_nc_sel(model_name, model_FNN, output_scale, output_offset, 
                                 submit_dataloader, t+1, R2value, device, output_selection, output_name)
    print("Done!")
    print(R2mean)
    print(R2value)
