import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import R2Score
from torch.optim  import  lr_scheduler
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# self defined func
from data_process_v2 import norm_mean_std, create_train_test_dataset_selectout, ClimSimDataset
from train_util_new import min_max_clip, test, train, save_submit_nc_sel
##########################################################################################################
# define model
##########################################################################################################
class CNN_block(nn.Module):
    def __init__(self, channel_dim_in, channel_dim_out, kernel_size):
        super().__init__() 
        padding_k = kernel_size//2
        self.block = nn.Sequential(  
            nn.Conv2d(channel_dim_in, channel_dim_out, kernel_size,   padding=padding_k), 
            nn.BatchNorm2d(channel_dim_out),  
            nn.ReLU(),
            nn.Conv2d(channel_dim_out, channel_dim_out, kernel_size,   padding=padding_k), 
            nn.BatchNorm2d(channel_dim_out),  
            nn.ReLU()
        )
    def forward(self, x):  
        out = self.block(x) 
        return out    
class Unet2d(nn.Module):
    def __init__(self, input_dim, output_dim, output_min, output_max, output_mask, device):
        super().__init__() 
        self.output_mask = torch.tensor(output_mask).to(device)
        self.output_min  = output_min.to(device)
        self.output_max  = output_max.to(device)
        self.device = device 
        channels = [1, 128, 128, 128] 
        kernel_size = 5
        self.channels = channels
        layer = len(channels) 
        self.layer = layer
        self.cnn2d_blocks_down = nn.ModuleList([CNN_block(channels[i],
                                                          channels[i+1], 
                                                          kernel_size).to(device) 
                                               for i in range(layer-1)]) 
        self.cnn2d_blocks_up = nn.ModuleList([CNN_block(channels[layer-i]+channels[layer-i-1],
                                                        channels[layer-i-1], 
                                                        kernel_size).to(device) 
                                              for i in range(1,layer-1)]) 
        self.down_sample = nn.MaxPool2d(2)  # size // 2
        self.up_sample = nn.ModuleList([nn.ConvTranspose2d(channels[layer-i],
                                                           channels[layer-i],
                                                           2,2).to(device)  # size x 2
                                       for i in range(1,layer-1)])
        self.cnn_out =  nn.Conv2d(channels[1], 2, 1) 
        self.dense = nn.Linear(576*2, output_dim)
    def forward(self, x): 
        identity = x
        x_scalar = x[:,360:376]
        x = torch.concat([x_scalar, x_scalar, x, x_scalar, x_scalar, ],dim=1) #Feture: 490 =>  16*2 + 490 +16*2  => 554 
        x = nn.functional.pad(x,(11,11)) # 554->576
        x = rearrange(x, "b (h w) -> b 1 h w", h=24) # b 1 24 24
        y = []
        #down
        for i in range(self.layer-1):
            x = self.cnn2d_blocks_down[i](x)
            if i < self.layer-2:
                y.append(x)
                x = self.down_sample(x)
            # print(x.shape)
            
        #up
        for i in range(self.layer-2):
            x = self.up_sample[i](x) 
            x_prev = y.pop() 
            x = torch.concat([x,x_prev],dim=1)
            x = self.cnn2d_blocks_up[i](x)
            # print(x.shape) 
        x = self.cnn_out(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch  
        x = self.dense(x)
        # x[:,132:148] =  -identity[:,132:148] # replace top q0002 
        # #x = min_max_clip(x, self.output_min, self.output_max)
        # x = torch.where(self.output_mask!=0,x,0)
        return x 

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
    
     
    batch_size = 2048
    submit = True
    model_name = 'Unet2D_k5_v21_all' 
    
    data_source = 'large' # or small for quick test
    
    R_crit = 0.65 # threshold to save output 
    T_0 = 10 # CosineAnnealingWarmRestarts
    epochs = 4*T_0 # total epoch NxT_0 N = 1,2,3,4
    
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
        = create_train_test_dataset_selectout(dataset = data_source, norm='std', test_size=0.05, output_selection = output_mask, submit=submit)
        # submit data
        submit_dataloader = DataLoader(submit_input, batch_size=1024, shuffle=False)
    else:
        train_dataset, test_dataset, output_mask, input_offset, input_scale, output_offset, output_scale, output_selection, output_name \
        = create_train_test_dataset_selectout(dataset = data_source, norm='std', test_size=0.05, output_selection = output_mask, submit=submit)
          
    output_min = torch.tensor(train_dataset.output.min(axis=0))
    output_max = torch.tensor(train_dataset.output.max(axis=0))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
    
    
    input_dim = input_offset.size
    output_dim = output_offset.size
    model_Unet2d = Unet2d(input_dim, output_dim, 
                          output_min, output_max,
                          output_mask,device).to(device)
    print(model_Unet2d)
    pytorch_total_params = sum(p.numel() for p in model_Unet2d.parameters())
    print(f'model parameters: {pytorch_total_params}')
    
    ##########################################################################################################
    # train model
    ########################################################################################################## 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model_Unet2d.parameters(), lr=1e-3, weight_decay = 1e-5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=T_0) 
    R2value_list = []
    loss_train = []
    loss_test = []
    for t in tqdm(range(epochs)):
        print(f"\n------------ Epoch {t+1} ------------{model_name}-------------")
        print('LR:',scheduler.get_last_lr())
        l_train = train(train_dataloader, model_Unet2d, loss_fn, optimizer, device)
        l_test, R2value = test(test_dataloader, model_Unet2d, loss_fn, output_mask, 
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
        if (t+1)%T_0 == 0 :
            print(R2value) 
            if submit :
                R_crit = R2mean
                save_submit_nc_sel(model_name, model_Unet2d, output_scale, output_offset, 
                                 submit_dataloader, t+1, R2value, device, output_selection, output_name)
    print("Done!")
    print(R2mean)
    print(R2value)
