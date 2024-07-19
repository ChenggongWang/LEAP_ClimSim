import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def norm_mean_std(x, x_mask=1, offset=None, scale=None,):
    x = xr.open_dataset(x).data.values[:,x_mask]
    if offset is None:
        x_std = x.std(axis=0)
        x_mean = x.mean(axis=0)
        # x_std[147] = 1e-6 # hard code
        # x_mean[147] = 0
        # x_std[135:147] = x_std[135:147]/100
        # x_std[140:142] = x_std[140:142]/10
        offset, scale = x_mean, x_std
        max_min = x.max(axis=0)-x.min(axis=0)
        scale = np.where(max_min==0, 1, scale)
    x = (x-offset)/scale
    # # clip data +- 100 std
    # x_range = 100
    # x = np.where(x>x_range,  x_range,x)  
    # x = np.where(x<-x_range,-x_range,x)  
    x = np.where(x_mask!=0,x,0) # mask out irrelevant values
    return x.astype(np.float32), offset, scale
    
# def norm_quantile(x,  x_mask=1, offset=None, scale=None,):
#     x = xr.open_dataset(x).data.values
#     x = np.where(x_mask!=0,x,0) # mask out irrelevant values 
#     if offset is None:
#         # range99 = np.quantile(x,[0.001,0.5,0.999],axis=0)
#         range99 = np.quantile(x[:2000000],[0, 0.01, 0.5, 0.99, 1.0],axis=0) 
#         max_min = range99[-1]-range99[0]
#         scale = (range99[3]-range99[1])/5
#         x_std = x.std(axis=0)
#         scale = np.where(max_min==0, 1, scale)
#         scale = (x_std+scale)/2
#         offset = range99[2]
#     x = (x-offset)/scale  
#     # # clipdata +- 100 scale
#     # x_range = 100000
#     # x = np.where(x>x_range,   x_range, x) 
#     # x = np.where(x<-x_range, -x_range, x)  
#     x = np.where(x_mask!=0,x,0) # mask out irrelevant values 
#     return x.astype(np.float32), offset, scale 
    
# def denorm_mean_std(x,x_mean,x_std):   
#     x = x.astype(np.float64)*x_std+x_mean  
#     return x   
    
def create_train_test_dataset_selectout(dataset = 'small', norm='std', submit=False, output_selection=None, test_size=0.3, randseed=123):
    # use small dataset by default
    # load all data into cpu memory
    
    # use std and mean in the !!! large dataset !!!
    root = '/tigress/cw55/work/2024_leap_climsim'
    input_raw = f'{root}/data/input_large.nc'
    output_raw = f'{root}/data/output_large.nc'
    input_offset  = xr.open_dataset(input_raw)['mean'].values
    input_scale   = xr.open_dataset(input_raw)['std'].values
    output_offset = xr.open_dataset(output_raw)['mean'].values
    output_scale  = xr.open_dataset(output_raw)['std'].values
    input_selection = np.argwhere(input_scale != 1).squeeze() # remove fixed data
    input_offset  = input_offset[input_selection]
    input_scale   = input_scale [input_selection]
    print('Removed fixed input. Now the number of the input feature is ',input_selection.shape)
    input_raw = f'{root}/data/input_{dataset}.nc'
    output_raw = f'{root}/data/output_{dataset}.nc'
    if submit:
        input_submit = f'{root}/data/submit_input.nc'
    # total_sample = input_raw.data.shape[0]
    # print(f'sample size: {total_sample}')
    # create output mask
    output_mask = np.ones((368))
    output_mask[60:72] = 0.0    # sphum
    output_mask[120:132] = 0    # liqud cloud
    output_mask[180:192] = 0    # ice cloud
    output_mask[240:252] = 0    # u
    output_mask[300:312] = 0    # v
    if output_selection is not None: # select output by mask
        output_mask = output_mask*output_selection
    output_selection = np.argwhere(output_mask == 1).squeeze() # pick output data
    output_dim_name = xr.open_dataset(output_raw)['var'][output_selection]
    output_offset  = output_offset[output_selection]
    output_scale   = output_scale [output_selection]
    print('Removed fixed input. Now the number of the output feature is', output_selection.shape)
    
    input_raw, input_offset, input_scale    = norm_mean_std(input_raw,  input_selection, 
                                                            input_offset, input_scale )
    output_raw, output_offset, output_scale = norm_mean_std(output_raw,  output_selection, 
                                                            output_offset, output_scale )
    if submit:
        input_submit_raw, input_offset, input_scale = \
            norm_mean_std(input_submit,  input_selection, input_offset, input_scale ) 

    
    # split train, test
    X_train, X_test, y_train, y_test = train_test_split(input_raw, output_raw, test_size=test_size, random_state=randseed)
    print(f'train sample size: {X_train.shape[0]}')
    print(f'test sample size: {X_test.shape[0]}') 

    train_dataset  = ClimSimDataset(X_train, y_train )
    test_dataset   = ClimSimDataset(X_test, y_test )
    if submit:
        submit_dataset = ClimSimDataset(input_submit_raw, input_submit_raw)
        return train_dataset, test_dataset, output_mask, input_offset, input_scale, output_offset, output_scale, submit_dataset, output_selection, output_dim_name
    else:
        return train_dataset, test_dataset, output_mask, input_offset, input_scale, output_offset, output_scale, output_selection, output_dim_name
 
        
class ClimSimDataset(Dataset):
    def __init__(self, x_data, y_data):  
        self.input = x_data
        self.output = y_data
        self.total_sample = self.input.shape[0] 

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx): 
        x = self.input[idx]
        y = self.output[idx]
        return x, y

def create_submit_file(output):
    
    return
    
