import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
from einops import rearrange
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
## # define model
########################################################################################################## 
class PatchEmbedding_PosiEmbedding(nn.Module):
    def __init__(self, N, embedding_dims, p_size):
        super().__init__()  
        self.PatchEmbedding = nn.Conv1d(1, embedding_dims, kernel_size=p_size, stride=p_size)
        self.PosiEmbedding = nn.Parameter(torch.rand((1, N, embedding_dims), requires_grad=True))
    def forward(self, x):  
        out = self.PatchEmbedding(x)
        out = rearrange(out, "b c n -> b n c")
        # print(out.shape, self.PosiEmbedding.shape)
        out = out + self.PosiEmbedding
        return out    # b x N x embedding
class MLP(nn.Module):
  def __init__(self, embedding_dims, mlp_size):
    super().__init__()
    self.embedding_dims = embedding_dims
    self.mlp_size = mlp_size 

    self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)
    self.mlp = nn.Sequential(
        nn.Linear(in_features = self.embedding_dims, out_features = self.mlp_size), 
        nn.GELU(), 
        nn.Linear(in_features = self.mlp_size, out_features = self.embedding_dims),  
    )
  def forward(self, x):
    return self.mlp(self.layernorm(x))
      
class MSA(nn.Module):
  def __init__(self,  embedding_dims, num_heads = 12):
    super().__init__() 
    self.embedding_dims = embedding_dims
    self.num_head = num_heads 
    self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims) 
    self.multiheadattention =  nn.MultiheadAttention(num_heads = num_heads,
                                                     embed_dim = embedding_dims,
                                                     dropout = 0,
                                                     batch_first = True)

  def forward(self, x):
    x = self.layernorm(x)
    output,_ = self.multiheadattention(query=x, key=x, value=x,need_weights=False)
    return output
class Trans_Block(nn.Module):
  def __init__(self, embedding_dims,
               mlp_size = 3072,
               num_heads = 12 ):
    super().__init__()

    self.msa_block = MSA(embedding_dims = embedding_dims,
                                                 num_heads = num_heads )

    self.mlp_block = MLP(embedding_dims = embedding_dims,
                                                    mlp_size = mlp_size ) 
  def forward(self,x):
    x = self.msa_block(x) + x
    x = self.mlp_block(x) + x 
    return x
      
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, output_min, output_max, device):
        super().__init__() 
        self.output_min  = output_min.to(device)
        self.output_max  = output_max.to(device)
        self.device = device   

        p_size = 15 
        N = (input_dim+20)//p_size # 490 =>  490 + 20 => 510 => 510/p_size
        self.embedding_dims = 360
        self.num_transformer_layers = 4
        self.mlp_size  = 3072
        self.num_heads = 12
        self.embedding = PatchEmbedding_PosiEmbedding(N, self.embedding_dims, p_size)
        
        self.encoder = nn.Sequential(*[Trans_Block(embedding_dims = self.embedding_dims,
                                                        mlp_size  = self.mlp_size,
                                                        num_heads = self.num_heads) 
                                        for _ in range(self.num_transformer_layers)])  
        self.out_linear =  nn.Linear(N*self.embedding_dims, output_dim)
    def forward(self, x): 
        identity = x
        x_scalar = x[:,360:376]
        x = torch.concat([x_scalar, x ],dim=1) #Feture: 490 =>  16 + 490   => 506 
        x = nn.functional.pad(x,(2,2)) # 506->510
        # print(x.shape)
        x = self.embedding(x[:,None,:])     # B x N x E : B x 34 x 64
        x = self.encoder(x)       # B x N x E : B x 34 x 64
        x = torch.flatten(x, 1) # flatten all dimensions except batch  
        x = self.out_linear(x)
        # x[:,132:148] =  -identity[:,132:148] # replace top q0002 
        #x = min_max_clip(x, self.output_min, self.output_max)
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
    model_name = 'Trans_1018_big360_v23'
    
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
        = create_train_test_dataset_selectout(dataset = data_source, norm='std', test_size=0.15, output_selection = output_mask, submit=submit, randseed=321)
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
    model_Trans = Transformer(  input_dim, output_dim, 
                                output_min, output_max,
                                device).to(device)
    
    print(model_Trans)
    pytorch_total_params = sum(p.numel() for p in model_Trans.parameters())
    print(f'model parameters: {pytorch_total_params}')
    
    ##########################################################################################################
    # train model
    ##########################################################################################################
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model_Trans.parameters(), lr=1e-4, weight_decay = 1e-5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=T_0)
    R2value_list = []
    loss_train = []
    loss_test = []
    for t in tqdm(range(epochs)):
        print(f"\n------------ Epoch {t+1} ------------{model_name}-------------")
        print('LR:',scheduler.get_last_lr())
        l_train = train(train_dataloader, model_Trans, loss_fn, optimizer, device)
        l_test, R2value = test(test_dataloader, model_Trans, loss_fn, output_mask, 
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
            if submit:
                R_crit = R2mean
                save_submit_nc_sel(model_name, model_Trans, output_scale, output_offset, 
                                 submit_dataloader, t+1, R2value, device, output_selection, output_name)
    print("Done!")
    print(R2mean)
    print(R2value)
