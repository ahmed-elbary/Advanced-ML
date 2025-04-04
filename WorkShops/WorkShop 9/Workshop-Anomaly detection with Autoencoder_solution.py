# %% [markdown]
# ## Workshop 10: Anomaly detection with autoencoder 
#                                                         

# %% [markdown]
# #####  -  downloading the data  through the link below  
# https://drive.google.com/drive/folders/1eB1OUAKrl4KXvgrNbmzRqywZUCqBuga6?usp=drive_link
# ##### -  motivation to use npy format to store your data
# https://towardsdatascience.com/what-is-npy-files-and-why-you-should-use-them-603373c78883

# %% [markdown]
# The folder contains 2 subfolders, 'trainingset' and 'testingset'
# ```
# trainingset/
# ├── train.npy
# 
# testingset/
# ├── test.npy
# ...
# ```

# %% [markdown]
# ### import packages 

# %%
import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torchvision.transforms as transforms

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from torch.optim import Adam, AdamW

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq, kmeans

from tqdm import tqdm
import pandas as pd

import pdb  # use pdb.set_trace() to set breakpoints for debugging

# %% [markdown]
# ### loading data 

# %%
train_all = np.load('C:/Users/Student/Desktop/Advanced-ML/WorkShops/WorkShop 9/FaceData_Autoencoders/trainingset/train.npy')
test_all = np.load('C:/Users/Student/Desktop/Advanced-ML/WorkShops/WorkShop 9/FaceData_Autoencoders/testingset/test.npy')

train = train_all[0:1000, :, :, :] ## using subdataset for testing codes 
test = test_all[0:500, :, :, :] ## using subdataset for testing codes
print (train_all.shape)
print(test_all.shape)
print (train.shape)
print(test.shape)

# %% [markdown]
# ## Model and loss
# refer to 
# [1] https://github.com/L1aoXingyu/pytorch-beginner
# [2] https://github.com/jellycsc/PyTorch-CIFAR-10-autoencoder/

# %%
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    ########################################################################################
    
    ###### write the network forward function here and return the output######
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return (x)
            
    ########################################################################################    

# %% [markdown]
# ### Dataset module
# 
# The transform function here normalizes image's pixels from [0, 255] to [-1.0, 1.0].

# %%
class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),
            transforms.Lambda(lambda x: 2. * x / 255. - 1.),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)

# %% [markdown]
# ## Training 
# ##### Initialize
# - hyperparameters
# - dataloader
# - model
# - optimizer & loss

# %%
num_epochs = 15 # 50
batch_size = 1000 # medium: smaller batchsize 10000 by default
learning_rate = 1e-3
model_type = 'cnn'
# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model

model = conv_autoencoder().cpu()
## model = conv_autoencoder().cuda() ## only if you have a GPU

##############################################################################################

#######  Loss and optimizer ####### 
criterion = torch.nn.MSELoss() ##  define mean square error loss functin here

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############################################################################################## 

# %% [markdown]
# ## Training loop

# %%
best_loss = np.inf
model.train()
tqdm_train = tqdm(range(num_epochs))

for epoch in tqdm_train:
    tot_loss = list()
    for data in train_dataloader:

        # ===================loading=====================

        img = data.float().cpu()
        #img = data.float().cuda() # for GPU case

        # ===================forward=====================
        output = model(img)
        
        ####################################################

        loss = criterion(output, img)  

        ###################################################
        
        
        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, 'best_model_{}.pt'.format(model_type))
    # ===================log========================
    
    print('Epoch: ', epoch, 'loss: ', mean_loss)
    

    # ===================save_last========================
    torch.save(model, 'last_model_{}.pt'.format(model_type))




    print(type(test))  # Should be <class 'numpy.ndarray'>


# %% [markdown]
# # Inference
# Model is loaded and generates its anomaly score predictions.
# 
# ## Initialize
# - dataloader
# - model
# - prediction file

# %%
eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)  #######################################################
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=0) # num_workers = 1 if you have GPUs
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = 'C:/Users/Student/Desktop/Advanced-ML/WorkShops/WorkShop 9/last_model_cnn.pt'
model = torch.load(checkpoint_path, weights_only=False)
model.eval()

# prediction file
out_file = 'PREDICTION_FILE.csv'

# %%
anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):

        img = data.float().cpu()
        output = model(img)
        loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['Predicted'])
df.to_csv(out_file, index_label = 'Id') # saving the results in the csv file

# %% [markdown]
# # Extra work:
# ### - updating the conv_autoencoder class, trying to use the resnet as an encoder or your own developed encoder 

# %%



