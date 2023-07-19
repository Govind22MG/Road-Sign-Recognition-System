import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transform
import cnn_arch

# Paths to files/dirs
data_folder_path = 'data/'
model_path       = data_folder_path + 'model/RSBC_MODEL.pt'

# Labels
labels = ["Crosswalk", "Speedlimit", "Stop", "Trafficlight"]

# Hyper Parameters
in_channels   = 3
out_channels  = 4
img_transformer = transform.ToTensor()

# CNN Arch. Object        
NET = cnn_arch.CNN(in_channels, out_channels).to('cpu')

# Load model if exists
if(os.path.exists(model_path)):
    NET = torch.load(model_path, map_location='cpu')
    print('MODEL LOADED')
else:
    raise(FileNotFoundError)

# Function to get prediction
def predict(x):
    img = img_transformer(x)
    img = img.unsqueeze(0)
    pred = NET.forward(img)
    pred = pred.detach().numpy()[0]
    return(labels[np.argmax(pred)])
