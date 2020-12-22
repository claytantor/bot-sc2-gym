import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):

# Layer	Input	    kernal 	    Stride	# filters	Activa	Output
# conv1	96x96x1	    8x8	        4	    32	        ReLU	[1, 32, 23, 23]
# conv2	23x23x32	4x4	        2	    64	        ReLU	[1, 64, 10, 10]
# conv3	10x10x64	3x3	        1	    64	        ReLU	[64, 8, 8]=4096
# fc4	8x8x64		                    512	        ReLU	512
# fc5	512			                    2	        Linear	2

    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=8, stride=4) 
            # output = torch.Size([1, 32, 23, 23])
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2) 
            # output = torch.Size([1, 64, 10, 10])
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1) 
            # output = torch.Size([1, 64, 8, 8])
        self.relu3 = nn.ReLU(inplace=True)

    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        out = self.conv1(x) 
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)


        out = self.conv3(out)
        out = self.relu3(out)

        return out
 

nn_network = Network()

view_dim = np.array([np.zeros((96, 96))])
# print(view_dim.shape)

screen = torch.tensor(view_dim, dtype=torch.float32, device=device)
out = nn_network.forward(screen)
print(out.shape)