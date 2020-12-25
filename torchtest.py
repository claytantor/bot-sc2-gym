import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from itertools import count

from pathlib import Path

# if gpu is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

pickleDir = os.path.join(Path(__file__).parent.absolute(),'pickle')

# eb02ece5 works
# 52d7e33d freeze

def save_array(pickleFile, array):
    np.save(pickleFile, array, allow_pickle=True, fix_imports=True)

def load_array(pickleFile):
    return np.load(pickleFile)

def show_array(array):
    np.set_printoptions(threshold=sys.maxsize)
    print(array)

def run_pickle_gen():

    for t in count():
        pickleFile = os.path.join(pickleDir,'tt-{}'.format(t)) 

        screen = np.random.rand(1, 96, 96)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / np.amax(screen)
        screen = np.array([screen])

        if t % 1000 == 0 and t != 0:
            save_array(pickleFile, screen)

        screen = torch.tensor(screen, device=torch.device("cuda:0"))
        print(t, screen)

def main(argv):
    items = ['eb02ece5','52d7e33d']
    for item in items:
        print('========= item {}'.format(item))
        pickleFile = os.path.join(pickleDir,'pysc2-{}.npy'.format(item))
        arrayval = load_array(pickleFile)
        show_array(arrayval)

        arrayval = torch.tensor(arrayval, device=torch.device("cuda:0"))
        print("converted {} to tensor.".format(arrayval))




if __name__ == "__main__":
    main(sys.argv[1:])