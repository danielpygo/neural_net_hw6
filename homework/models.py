import torch
from torch import nn
import time
# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes

class Block(nn.Module):
    '''
    Your code for resnet blocks
    '''
    def __init__(self, in_channel,bottle_channel, out_channel, stride):
        super(Block, self).__init__()
        '''
        Your code here
        '''
        self.res_block = nn.Sequential(
        	nn.Conv2d(in_channel,bottle_channel,1,stride,0),
            nn.BatchNorm2d(bottle_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        	nn.LeakyReLU(.01),
        	nn.Conv2d(bottle_channel,bottle_channel,3,stride,0),
            nn.BatchNorm2d(bottle_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        	nn.LeakyReLU(.01),
        	nn.Conv2d(bottle_channel,out_channel,1,stride,1),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        	nn.LeakyReLU(.01)
        )

    def forward(self, x):
        '''
        Your code here
        '''
        return self.res_block(x)




class ConvNetModel(nn.Module):
    '''
    Your code for the model that computes classification from the inputs to the scalar value of the label.
    Classification Problem (1) in the assignment
    '''

    def __init__(self):
        super(ConvNetModel, self).__init__()
        '''
        Your code here
        '''
        self.b1 = Block(3,10,10,1)
        self.b2 = Block(10,7,26,1)
        self.b3 = Block(26,12,40,1)
        self.b4 = Block(40,20,55,1)
        self.b5 = Block(55,25,64,1)
        self.relu = nn.ReLU(True) #try leaky relu
        self.pool = nn.AvgPool2d(3)
        #
        self.conv1 = nn.Conv2d(3, 10, 1, 1, 0)
        self.conv2 = nn.Conv2d(10, 26, 1, 1, 0)
        self.conv3 = nn.Conv2d(26, 40, 1, 1, 0)

        self.conv4 = nn.Conv2d(40, 55, 1, 1, 0)
        self.conv5 = nn.Conv2d(55, 64, 1, 1, 0)
        self.fc1 = nn.Linear(2646, 6)
        self.bn1 = nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv7 = nn.Conv2d(64, 6, 1, 1, 0)

    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        Your code here
        '''
        id = x
        id = self.conv1(id)
        x = self.b1(x)
        # print(x.shape)
        x = x.add(id)
        # print(x.shape)

        id = x
        id = self.conv2(id)
        x = self.b2(x)
        x = x.add(id)
        id = x
        id = self.conv3(id)
        x = self.b3(x)
        x = x.add(id)

        id = x
        id = self.conv4(id)
        x = self.b4(x)
        x = x.add(id)

        id = x
        start = time.time()
        id = self.conv5(id)
        x = self.b5(x)
        x = x.add(id)
        end = time.time()
        # print("Time for b5: ", end - start)

        x = self.conv7(x)
        x = self.bn1(x)

        x = self.pool(x)
        x = x.view(-1,2646)
        start1 = time.time()
        x = self.fc1(x)
        end1 = time.time()
        # print("Time for fc1: ", end1 - start1)
        # print()
        # x = self.linear(x)
        return x
