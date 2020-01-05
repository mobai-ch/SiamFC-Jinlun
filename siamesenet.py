import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class alexnetpart(nn.Module):
    def __init__(self):
        super(alexnetpart, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=2)
        self.conv2 = nn.Conv2d(96, 256, 5)
        self.conv3 = nn.Conv2d(256, 192, 3)
        self.conv4 = nn.Conv2d(192, 192, 3)
        self.conv5 = nn.Conv2d(192, 128, 3)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3), stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3), stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

class Siamesenet(nn.Module):
    def __init__(self, branch):
        super(Siamesenet, self).__init__()
        self.branch = branch
        self.bn_adjust = nn.Conv2d(1, 1, 1, stride=1, padding=0)

    def forward(self, example_patch, search_patch):
        # Example patch
        example_patch = self.branch(example_patch)
        # Search patch
        search_patch = self.branch(search_patch)       
        convd_out = self.convolution_example_search(example_patch, search_patch)
        return self.bn_adjust(convd_out)

    def convolution_example_search(self, example, search):
        output = []
        size = example.size()
        conv_example = torch.nn.Conv2d(size[0], 1, size[1], padding=1)
        for i in range(size[0]):
            output.append(F.conv2d(search[i:i+1, :, :, :], example[i:i+1, :, :, :]))
        result = torch.cat(output, dim=0)
        return result

def test_current_net():
    alex = alexnetpart()
    net = Siamesenet(branch = alex)
    example_patch = Variable(torch.randn(8, 3, 127, 127))
    search_patch = Variable(torch.randn(8, 3, 255, 255))
    map = net.forward(example_patch, search_patch)
    print(map.size())

# if __name__ == '__main__':
#     test_current_net()