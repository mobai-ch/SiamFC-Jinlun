import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Module
import numpy as np

# Alexnet branch
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return x

# Siamese network
class SiameseKMFC(nn.Module):
    def __init__(self):
        super(SiameseKMFC, self).__init__()
        self.framew_1 = nn.Parameter(data=Variable(torch.ones(1)), requires_grad=True)
        self.framew_2 = nn.Parameter(data=Variable(torch.ones(1)), requires_grad=True)
        self.framew_3 = nn.Parameter(data=Variable(torch.ones(1)), requires_grad=True)
        self.framew_4 = nn.Parameter(data=Variable(torch.ones(1)), requires_grad=True)
        self.framew_5 = nn.Parameter(data=Variable(torch.ones(1)), requires_grad=True)
        self.correlation_weight = nn.Parameter(data=Variable(torch.ones(1, 1, 17, 17)), requires_grad=True)
        self.correlation_bias = nn.Parameter(data=Variable(torch.ones(1, 1, 17, 17)), requires_grad=True)
    
    def forward(self, X, searchRegion):
        wsum = torch.exp(self.framew_1) + torch.exp(self.framew_2) + torch.exp(self.framew_3) \
            + torch.exp(self.framew_2) + torch.exp(self.framew_3)
        w1 = torch.exp(self.framew_1) / wsum
        w2 = torch.exp(self.framew_2) / wsum
        w3 = torch.exp(self.framew_3) / wsum
        w4 = torch.exp(self.framew_4) / wsum
        w5 = torch.exp(self.framew_5) / wsum
        feature_1 = X[0] * w1
        feature_2 = X[1] * w2
        feature_3 = X[2] * w3
        feature_4 = X[3] * w4
        feature_5 = X[4] * w5
        feature_summerize = feature_1 + feature_2 + feature_3 + feature_4 + feature_5
        template = torch.nn.Parameter(data=feature_summerize, requires_grad=False)
        result = torch.nn.functional.conv2d(searchRegion, weight=template)
        ret = result/torch.sum(result)
        return ret

def test():
    net = SiameseKMFC()
    net = net.float()
    x1 = np.random.random((1, 128, 6, 6))
    X = []
    for i in range(5):
        X.append(torch.from_numpy(x1).float())
    searchRegion = np.random.random((1, 128, 22, 22))
    searchRegion = torch.from_numpy(searchRegion).float()
#     result = net.forward(X, searchRegion)

# if __name__ == '__main__':
#     test()


    
        
        
