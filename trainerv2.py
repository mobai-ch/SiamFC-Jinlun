import cv2
import numpy as np
from torch import nn
from torch.autograd import Variable
from siamesenetv2 import SiameseKMFC
import torch
import os,sys
import math
import torch.optim as optim
from siamesenetv2 import AlexNet
import queue
from datautil import DataUtil

class limited_queue:
    def __init__(self, K):
        self.queue = queue.Queue()
        self.K = K
    
    def insert(self, x):
        if self.queue.empty():
            for i in range(self.K):
                self.queue.put(x)
        else:
            self.queue.put(x)
            self.queue.get()
    
    def aslist(self):
        out_list = []
        for i in range(self.queue.qsize()):
            out_list.append(self.queue.get())
        for out in out_list:
            self.queue.put(out)
        return out_list    
    
    def empty(self):
        for i in range(self.queue.qsize()):
            self.queue.get()
        

class trainer:
    def __init__(self, branch, dataset):
        self.dataset = dataset
        self.fnet = branch.float()
        self.bnet = SiameseKMFC()
        self.bnet = self.bnet.float()
        self.optimizer = optim.SGD(self.bnet.parameters(), lr=0.001, momentum=0.9)

    def loadParameter(self, path = ""):
        if os.path.exists(path):
            self.bnet.load_state_dict(torch.load(path))
    
    def normalLoss(self, output, target):
        nums = output.shape[0]
        truetar = np.zeros((nums, 1, 17, 17))
        for k in range(nums):
            [x, y] = target[k]
            lengthx = max(17-x, x)
            lengthy = max(17-y, y)
            for j in range(x-lengthx, x+lengthx):
                for i in range(y-lengthy, y+lengthy):
                    if i >= 17 or j >= 17 or i < 0 or j < 0:
                        pass
                    else:
                        radius = math.sqrt((i-x)**2 + (j-y)**2)
                        truetar[k, 0, i, j] = math.exp(-radius/9)
        truetar = torch.from_numpy(truetar).float()
        Loss = torch.sum((truetar - output)*(truetar - output))
        return Loss
    
    def train_step(self, X, searchRegion, target):
        self.optimizer.zero_grad()
        output = self.bnet.forward(X, searchRegion)
        loss = self.normalLoss(output, target)
        loss.requires_grad = True
        loss.backward()
        self.lossvalue = loss.values().numpy()[0]
        self.optimizer.step()
    
    def train(self, epochs):
        util = DataUtil("OTB50")
        util.set_batchSize(5)
        for i in range(epochs):
            for j in range(6):
                ques = []
                img_paths = []
                for t in range(util.batchsize):
                    ques.append(limited_queue(5))
                    current_series_num = j*util.batchsize + t
                    series_name = util.seriesNames[current_series_num]
                    img_path = os.path.join(series_name, "img")
                    img_paths.append(os.listdir(img_path))
                for k in range(1, 200):
                    for b in range(util.batchsize):
                        template = cv2.imread(img_paths[b][k-1])
                        search = cv2.imread(img_paths[b][k])
                        [x, y, w, h] = util.allLines[j*util.batchsize + b][k-1]
                        template_region = util.getTemplateRegion(template, (x, y, w, h))
                        template_region = np.reshape(template_region, (1, 3, 128, 128))
                        template_region = self.fnet.forward(template_region)
                        ques[b].insert(template_region)
                        search_region = util.getTemplateRegion(search, (x, y, w, h))
                        search_region = np.reshape(search_region, (1, 3, 256, 256))
                        search_region = self.fnet.forward(search_region)
                        targets = util.getTargetBatch(j, k)
                        self.train_step(ques[b].aslist(), search_region, targets)
                        print("Epoch: %d step: %d loss %f" %(i, k*util.batchsize + b, self.lossvalue))
                    self.saveParameter("/model/bnet.path")
    
    def saveParameter(self, path=""):
        torch.save(self.bnet.state_dict(), path)

def test():
    # branch = AlexNet()
    # train_m = trainer(branch, "../OTB50/")
    
    # target = [[7, 8], [5, 6], [9, 2], [10, 10], [9, 7]]
    # x1 = np.random.random((1, 128, 6, 6))
    # X = []
    # for i in range(5):
    #     X.append(torch.from_numpy(x1).float())
    # searchRegion = np.random.random((1, 128, 22, 22))
    # searchRegion = torch.from_numpy(searchRegion).float()
    # train_m.train_step(X, searchRegion, target)
    q = limited_queue(4)
    for i in range(5):
        q.insert(i)
    print(q.aslist())
    print(q.aslist())

if __name__ == '__main__':
    test()








