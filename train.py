'''
include: train, load parameters, save
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

from torch.autograd import Variable
import torch.nn as nn
import cv2
import datautil
from siamesenet import Siamesenet
from siamesenet import alexnetpart
from datautil import datasetutil
import torch
import torch.optim as optim

class trainer:
    def __init__(self, T, radius):
        alex = alexnetpart()
        self.net = Siamesenet(branch=alex)
        self.dataset = datasetutil(T, radius)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.loss_value = 0
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.net.cuda()

    def train_single_step(self):
        example_region_array, search_region_array, result_label_array = self.dataset.get_next_batch()
        if self.use_gpu:
            example_region_array = example_region_array.cuda()
            search_region_array = search_region_array.cuda()
            result_label_array = result_label_array.cuda()
        output = self.net.forward(example_region_array, search_region_array)
        loss = self.calc_loss(output, result_label_array)
        self.loss_value = loss.item()
        self.update_parameters(loss)

    def calc_loss(self, output, target):
        loss = torch.mean(torch.log(1+torch.exp(output.mul(-1*target))))
        return loss
    
    def update_parameters(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_parameters(self, file_path):
        checkpoint = torch.load('./model/checkpoint.pth.tar')
        self.net.load_state_dict(checkpoint['state_dict'])
    
    def save_parameters(self, file_path):
        torch.save({'state_dict': self.net.state_dict()}, file_path)
    
    def multi_step_train(self, epochs):
        for i in range(epochs):
            for j in range(50000):
                self.train_single_step()
                if j % 1000 == 0:
                    print('Epoch:{} Step:{} Average loss: {:4f}'.format(i, j, self.loss_value))
                    self.save_parameters('./model/checkpoint.pth.tar')

def test_trainer():
    m_trainer = trainer(8, 4)
    m_trainer.multi_step_train(50)

if __name__ == '__main__':
    test_trainer()
