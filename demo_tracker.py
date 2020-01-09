from siamesenet import Siamesenet
from siamesenet import alexnetpart
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import os, sys
import create_label
from datautil import datasetutil
import numpy as np

class demotracker:
    def __init__(self):
        alexpart = alexnetpart()
        self.net = Siamesenet(alexpart)
        self.load_parameters()

    def load_parameters(self):
        checkpoint = torch.load('./model/checkpoint.pth.tar', map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['state_dict'])

    def select_first_region(self, example_frame, origin_coor):
        example_label = create_label.create_example_x_label(origin_coor, example_frame)
        self.current_coor = origin_coor
        self.example_label = torch.from_numpy(np.array([example_label]).transpose(0,3,1,2)).float()

    def predict_future_coor(self, search_frame):
        '''
        predict target in the future frame
        input:
        example: first frame in video
        search_region: The frame choosen to track
        before_coor: coordinate in previous frame 
        '''
        max_sum_up = -1000000
        final_rate = 1
        final_max_label_loc = [0, 0] 
        for i in range(-2, 2):
            rate = pow(1.025, i)
            search_label = create_label.scaled_search_region(self.current_coor, search_frame, rate)
            search_label = torch.from_numpy(np.array([search_label]).transpose(0,3,1,2)).float()
            result = self.net.forward(self.example_label, search_label)
            ans = result.detach().numpy()[0,0,:,:]
            xs = 0
            xnum = 0
            ys = 0
            ynum = 0
            tarray = ans.reshape((-1))
            tarray = np.sort(tarray)
            limit_range = tarray[263]
            avg_limit = np.mean(tarray[263:])

            for i in range(17):
                for j in range(17):
                    if ans[i, j] > limit_range:
                        ys = ys + i
                        xs = xs + j
                        xnum = xnum + 1
                        ynum = ynum + 1
            xend = xs / xnum
            yend = ys / ynum
            max_label_loc =  [xend, yend]
            # print(max_label_loc)
            if avg_limit > max_sum_up:
                max_sum_up = avg_limit
                final_rate = rate
                final_max_label_loc = max_label_loc
        self.current_coor = create_label.get_bouding_box(max_label_loc, self.current_coor, final_rate)
        return self.current_coor

if __name__ == '__main__':
    tracker = demotracker()
    util = datasetutil(8, 4)
    path_sequence, coor_sequence = util.get_one_train_sequence(800)
    example_frame = cv2.imread(path_sequence[0])
    tracker.select_first_region(example_frame, coor_sequence[0])

    videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('I','4','2','0'), 10, (500, 300))

    for i in range(1, 60):
        search_frame = cv2.imread(path_sequence[i])
        [xc, yc, wc, hc] = tracker.predict_future_coor(search_frame)

        [x, y, w, h] = coor_sequence[0]
        [x1, y1, w1, h1] = coor_sequence[i]

        # cv2.rectangle(example_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 255), thickness=5)
        # cv2.rectangle(search_frame, (int(x1), int(y1)), (int(x1+w1), int(y1+h1)), (255, 255, 0), thickness=5)
        cv2.rectangle(search_frame, (int(xc), int(yc)), (int(xc+wc), int(yc+hc)), (255, 0, 0), thickness=3)

        # cv2.imshow('example_frame', cv2.resize(example_frame, (500, 300)))
        videoWriter.write(cv2.resize(search_frame, (500, 300)))
        cv2.imshow('search_frame', cv2.resize(search_frame, (500, 300)))

        cv2.waitKey(10)
    
    videoWriter.release()

