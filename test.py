from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
from demo_tracker import demotracker
import numpy as np
import PIL
import cv2
from got10k.datasets import GOT10k
import torch
import os
import time
import psutil
# os.environ["CUDA_VISIBLE_DEVICES"]='5'

class IdentityTracker():
    def __init__(self):
        self.tracker = demotracker()
        # self.KCFtracker = cv2.TrackerKCF_create()
    
    def init(self, image, box):
        self.box = box
        origin_coor = box.tolist()
        # [x, y, w, h] = origin_coor
        # origin_coor = (x, y, w, h)
        self.tracker.select_first_region(image, origin_coor)
        # self.KCFtracker.init(image, origin_coor)

    def update(self, image):
        track_result = self.tracker.predict_future_coor(image)
        # KCF_result = self.KCFtracker.update(image)
        return np.array(track_result)

def getcurrent_time():
    return time.time()

def calc_Hz(time1, time2, frame_num):
    return frame_num / (time2 - time1) 

def get_Current_memory_percent():
    return psutil.virtual_memory().percent

def get_Current_cpu_percent():
    return psutil.cpu_percent(0)

def get_data_from_one_series(num):
    imgs = []
    dataset = GOT10k(root_dir='../GOT-10k', subset='test')
    img_file, anno = dataset[num]
    for i in range(10):
        img = cv2.imread(img_file[i])
        imgs.append(img)
    return imgs, anno[0, :]

def AnalysisEuismod():
    before_cpu = get_Current_cpu_percent()
    before_memory = get_Current_memory_percent()
    after_cpu = 0
    after_memory = 0
    trackerana = IdentityTracker()
    imgs, rect = get_data_from_one_series(10)
    trackerana.init(imgs[0], rect)
    after_memory = get_Current_memory_percent()
    before_time = getcurrent_time()
    for i in range(1, 10):
        trackerana.update(imgs[i])
        if i == 5:
            after_cpu = get_Current_cpu_percent()
    after_time = getcurrent_time()
    print("Hz:%f"%(10/(after_time-before_time)))
    print("cpu_occupy:%f"%(after_cpu-before_cpu))
    print("memory_occupy:%f"%(after_memory-before_memory))

if __name__ == '__main__':
    AnalysisEuismod()
