from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
from demo_tracker import demotracker
import numpy as np
import PIL
import cv2

import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"]='5'

class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker')
        self.tracker = demotracker()
    
    def init(self, image, box):
        self.box = box
        origin_coor = box.tolist()
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        self.tracker.select_first_region(frame, origin_coor)

    def update(self, image):
        try:
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            track_result = self.tracker.predict_future_coor(frame)
            return np.array(track_result)
        except Exception:
            return np.array(self.tracker.current_coor)

if __name__ == '__main__':
    # setup tracker
    tracker = IdentityTracker()

    # setup experiment (validation subset)
    experiment = ExperimentGOT10k(
        root_dir='../GOT10K',    # GOT-10k's root directory
        subset='val',               # 'train' | 'val' | 'test'
        result_dir='results',       # where to store tracking results
        report_dir='reports'        # where to store evaluation reports
    )

    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])