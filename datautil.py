from got10k.datasets import GOT10k
from got10k.utils.viz import show_frame
import random
import cv2
import create_label
import torch
from torch.autograd import Variable
import numpy as np
import os

'''
We get the batch randomly, and there are T frame distance between the example and the search region
T: T frame distance, or it will be out of range
'''

class datasetutil:
    def __init__(self, T, Radius):
        self.dataset = GOT10k(root_dir='../GOT10K', subset='train')
        self.frame_distance = T
        self.frame_num = len(self.dataset)
        self.raduis = Radius
    def get_next_batch(self):
        example_region_array = []
        search_region_array = []
        result_label_array = []
        # Get xlabel randomly
        for i in range(8):
            current_sequence_pos = random.randint(0, self.frame_num - 1)
            sequence = self.dataset[current_sequence_pos]
            example_frame_pos = random.randint(0, len(sequence[0]) - self.frame_distance -1)
            search_frame_pos = example_frame_pos + self.frame_distance
            # Deal with example frame 
            example_frame = cv2.imread(sequence[0][example_frame_pos])
            example_coor = sequence[1][example_frame_pos]
            example_region = create_label.create_example_x_label(example_coor, example_frame)
            example_region_array.append(example_region)
            # get the region of search frame 
            search_frame = cv2.imread(sequence[0][search_frame_pos])
            search_coor = sequence[1][search_frame_pos]
            search_region = create_label.create_search_x_label(example_coor, search_frame)
            search_region_array.append(search_region)
            # get the final result label
            result_label = create_label.create_y_label(example_coor, search_coor, self.raduis)
            result_label_array.append([result_label])
        
        example_region_array = torch.from_numpy(np.array(example_region_array).transpose(0,3,1,2)).float()
        search_region_array = torch.from_numpy(np.array(search_region_array).transpose(0,3,1,2)).float()
        result_label_array = torch.from_numpy(np.array(result_label_array)).float()

        return example_region_array, search_region_array, result_label_array 

# def util_test():
#     my_util = datasetutil(8)
#     x1,x2,y = my_util.get_next_batch()
#     print(x1.size())
#     print(x2.size())
#     print(y.size())


def test_got10k():
    dataset = GOT10k(root_dir='../GOT10K', subset='train')
    for i in range(len(dataset)):
        try:
            filename = dataset[i]
            for j in filename[0]:
                if os.path.exists(j) == False:
                    print("Error:{}".format(i))
                    break
        except Exception:
            print(i)

if __name__ == '__main__':
    test_got10k()





# Test part of the rectangle
    # [x, y, w, h] = example_coor
    # px = int(5/2*w + h)
    # py = int(5/2*h + w)
    # p = int((w+h)/4)
    # cv2.rectangle(example_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0),thickness=10)
    # cv2.rectangle(example_frame, (int(x-px), int(y-py)), (int(x+w+px), int(y+h+py)), (255, 0, 200),thickness=10)
    # cv2.rectangle(example_frame, (int(x-p), int(y-p)), (int(x+w+p), int(y+h+p)), (0, 255, 200), thickness=10)
    
    # cv2.rectangle(search_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), thickness=10)
    # cv2.rectangle(search_frame, (int(x-px), int(y-py)), (int(x+w+px), int(y+h+py)), (255, 0, 200), thickness=10)
    # cv2.rectangle(search_frame, (int(x-p), int(y-p)), (int(x+w+p), int(y+h+p)), (0, 255, 200),thickness=10)
    
    # search_region = cv2.resize(search_region, (int(search_region.shape[1]/5), int(search_region.shape[0]/5)), interpolation=cv2.INTER_CUBIC)
    # search_frame = cv2.resize(search_frame, (int(search_frame.shape[1]/5), int(search_frame.shape[0]/5)), interpolation=cv2.INTER_CUBIC)
    # example_frame = cv2.resize(example_frame, (int(example_frame.shape[1]/5), int(example_frame.shape[0]/5)), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('search', search_region)
    # cv2.imshow('example', search_frame)
    # cv2.imshow('mat', example_frame)
    # cv2.waitKey(0)
    # generate labels from 

    # cv2.imshow('example_region', example_region)
    # cv2.imshow('search_region', search_region)
    # coorx = 0
    # coory = 0
    # nums = 0
    # for i in range(17):
    #     for j in range(17):
    #         if result_label[i][j] == 1:
    #             coorx += i
    #             coory += j
    #             nums += 1
    # print(coorx/nums)
    # print(coory/nums)    

    # cv2.waitKey(0)
