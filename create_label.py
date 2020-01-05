import cv2
import numpy as np

# Generate xlabel from the example image
def create_example_x_label(current_coor, example_region):
    [x, y, w, h] = current_coor
    p = int((w + h) / 4)
    # get the region of the cut rect in origion example region
    disx, disy = 0, 0
    x1 = int(x - p)
    x2 = int(x + w + p)
    y1 = int(y - p)
    y2 = int(y + h + p)
    if x2 > example_region.shape[1]:
        x2 = example_region.shape[1]
    if y2 > example_region.shape[0]:
        y2 = example_region.shape[0]
    if x1 < 0:
        disx = abs(x1)
        x1 = 0
    if y1 < 0:
        disy = abs(y1)
        y1 = 0
    selected_region = example_region[y1:y2, x1:x2, :]
    mean_val = np.mean(example_region, axis=(0, 1))
    mean_val = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
    selected_region_border = cv2.copyMakeBorder(selected_region, disy, int(h+2*p-y2+y1-disy), disx, \
        int(w+2*p-x2+x1-disx), cv2.BORDER_CONSTANT, value=mean_val)
    # return selected_region_border
    return cv2.resize(selected_region_border, (127, 127), interpolation=cv2.INTER_CUBIC)/255.0

# Generate xlabel from 
def create_search_x_label(origin_coor, search_region):
    [x, y, w, h] = origin_coor
    px = int(3/2*w)
    py = int(3/2*h)
    disx, disy = 0, 0
    x1 = int(x - px)
    x2 = int(x + w + px)
    y1 = int(y - py)
    y2 = int(y + h + py)
    if x2 > search_region.shape[1]:
        x2 = search_region.shape[1]
    if y2 > search_region.shape[0]:
        y2 = search_region.shape[0]
    if x1 < 0:
        disx = abs(x1)
        x1 = 0
    if y1 < 0:
        disy = abs(y1)
        y1 = 0
    selected_region = search_region[y1:y2, x1:x2, :]
    mean_val = np.mean(search_region, axis=(0, 1))
    mean_val = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
    selected_region_border = cv2.copyMakeBorder(selected_region, disy, int(h+2*py-y2+y1-disy), disx, \
        int(w+2*px-x2+x1-disx), cv2.BORDER_CONSTANT, value=mean_val)
    # return selected_region_border
    return cv2.resize(selected_region_border, (255, 255), interpolation=cv2.INTER_CUBIC)/255.0

# Get the ylabel in the training process 
def create_y_label(orgin_coor, current_coor, Radius):
    [x_o, y_o, w_o, h_o] = orgin_coor
    [x_c, y_c, w_c, h_c] = current_coor

    cx_o = x_o + int(w_o/2)
    cy_o = y_o + int(h_o/2)
    cx_c = x_c + int(w_c/2)
    cy_c = y_c + int(h_c/2)
    
    x_in_label = int((cx_c - cx_o)/(3/16*w_o))+8
    y_in_label = int((cy_c - cy_o)/(3/16*h_o))+8
    result_label = np.ones((17, 17))
    
    for i in range(17):
        for j in range(17):
            if (i-x_in_label)*(i-x_in_label) + (j-y_in_label)*(j-y_in_label) < Radius * Radius:
                result_label[i, j] = 1
            else:
                result_label[i, j] = -1
    return result_label

def get_bouding_box(max_value_label_loc, current_loc, rate):
    [x, y, w, h] = current_loc
    [i, j] = max_value_label_loc
    cx = x + w/2
    cy = y + h/2
    x_c = cx + (i-8)*3/16*w*rate
    y_c = cy + (j-8)*3/16*w*rate
    w = int(rate*w)
    h = int(rate*w)
    x_c = int(x_c-rate*w)
    y_c = int(y_c-rate*h)
    return [x_c, y_c, w, h]

def scaled_search_region(origin_coor, search_region, rate):
    [x, y, w, h] = origin_coor
    cx = x + w/2
    cy = y + h/2
    w = rate*w
    h = rate*h
    x = cx - w/2
    y = cy - h/2
    px = int(3/2*w)
    py = int(3/2*h)
    disx, disy = 0, 0
    x1 = int(x - px)
    x2 = int(x + w + px)
    y1 = int(y - py)
    y2 = int(y + h + py)
    if x2 > search_region.shape[1]:
        x2 = search_region.shape[1]
    if y2 > search_region.shape[0]:
        y2 = search_region.shape[0]
    if x1 < 0:
        disx = abs(x1)
        x1 = 0
    if y1 < 0:
        disy = abs(y1)
        y1 = 0
    selected_region = search_region[y1:y2, x1:x2, :]
    mean_val = np.mean(search_region, axis=(0, 1))
    mean_val = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
    selected_region_border = cv2.copyMakeBorder(selected_region, disy, int(h+2*py-y2+y1-disy), disx, \
        int(w+2*px-x2+x1-disx), cv2.BORDER_CONSTANT, value=mean_val)
    # return selected_region_border
    return cv2.resize(selected_region_border, (255, 255), interpolation=cv2.INTER_CUBIC)/255.0