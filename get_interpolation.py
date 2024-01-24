import cv2
import numpy as np
import os
import sys

# file_root = "./CREMI_A"
# file_root = "./CREMI_B"
file_root = "./CREMI_C"

Flist_path = file_root + '/train.flist'
data = [os.path.join(file_root, i) 
            for i in np.genfromtxt(Flist_path, dtype=np.str_, encoding='utf-8')]
data.sort()

def make_prediction( reference_image, flow):
    
        height, width = flow.shape[:2]
        map_x = np.tile(np.arange(width), (height, 1))
        map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
        map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
        return cv2.remap(src = reference_image,
                        map1 = map_xy,
                        map2 = None,
                        interpolation = cv2.INTER_LINEAR,
                        borderMode = cv2.BORDER_CONSTANT)
print(len(data))
from tqdm import tqdm
for index in tqdm(range(len(data))):
    img_path = os.path.dirname(data[index]) +'/'
    img1_name = 'img'+str(index).zfill(3)+'_1.png'
    img2_name = 'img'+str(index).zfill(3)+'_2.png'
    img3_name = 'img'+str(index).zfill(3)+'_3.png'
    print(img1_name)
    print(img_path+img1_name)
    img_1 = cv2.imread(img_path+img1_name, cv2.IMREAD_UNCHANGED)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY) 
    img_2 = cv2.imread(img_path+img2_name, cv2.IMREAD_UNCHANGED)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY) 
    img_3 = cv2.imread(img_path+img3_name, cv2.IMREAD_UNCHANGED)
    img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY) 
    forward_flow = np.zeros((img_1.shape[0], img_1.shape[1], 2), dtype=np.float32)
    backward_flow = np.zeros((img_1.shape[0], img_1.shape[1], 2), dtype=np.float32)
    forward_flow = cv2.calcOpticalFlowFarneback( \
            prev=img_3, next=img_1, flow=forward_flow, pyr_scale=0.5, levels = 3, winsize = 33, \
            iterations = 3, poly_n = 5, poly_sigma = 1.2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW | \
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    backward_flow = cv2.calcOpticalFlowFarneback( \
            prev=img_1, next=img_3, flow=backward_flow, pyr_scale=0.5, levels = 3, winsize = 33, \
            iterations = 3, poly_n = 5, poly_sigma = 1.2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW | \
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    prev_prediction = make_prediction(reference_image = img_1, flow = forward_flow * 0.5)
    next_prediction = make_prediction(reference_image = img_3, flow = backward_flow * 0.5)
    interpolation_image = \
            (prev_prediction * 0.5 + \
                next_prediction * 0.5).astype(np.uint8)
    inter_name = 'img'+str(index).zfill(3)+'inter_2.png'
    cv2.imwrite(img_path+inter_name,interpolation_image)
#     sys.exit()