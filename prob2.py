#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:50:05 2018

@author: sumi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 01:44:08 2018

@author: sumi
"""

# Segmantation based on connected components 
# Programmed by Olac Fuentes
# Last modified November 19, 2018

import numpy as np
import cv2
import pylab as plt
import time
import random

def find(i):
    if S[i] <0:
        return i
    s = find(S[i]) 
    S[i] = s #Path compression
    return s


def union(i,j,thr):
    # Joins two regions if they are similar
    # Keeps track of size and mean color of regions
    ri =find(i)
    rj = find(j)
    if (ri!= rj):
        d =  sum_pixels[ri,:]/count[ri] - sum_pixels[rj,:]/count[rj]
        diff = np.sqrt(np.sum(d*d))
        if diff < thr:	
            S[rj] = ri
            count[ri]+=count[rj]
            count[rj]=0
            sum_pixels[ri,:]+=sum_pixels[rj,:]
            sum_pixels[rj,:]=0
                  
def initialize(I):
    rows = I.shape[0]
    cols = I.shape[1]   
    S=np.zeros(rows*cols).astype(np.int)-1
    count=np.ones(rows*cols).astype(np.int)       
    sum_pixels = np.copy(I).reshape(rows*cols,3)      
    return S, count, sum_pixels        

def connected_components_segmentation(I,thr):
    rows = I.shape[0]
    cols = I.shape[1]   
    for p in range(S.shape[0]):
        if p%cols < cols-1:  # If p is not in the last column
            union(p,p+1,thr) # p+1 is the pixel to the right of p  
        if p//cols < rows-1: # If p is not in the last row   
            union(p,p+cols,thr) # p+cols is the pixel to below p  
        if p%cols < cols-1 and p//cols < rows-1:
            union(p,p+cols+1,thr)



I  =  (cv2.imread('/Users/sumi/python/computer_vision/lab5/car.jpg',1))
#I  =  (cv2.imread('/Users/sumi/python/computer_vision/lab5/building1.jpg',1))
#I  =  (cv2.imread('/Users/sumi/python/computer_vision/lab5/car.jpg',1))
#I  =  (cv2.imread('capetown.jpg',1)/255)
#I  =  (cv2.imread('mayon.jpg',1)/255)
#cv2.imwrite('/Users/sumi/python/computer_vision/lab5/Original_nature.jpg',I)

###### change colorspace ######
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I[gray > 200] = 255
I = I/255

###### automatic threshold ######
ret,thresh = cv2.threshold(I,0.15,1,cv2.THRESH_TRUNC)



thr=ret
rows = I.shape[0]
cols = I.shape[1]   
S, count, sum_pixels = initialize(I)
connected_components_segmentation(I,thr)

print('Regions found: ',np.sum(S==-1))
print('Size 1 regions found: ',np.sum(count==1))



rand_cm = np.random.random_sample((rows*cols, 3))
seg_im_mean = np.zeros((rows,cols,3))
seg_im_rand = np.zeros((rows,cols,3))
for r in range(rows-1):
    for c in range(cols-1):
        f = find(r*cols+c)
        seg_im_mean[r,c,:] = sum_pixels[f,:]/count[f]
        seg_im_rand[r,c,:] = rand_cm[f,:]

cv2.imwrite('/Users/sumi/python/computer_vision/lab5/seg_im_mean_car.jpg',seg_im_mean*255)
cv2.imwrite('/Users/sumi/python/computer_vision/lab5/seg_im_rand_car.jpg',seg_im_rand*255) 
           
#cv2.imwrite('Segmentation 1 - using mean colors',seg_im_mean)
#cv2.imwrite('Segmentation 2 - using random colors',seg_im_rand)