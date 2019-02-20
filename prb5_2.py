#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:15:34 2018

@author: sumi
"""


import numpy as np
import pylab as plt
from PIL import Image
import time

source_im = np.array(Image.open('/Users/sumi/python/research/messi5.jpg'), dtype=np.uint8)
#source_im = np.array(Image.open('/Users/sumi/python/computer_vision/lab3/eiffel-tower.jpg'), dtype=np.uint8)
plt.figure(1)
plt.imshow(source_im)
plt.show()

K=2
p_all = []
for i in range(0, 2*K, 2) :
    print("Click source and destination of warp point")
    p = np.asarray(plt.ginput(n=2), dtype=np.float32)
    p_all.append(p)
    print(p)
    print(p[0]-p[1])
    plt.plot(p[:,0], p[:,1], color="blue")
    plt.plot(p[0][0], p[0][1],marker='x', markersize=3, color="red")
    plt.plot(p[1][0], p[1][1],marker='x', markersize=3, color="red")
    plt.show()

p_arr = np.array(p_all)
p = np.concatenate(p_arr, axis =0)

rows = np.arange(2*K)
rows_odd = rows%2==1
p_rows_odd = p[rows_odd]

start = time.time()

#Generate pixels coordinates in the destination image       
dest_im = np.zeros(source_im.shape, dtype=np.uint8)                 
max_row = source_im.shape[0]-1
max_col = source_im.shape[1]-1
dest_rows = dest_im.shape[0]
dest_cols = dest_im.shape[1]

#Painting outline of source image black, so out of bounds pixels can be painted black  
source_im[0]=0
source_im[max_row]=0         
source_im[:,0]=0
source_im[:,max_col]=0 
         
#Generate pixel coordinates in the destination image         
ind = np.arange(dest_rows*dest_cols )
row_vect = ind//dest_cols
col_vect = ind%dest_cols
coords = np.vstack((row_vect,col_vect))

#Computing pixel weights, pixels close to p[1] will have higher weights 
dist_ = np.zeros((coords.shape[1], K))
weight_ = np.zeros((coords.shape[1], K))
weight_avg = np.zeros((coords.shape[1], K))

for i in range(K) :
    #dist_[:,i] = np.sqrt(np.square(p[rows_odd][i,1] - row_vect) + np.square(p[rows_odd][i,0] - col_vect))
    dist_[:,i] = np.sqrt(np.square(p[2*i+1][1] - row_vect) + np.square(p[2*i+1][0] - col_vect))
    weight_avg[:,i] = np.exp(-dist_[:,i]/100)
    #weight_[:,i] = 1/dist_[:,i]
    weight_avg[:,i] = (weight_avg[:,i]/np.sum(weight_avg[:,i]))
    
    
    #weight = np.exp(-dist/100)
    #weight = (1/dist)/(1/dist).sum(axis=0)
    #weight = 1/dist/1/np.sum(dist,axis=0)
    #weight = np.exp(-dist/100)
    #weight_o = (1/dist)/np.sum(dist,axis=0)
    #weight = weight_o/np.sum(weight_o)
    
    #weight_o = np.exp(-dist/100)         #Constant needs to be tweaked depending on image size
    #weight = weight_o/np.sum(weight_o)
    
    
    #Computing pixel weights, pixels close to p[1] will have higher weights  
for i in range(K) :
    source_coords = np.zeros(coords.shape, dtype=np.int)
    disp_r = (weight_avg[:,i]*(p[2*i][1]-p[2*i+1][1])).astype(int)
    disp_c = (weight_avg[:,i]*(p[2*i][0]-p[2*i+1][0])).astype(int)
    source_coords[0] = coords[0] + disp_r
    source_coords[1] = coords[1] + disp_c
                 
    #Fixing out-of-bounds coordinates               
    source_coords[source_coords<0] = 0
    source_coords[0,source_coords[0]>max_row] = max_row             
    source_coords[1,source_coords[1]>max_col] = max_col      
          
    dest_im = source_im[source_coords[0],source_coords[1],:].reshape(dest_rows,dest_cols,3)
    
    plt.figure(i+2)
    plt.imshow(dest_im)
    plt.show()
    source_im = dest_im

elapsed_time = time.time()-start
print('Elapsed time: {0:.2f} '.format(elapsed_time))   
