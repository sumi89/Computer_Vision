
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:04:27 2018

@author: sumi
"""

import numpy as np
from PIL import Image
import pylab as plt
import time

def transform(H,fp):
    # Transforming point fp according to H
    # Convert to homogeneous coordinates if necessary
    if fp.shape[0]==2:
          t = np.dot(H,np.vstack((fp,np.ones(fp.shape[1]))))
    else:
        t = np.dot(H,fp)
    return t[:2]
    
im2 = np.array(Image.open('/Users/sumi/python/computer_vision/lab3/banner_small.jpg'), dtype=np.uint8)
plt.figure(1)
plt.imshow(im2)
plt.show()

source_im = np.array(Image.open('/Users/sumi/python/computer_vision/lab3/tennis.jpg'), dtype=np.uint8)
plt.figure(2)
plt.imshow(source_im)
plt.show()

##### why x and y are in this arrangement ? Ans : may be bcs of pylab library
x_u = [0,0,im2.shape[0]-1]
y_u = [0,im2.shape[1]-1,im2.shape[1]-1]
fp_u = np.vstack((x_u,y_u))

x_l = [im2.shape[0]-1,im2.shape[0]-1,0]
y_l = [im2.shape[1]-1,0,0]
fp_l = np.vstack((x_l,y_l))

print("Click destination points, top-left, top_right, bottom-left and bottom-right corners")
tp_ = np.asarray(plt.ginput(n=4), dtype=np.float).T
tp = tp_[[1,0],:]
tp_u = np.vstack((tp[:,0], tp[:,1], tp[:,2])).T
tp_l = np.vstack((tp[:,2], tp[:,3], tp[:,0])).T

#tp_c = np.concatenate((tp[:,0],tp[:,2]), axis=1)
#tp_l = tp
print('fp_l',fp_l)
print('fp_u',fp_u)
print('tp_l',tp_l)
print('tp_u',tp_u)
start = time.time()
#Using pseudoinverse
# Generating homogeneous coordinates
fph_l = np.vstack((fp_l,np.ones(fp_l.shape[1])))
fph_u = np.vstack((fp_u,np.ones(fp_u.shape[1])))
tph_l = np.vstack((tp_l,np.ones(tp_l.shape[1])))
tph_u = np.vstack((tp_u,np.ones(tp_u.shape[1])))
H_l = np.dot(tph_l,np.linalg.pinv(fph_l))
H_u = np.dot(tph_u,np.linalg.pinv(fph_u))

print((transform(H_l,fp_l)+.5).astype(np.int))
print((transform(H_u,fp_u)+.5).astype(np.int))

#Generating pixel coordinate locations
ind = np.arange(im2.shape[0]*im2.shape[1])
row_vect = ind//im2.shape[1]
col_vect = ind%im2.shape[1]
coords = np.vstack((row_vect,col_vect))

k = (im2.shape[1] - 1)/(im2.shape[0] - 1)
cols = coords[1] - k * (coords[0]) < 0
coords_l = coords[:, cols]
coords_u = coords[:, ~cols]

coords_l_n = transform(H_l, coords_l).astype(np.int)
coords_u_n = transform(H_u, coords_u).astype(np.int)
target_im = source_im
target_im[coords_l_n[0], coords_l_n[1], :] = im2[coords_l[0], coords_l[1], :]
target_im[coords_u_n[0], coords_u_n[1], :] = im2[coords_u[0], coords_u[1], :]


plt.figure(3)
plt.imshow(target_im)
plt.show()

elapsed_time = time.time()-start
print('Elapsed time: {0:.2f} '.format(elapsed_time)) 