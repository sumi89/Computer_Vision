#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:42:04 2018

@author: sumi
"""


import numpy as np
from PIL import Image
import pylab as plt
from keras.preprocessing import image

def transform(H,fp):
    # Transforming point fp according to H
    # Convert to homogeneous coordinates if necessary
    if fp.shape[0]==2:
          t = np.dot(H,np.vstack((fp,np.ones(fp.shape[1]))))
    else:
        t = np.dot(H,fp)
    return t[:2]
    
im2 = np.array(Image.open('/Users/sumi/python/computer_vision/lab3/cat.jpg'), dtype=np.uint8)
plt.figure(1)
plt.imshow(im2)
plt.show()


print("Click eyes and nose")
fp_ = np.asarray(plt.ginput(n=3), dtype=np.float).T
fp = fp_[[1,0],:]
print(fp)

source_im = np.array(Image.open('/Users/sumi/python/computer_vision/lab3/man.jpg'), dtype=np.uint8)
source_im=source_im[98:370, 30:430, :]
plt.figure(2)
plt.imshow(source_im)
plt.show()


print("Click eyes and nose")
tp_ = np.asarray(plt.ginput(n=3), dtype=np.float).T
tp = tp_[[1,0],:]
print(tp)



#Using pseudoinverse
# Generating homogeneous coordinates
fph = np.vstack((fp,np.ones(fp.shape[1])))
tph = np.vstack((tp,np.ones(tp.shape[1])))
H = np.dot(tph,np.linalg.pinv(fph))

print((transform(H,fp)+.5).astype(np.int))

#Generating pixel coordinate locations
ind = np.arange(im2.shape[0]*im2.shape[1])
row_vect = ind//im2.shape[1]
col_vect = ind%im2.shape[1]
coords = np.vstack((row_vect,col_vect))

new_coords = transform(H,coords).astype(np.int)
target_im = source_im
target_im[new_coords[0],new_coords[1],:] = im2[coords[0],coords[1],:]

faceMorph = ((target_im * .5) + (source_im * 0.5)).astype(np.uint8)

plt.figure(3)
plt.imshow(faceMorph)
plt.show()
