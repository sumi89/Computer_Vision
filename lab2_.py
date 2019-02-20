#Apendix:

import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt 
from scipy import interpolate 

def eqHist(original):
    image = original.copy()

    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])

    return image

def f2d(original, kernel):
    if kernel is 'box':
        kernel = 'boxfilter'
    if kernel is 'blur': 
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    if kernel is 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filt = cv2.filter2D(original, -1, kernel)
    return filt


def display(image):
    while True:
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def write(image, name, mod):
    cv2.imwrite(mod+"/" + mod + "_" + name, image)

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
def plotHist(image, imageName):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title(imageName)
    plt.savefig('filt_hist/filt-hist_' + imageName)

def interp(image):
    new = np.zeros(((image.shape[0]*2) - 1, (image.shape[1] * 2) - 1, image.shape[2]))
    new[::2,::2] = image[:,:,:]
    new[1::2] = (new[:-1:2] + new[2::2])/2
    new[:, 1::2] = (new[:, :-1:2] + new[:, 2::2]) / 2


def medFilt(image):
    return(cv2.medianBlur(image,3))

elapsed_times = []
for image in os.listdir("lab2"):
    im = cv2.imread('lab2/' + image)

    #time
    start_time = time.time()

    #histogram equalization
    #histim = eqHist(im)
    #plotHist(histim, image)
    #write(histim, image, 'histeq')

    # sharpen or blur
    #f2dim = f2d(im, 'sharpen')
    #f2dim = f2d(im, 'box')
    #f2dim = f2d(im, 'blur')
    #write(f2dim, image, 'sharpen')
    #write(f2dim, image, 'blur')

    #medF = medFilt(im)
    #medF = medFilt(im)

    #display(im)
    #display(histim)
    #display(f2dim)
    #display(eqHist(f2dim))
    #display(medF)

    #write(medF, image, 'medFilter')
    #write(f2dim, image, 'sharp_blur')

    # plot original images to histogram
    #plotHist(im, image)

    interp(im)
    #end time
    elapsed_times.append(time.time() - start_time)

print(np.mean(elapsed_times))
