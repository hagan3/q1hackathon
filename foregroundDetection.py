from __future__ import print_function
import cv2 as cv
from os import listdir
from os.path import isfile, join
import numpy


# local images folder
mypath='./images/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
im_gray = numpy.empty(len(onlyfiles), dtype=object)
mask = numpy.empty(len(onlyfiles), dtype=object)
im_thresh_gray = numpy.empty(len(onlyfiles), dtype=object)

backSub = cv.createBackgroundSubtractorKNN()

for n in range(0, len(onlyfiles)):
    images[n] = cv.imread( join(mypath,onlyfiles[n]) )
    im_gray[n] = cv.cvtColor(images[n], cv.COLOR_BGR2GRAY)
    _, mask[n] = cv.threshold(im_gray[n], thresh=180, maxval=255, type=cv.THRESH_BINARY)
    im_thresh_gray[n] = cv.bitwise_and(im_gray[n], mask[n])

    while True:
        cv.imshow('Frame', images[n])
        cv.imshow('FG Mask', im_thresh_gray[n])
        keyboard = cv.waitKey(0)
        if keyboard == 'q' or keyboard == 27:
            break





