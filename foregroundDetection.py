from __future__ import print_function
import cv2 as cv
from os import listdir
from os.path import isfile, join
import numpy


# local images folder
mypath='./images/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)


backSub = cv.createBackgroundSubtractorKNN()



for n in range(0, len(onlyfiles)):
    images[n] = cv.imread( join(mypath,onlyfiles[n]) )
    fgMask = backSub.apply(images[n])

    while True:
        cv.imshow('Frame', images[n])
        cv.imshow('FG Mask', fgMask)
        keyboard = cv.waitKey(0)
        if keyboard == 27:
            break





