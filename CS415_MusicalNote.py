import numpy as np
import pathlib
import cv2
import math
import sys
import random
import time
import bisect

from FolderHandler import FolderHandler
from CenteredImage import centeredImage


def main():
    # ********************
    # Image Classification
    print("begin project setup\n")
    setupStart = time.time()
    # ********************

    # prepare workspace
    # workspace contains all the paths to training and validation sets
    wrkSpace1 = FolderHandler()

    # get user input
    f = wrkSpace1.returnFile("input.txt")
    fileArgs = []
    try:
        for line in f:
            fileArgs.append(line)
        f.close()
    except:
        print("ERROR: problem reading in file names")
        return

    ### extract info
    # get image
    fileArgs.pop(0)  # remove instruction line
    imgName = fileArgs.pop(0).rstrip()
    imgNameNoExtension = imgName.rsplit('.')[0]
    testImg = cv2.imread(str(wrkSpace1.srcImgPath.joinpath(imgName)))

    # parameters for gaussian
    gaussianKernel = int(fileArgs.pop(0))
    # parameters for thresholding
    thresh = int(fileArgs.pop(0).rstrip())
    maxVal = int(fileArgs.pop(0).rstrip())
    # hough
    houghVotes = int(fileArgs.pop(0))
    houghLines = int(fileArgs.pop(0))

    print(f"***** Execution Info ******\n"
          f"Playing image: {imgName}\n"
          f"Gaussian Kernel Info: {gaussianKernel}\n"
          f"Threshold Info \n   thresh: {thresh} | maxVal: {maxVal}\n"
          f"Hough Info \n   Vote threshold: {houghVotes} | Looking for {houghLines} lines\n")

    # ********************
    # Preprocess Image
    print("1. Preprocess Image\n")
    preprocessStart = time.time()
    # ********************

    # convert 3 channels to 1 channel
    testImgBW = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)

    # apply smoothing
    testImgBW = cv2.GaussianBlur(testImgBW, (gaussianKernel, gaussianKernel), 0)

    # adaptive threshold
    # testImgBWAdap = cv2.adaptiveThreshold(testImgBW, maxValue=threshMaxVal, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                       thresholdType=cv2.THRESH_BINARY, blockSize=threshBlockSize, C=2)

    ret, testImgBWAdap = cv2.threshold(testImgBW, thresh, maxVal, cv2.THRESH_BINARY)

    # write intermediate image
    wrkSpace1.writeImageToOutput(testImgBWAdap, imgNameNoExtension + "_adaptiveThreshold" + ".png")

    # ********************
    # Hough Lines Detection
    print("2. Line Detection\n")
    cannyStart = time.time()
    # ********************

    # canny edge detection
    imgCanny = cv2.Canny(testImgBWAdap, 100, 200, apertureSize=7)

    # write intermediate image
    wrkSpace1.writeImageToOutput(imgCanny, imgNameNoExtension + "_cannyEdge" + ".png")

    # hough transform
    lines = cv2.HoughLines(imgCanny, rho=1, theta=np.pi / 180, threshold=houghVotes)

    while lines is None or len(lines) < houghLines:
        houghVotes = houghVotes - 20
        print(f"lower threshold to {houghVotes}")
        lines = cv2.HoughLines(imgCanny, rho=1, theta=np.pi / 180, threshold=houghVotes)
        if houghVotes < 50:
            print("ERROR: could not find any lines")
            break

    for i in range(len(lines)):
        for rho, theta in lines[i]:
            print(rho, theta)

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(imgCanny, (x1, y1), (x2, y2), 122, 2)

    # write initial lined image
    wrkSpace1.writeImageToOutput(imgCanny, imgNameNoExtension + "_houghInit" + ".png")


main()
