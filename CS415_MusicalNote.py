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
    #get image
    fileArgs.pop(0)  # remove instruction line
    imgName = fileArgs.pop(0).rstrip()
    imgNameNoExtension = imgName.rsplit('.')[0]
    testImg = cv2.imread(str(wrkSpace1.srcImgPath.joinpath(imgName)))

    #parameters for thresholding
    threshMaxVal = int(fileArgs.pop(0).rstrip())
    threshBlockSize = int(fileArgs.pop(0).rstrip())

    print(f"***** Execution Info ******\n"
          f"Playing image: {imgName}\n"
          f"Threshold Info \n   maxVal: {threshMaxVal} | blockSize: {threshBlockSize}\n")


    # ********************
    # Preprocess Image
    print("1. Preprocess Image\n")
    preprocessStart = time.time()
    # ********************

    #convert 3 channels to 1 channel
    testImgBW = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)

    #adaptive threshold
    testImgBWAdap = cv2.adaptiveThreshold(testImgBW, maxValue=threshMaxVal, adaptiveMethod= cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType= cv2.THRESH_BINARY, blockSize=threshBlockSize,C=2)
    print(type(testImgBW))

    #write intermediate image
    wrkSpace1.writeImageToOutput(testImgBWAdap, imgNameNoExtension + "_adaptiveThreshold" + ".png")

    # ********************
    # Hough Lines Detection
    print("2. Line Detection\n")
    cannyStart = time.time()
    # ********************

    testCentered = centeredImage(testImgBWAdap)

    print(testImgBWAdap.shape)

    testCentered.drawGrid()

    allPixels = testCentered.returnEveryCoordinate()

    print(len(allPixels))

    wrkSpace1.writeImageToOutput(testCentered.image, imgNameNoExtension + "_testDraw" + ".png")

    # #canny edge detection
    # imgCanny = cv2.Canny(testImgBWAdap, 50, 100, apertureSize=5)
    #
    # # write intermediate image
    # wrkSpace1.writeImageToOutput(imgCanny, imgNameNoExtension + "_cannyEdge" + ".png")
    #
    # # hough transform
    # lines = cv2.HoughLines(imgCanny, rho=1000, theta=np.pi/1000, threshold=30000)
    #
    # print(lines)
    # print(len(lines))
    #
    # #print lines on image
    # testImgCopy = testImg.copy()
    # for rho, theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #
    #     cv2.line(testImgCopy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # #write initial lined image
    # wrkSpace1.writeImageToOutput(testImgCopy, imgNameNoExtension + "_houghInit" + ".png")



main()