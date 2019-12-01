import numpy as np
import pathlib
import cv2
import pytest
import math
import sys

from CenteredImage import centeredImage


class houghLineCustom:

    # A class made to find the hough lines
    #
    #

    def __init__(self, srcImg=np.ones((100, 100)), xQuantization=.01, yQuantization=.01):

        # src image has canny applied to it and has one channel

        # extract info
        # get image
        self.centeredImage = centeredImage(srcImg)
        self.xQuant = xQuantization
        self.yQuant = yQuantization

        # calculate largest rho
        self.rhoCeiling = self.largestRho()

        # build accumulator
        self.accumulator = self.accumulatorSpace()

        # fill accumulator
        self.fillAccumulator()

    def fillAccumulator(self):
        self.transformImagetoAccumulator()
        print(self.accumulator)

        normalizeImage(self.accumulator)

    def transformImagetoAccumulator(self):
        """
        :param imageSpace: img - after canny detection
        :param imageSpaceOffsets: [] - distance to each side of the centered origin
        :param accumulatorSpace: matrix - same dimension as parameter space
        :param parameterSpaceInfo: [] - how big the bins are
        :return: accumulator modified
        """

        # go through each pixel
        everyPixel = self.centeredImage.returnEveryCoordinate()

        for point in everyPixel:
            valOnArray = self.centeredImage.getPixelVal(point[0], point[1])

            if self.centeredImage.image[valOnArray[1]][valOnArray[0]] > 20:
                print("we made it")
                parameterSpaceInfo = []
                paramSpaceFourPoint = self.buildParameterSpace(parameterSpaceInfo)
                self.imagePointToParameterSpace(paramSpaceFourPoint, parameterSpaceInfo, point[0], point[1])
                self.accumulator = np.add(self.accumulator, paramSpaceFourPoint)
            else:
                pass

    def imagePointToParameterSpace(self, parameterSpace, parameterSpaceInfo, xPos, yPos):
        '''
        :param parameterSpace: matrix
        :param parameterSpaceInfo: [] - [bin size for each pixel]
        :param xPos: int - x location on image (centered)
        :param yPos: int - y location on image (centered)
        :return: modify parameterSpace
        '''

        # extract necessary data
        paramSpaceShape = parameterSpace.shape
        numXBins = paramSpaceShape[0]
        numYBins = paramSpaceShape[1]

        xBinSize = parameterSpaceInfo[0]
        yBinSize = parameterSpaceInfo[1]

        # build the list of thetas to be tested
        thetas = [(xBinSize / 2) + j * xBinSize for j in range(numXBins)]

        # for each xBin we will calculate which yBin its associated with
        for xPixel in range(numXBins):
            currTheta = thetas[xPixel]
            currRho = outputRho(xPos, yPos, currTheta)
            yPixel = rhoToIndex(currRho, yBinSize, self.rhoCeiling)
            parameterSpace[yPixel][xPixel] = 1

    def largestRho(self):
        maxDim = math.sqrt(self.centeredImage.offsetRight ** 2 + self.centeredImage.offsetUp ** 2)
        return maxDim

    def buildParameterSpace(self, parameterSpaceInfo=[]):
        '''
        :param rho: int - largest rho possible given src img
        :param alpha: double - 0 to 1 for the values of theta. 0 almost no quantization to 1 fully quantizationed
        :param beta: double - 0 to 1 for values of rho. 0 limited quantization to 1 fully quantized (one pixel)
        :return: the parameter space (a matrix)
        '''

        # determine number of bins and size of each bin
        # xaxis - theta
        dividor = self.xQuant * 2 * math.pi
        numofBinsXAxis = round(2 * math.pi / dividor)
        sizeofBinX = (2 * math.pi) / numofBinsXAxis

        # yaxis - rho
        dividor = self.yQuant * 2 * self.rhoCeiling  # doubled rhoCeiling for negative rho values
        numofBinsYAxis = round(2 * self.rhoCeiling / dividor)
        sizeofBinY = (2 * self.rhoCeiling) / numofBinsYAxis

        parameterSpaceInfo.append(sizeofBinX)
        parameterSpaceInfo.append(sizeofBinY)

        return np.zeros((numofBinsXAxis, numofBinsYAxis))

    def accumulatorSpace(self):
        #
        # Prepare to build parameter space and accumulator
        #

        print("1. Build Parameter Space \n")

        rhoCeiling = self.largestRho()

        parameterSpaceInfo = []

        accumulatorSpace = self.buildParameterSpace(parameterSpaceInfo)

        print("Parameter space info")
        print(f"rhoCeiling: {rhoCeiling} XBinSize: {parameterSpaceInfo[0]} YBinSize: {parameterSpaceInfo[1]}")

        return accumulatorSpace


def findMaxPixel(workingImg=np.zeros((100, 100))):
    workingImgShape = workingImg.shape
    rows = workingImgShape[0]
    cols = workingImgShape[1]

    currMax = 0

    for row in range(rows):
        for col in range(cols):
            currPixel = workingImg[row][col]
            if currPixel > currMax:
                currMax = currPixel

    print(f"The max pixel is: {currMax}")
    return currMax


def normalizeImage(workingImg=np.zeros((100, 100))):
    MaxPixel = findMaxPixel(workingImg)

    if MaxPixel is 0:
        return

    normalizingConstant = 255 / MaxPixel

    # extract necessary info
    workingImgShape = workingImg.shape
    rows = workingImgShape[0]
    cols = workingImgShape[1]

    # go through all values and normalize
    for row in range(rows):
        for col in range(cols):
            currPixel = workingImg[row][col]
            newPixel = int(currPixel * normalizingConstant)
            workingImg[row][col] = newPixel


def outputRho(xVal, yVal, theta):
    # normal form function
    return xVal * math.cos(theta) + yVal * math.sin(theta)


def rhoToIndex(rho, yBinSize, rhoCeiling):
    """
    :param rho: double - rho values
    :param yBinSize: double - range that each pixel represents
    :param rhoCeiling: int - the largest rho possible
    :return: int - 2d array index on center parameter space
    e.g. rho = 1
    1 is the index 100 from the top
    return 100
    """

    index = 0
    currValue = rhoCeiling - yBinSize

    while rho < currValue:
        currValue = currValue - yBinSize
        index = index + 1

    return index
