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

    def __init__(self, srcImg = np.ones((100, 100)), xQuantization = 1000, yQuantization = 1000):

        #src image has canny applied to it and has one channel

        #extract info
        #get image
        self.centeredImage = centeredImage(srcImg)
        self.xQuant = xQuantization
        self.yQuant = yQuantization

        #calculate largest rho
        self.rhoCeiling = self.largestRho()

        #build accumulator
        self.accumulator = self.accumulatorSpace()

        #fill accumulator


    def fillAccumulator(self):
        self.accumulator = transformImagetoAccumulator()
        accumulatorSpaceCopy = accumulatorSpace.copy()

        normalizeImage(accumulatorSpaceCopy)

    def transformImagetoAccumulator(self):
        """
        :param imageSpace: img - after canny detection
        :param imageSpaceOffsets: [] - distance to each side of the centered origin
        :param accumulatorSpace: matrix - same dimension as parameter space
        :param parameterSpaceInfo: [] - how big the bins are
        :return: accumulator modified
        """

        #go through each pixel


        # extract necessary info from src image
        rows = self.centeredImage.image.shape[0]
        cols = self.centeredImage.image.shape[1]

        numPoints = 0


        for row in range(rows):
            for col in range(cols):
                if self.centeredImage.image[row][col] > 250:  # and numPoints < 10:
                    xCenteredPixel = SrcIndextoCenteredIndexXDim(imageSpaceOffsets, col)
                    yCenteredPixel = SrcIndextoCenteredIndexYDim(imageSpaceOffsets, row)
                    parameterSpaceInfo = []
                    paramSpaceFourPoint = self.buildParameterSpace()
                    imagePointToParameterSpace(paramSpaceFourPoint, parameterSpaceInfo, xCenteredPixel, yCenteredPixel,
                                               rhoCeiling)
                    numPoints = numPoints + 1
                    accumulatorSpace = np.add(accumulatorSpace, paramSpaceFourPoint)
                else:
                    pass

        return accumulatorSpace

    def largestRho(self):
        maxDim = math.sqrt(self.centeredImage.offsetRight ** 2 + self.centeredImage.offsetUp[2] ** 2)
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

        accumulatorSpace = self.buildParameterSpace(rhoCeiling, parameterSpaceInfo)

        print("Parameter space info")
        print(f"rhoCeiling: {rhoCeiling} XBinSize: {parameterSpaceInfo[0]} YBinSize: {parameterSpaceInfo[1]}")

        return accumulatorSpace
