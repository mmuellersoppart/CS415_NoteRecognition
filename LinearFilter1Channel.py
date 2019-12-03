import numpy as np
from Kernel import Kernel
import cv2
import statistics


class LinearFilter:

    def __init__(self, srcImage=np.zeros((3, 3)), kernel=Kernel("mean", 3), thresh=50):
        """
        :param image: numpy.ndarray
        :param kernel: numpy.array
        """
        self.srcImage = srcImage
        self.intermediateImg = srcImage
        self.finalImage = np.ones((srcImage.shape)) * 255
        self.kernel = kernel

        # add padding
        padLen = int((self.kernel.kernelSize - 1) / 2)

        self.intermediateImg = cv2.copyMakeBorder(srcImage, padLen, padLen, padLen, padLen, cv2.BORDER_REPLICATE)

        self.srcImgH, self.srcImgW = self.srcImage.shape
        # keep track of image location in padding
        self.xStartPixel = 0
        self.xStartPixel = self.xStartPixel + padLen
        self.yStartPixel = 0
        self.yStartPixel = self.yStartPixel + padLen

        self.thresh = thresh

    def changeKernel(self, kernel):
        """
        :param kernel: kernel (user class)
        :return: nothing
        """
        # change kernel
        self.kernel = kernel
        # update intermediate file
        # add padding
        padLen = int((self.kernel.kernelSize - 1) / 2)
        self.intermediateImg = cv2.copyMakeBorder(self.srcImage, padLen, padLen, padLen, padLen, cv2.BORDER_REPLICATE)
        self.finalImage = self.srcImage.copy

    # for testing
    def printLinearFilterContentsPicture(self, fileName):
        cv2.imshow("src of " + fileName, self.srcImage)
        # cv2.imshow("intermediateImg of" + fileName, self.intermediateImg)
        cv2.imshow("finalImage of " + fileName, self.finalImage)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(f"""
        Source Image Info:
        srcImgDType: {self.srcImage.dtype} srcImgH: {self.srcImgH} srcImgw: {self.srcImgW}
        kernelType: {self.kernel.type} kernelDim: {self.kernel.kernelSize} 
        xStartPixel: {self.xStartPixel} yStartPixel: {self.yStartPixel}
        """)

    def printLinearFilterContents(self, fileName):
        print(f"""
        Linear Filter Info:
        srcImgDType: {self.srcImage.dtype} srcImgH: {self.srcImgH} srcImgw: {self.srcImgW} 
        kernelType: {self.kernel.type} kernelDim: {self.kernel.kernelSize} 
        xStartPixel: {self.xStartPixel} yStartPixel: {self.yStartPixel}
        """)

    def computeWeightedSum(self, xPos, yPos):
        """
        Computes the weighted average that the kernel specifics for a point.
        Works for a 3 channel img
        :param xPos: x pixel in x coordinate
        :param yPos: y pixel in y coordinate
        :return: int representing the resulting value for the blue channel
        """
        kernelMaxDistFromOrigin = int((self.kernel.kernelSize - 1) / 2)
        allDistFromOrigin = list(range(-kernelMaxDistFromOrigin, kernelMaxDistFromOrigin + 1))

        sumChannel = 0

        for distY in allDistFromOrigin:
            for distX in allDistFromOrigin:
                kernelVal = self.kernel.kernel[distX + kernelMaxDistFromOrigin][distY + kernelMaxDistFromOrigin]
                valOnIntermediateImg = \
                    self.intermediateImg[(self.xStartPixel + xPos) + distX][(self.yStartPixel + yPos) + distY]
                sumChannel = sumChannel + (kernelVal * valOnIntermediateImg)

        if self.kernel.type is "gaussianD1X" or self.kernel.type is "gaussianD1Y":
            sumChannel = sumChannel / 2
            sumChannel = sumChannel + 255 / 2

        return int(sumChannel)

    def getMedianValue4Point1Channel(self, xPos, yPos):
        """
        Computes the weighted average that the kernel specifics for a point.
        Works for a 3 channel img
        :param xPos: x pixel in x coordinate
        :param yPos: y pixel in y coordinate
        :return: int representing the resulting value for the blue channel
        """
        kernelMaxDistFromOrigin = int((self.kernel.kernelSize - 1) / 2)
        allDistFromOrigin = list(range(-kernelMaxDistFromOrigin, kernelMaxDistFromOrigin + 1))
        allPossibleIndex = list(range(self.kernel.kernelSize))

        allValues = []

        # go through all the values around the point (determined by kernel size)
        for distY in allDistFromOrigin:
            for distX in allDistFromOrigin:
                allValues.append(
                    self.intermediateImg[(self.xStartPixel + xPos) + distX][(self.yStartPixel + yPos) + distY])

        # find median in list
        median = statistics.median(allValues)

        return median

    def applyKernelToOnePixel(self, xPos, yPos):
        """
        :param xPos: x index on src image where to apply kernel
        :param yPos: y index on src image where to apply kernel
        :return: changes pixel on final output
        """
        allChannels = 1

        channelWeightedSum = 0
        pixelVal = []

        # compute the weighted sum of the kernel for a single channel
        channelWeightedSum = self.computeWeightedSum(xPos, yPos)

        # rules specific to this
        if channelWeightedSum > 255:
            channelWeightedSum = 255
        if channelWeightedSum < 0:
            channelWeightedSum = 0

        # update image with new pixel
        self.finalImage[xPos][yPos] = channelWeightedSum
        return [xPos, yPos]

    def applyKernelToOnePixelDarkness(self, xPos, yPos):
        """
        :param xPos: x index on src image where to apply kernel
        :param yPos: y index on src image where to apply kernel
        :return: changes pixel on final output
        """

        # compute the weighted sum of the kernel for a single channel
        channelWeightedSum = self.computeWeightedSum(xPos, yPos)

        # rules specific to this
        if channelWeightedSum > 255:
            channelWeightedSum = 255
        if channelWeightedSum < 0:
            channelWeightedSum = 0

        if channelWeightedSum < self.thresh:
            print(f"        clusterFound: {channelWeightedSum}")
            # update image with new pixel
            self.finalImage[xPos][yPos] = channelWeightedSum
            return [xPos, yPos]
        else:
            channelWeightedSum = 255
            self.finalImage[xPos][yPos] = channelWeightedSum
            return None

    def applyMedianKernelToOnePixelAllChannels(self, xPos, yPos):
        """
        :param xPos: x index on src image where to apply median kernel
        :param yPos: y index on src image where to apply median kernel
        :return: changes pixel on final output
        """
        allChannels = list(range(self.srcImage.shape[2]))

        channelWeightedSum = 0
        pixelVal = []

        # compute the weighted sum of the kernel for a single channel
        for channel in allChannels:
            channelMedianVal = self.getMedianValue4Point1Channel(xPos, yPos)
            pixelVal.append(channelMedianVal)

        # update image with new pixel
        self.finalImage[xPos][yPos] = pixelVal

    def correlation(self, slideLength):
        """
        ob - performs a correlation operation using the inputted kernel and source image
        :return: - the final image in the class is updated based on the kernel
        """
        # traverse
        if slideLength == 0:
            xIndices = list(range(self.srcImgW))
            yIndices = list(range(self.srcImgH))
        else:
            xIndices = list(range(0, self.srcImgW, slideLength))
            yIndices = list(range(0, self.srcImgH, slideLength))

        operations = ((self.srcImgW + 2 * (self.kernel.kernelSize / 2 - 1)) / slideLength) * \
                     ((self.srcImgH + 2 * (self.kernel.kernelSize / 2 - 1)) / slideLength)

        operations10 = int(operations / 10)

        print(f"Correlation total ops estimate is : {int(operations)}")
        print("operation total\n")

        pixelsProcessed = 0

        if self.kernel.type is not "median":
            for yIndex in yIndices:  # rows
                for xIndex in xIndices:  # cols
                    pixelsProcessed = pixelsProcessed + 1
                    if pixelsProcessed % operations10 == 0:
                        print(f"    {pixelsProcessed}")
                    self.applyKernelToOnePixel(yIndex, xIndex)
        else:
            for yIndex in yIndices:  # rows
                for xIndex in xIndices:  # cols
                    self.applyMedianKernelToOnePixelAllChannels(yIndex, xIndex)

        return self.finalImage

    def correlationKernelDarkness(self, slideLength):
        """
        ob - performs a correlation operation using the inputted kernel and source image
        :return: - the final image in the class is updated based on the kernel
        """

        # collect cluster points
        clusterPoints = []

        # traverse
        if slideLength == 0:
            xIndices = list(range(self.srcImgW))
            yIndices = list(range(self.srcImgH))
        else:
            xIndices = list(range(0, self.srcImgW, slideLength))
            yIndices = list(range(0, self.srcImgH, slideLength))

        # estimate of the times the kernel will be applied
        operations = ((self.srcImgW + 2 * (self.kernel.kernelSize / 2 - 1)) / slideLength) * \
                     ((self.srcImgH + 2 * (self.kernel.kernelSize / 2 - 1)) / slideLength)

        # for loading bar purposes
        operations10 = int(operations / 10)
        pixelsProcessed = 0
        print(f"Correlation total ops estimate is : {int(operations)}")
        print("operation total\n")

        for yIndex in yIndices:  # rows
            for xIndex in xIndices:  # cols
                pixelsProcessed = pixelsProcessed + 1
                if pixelsProcessed % operations10 == 0:
                    print(f"    {pixelsProcessed}")
                point = self.applyKernelToOnePixelDarkness(yIndex, xIndex)

                if point is not None:
                    clusterPoints.append(point)

        print("    Done!\n")

        return clusterPoints

    def convolution(self):
        """
        ob - performs a correlation operation using the inputted kernel and source image
        :return: - the final image in the class is updated based on the kernel
        """

        # prepare convolution
        self.kernel.flipVandH()

        # traverse
        xIndices = list(range(self.srcImgW))
        yIndices = list(range(self.srcImgH))

        if self.kernel.type is not "median":
            for yIndex in yIndices:  # rows
                for xIndex in xIndices:  # cols
                    self.applyKernelToOnePixel(yIndex, xIndex)
        else:
            for yIndex in yIndices:  # rows
                for xIndex in xIndices:  # cols
                    self.applyMedianKernelToOnePixelAllChannels(yIndex, xIndex)

        # revert back to original kernel
        self.kernel.flipVandH()

        return self.finalImage
