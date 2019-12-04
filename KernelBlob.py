import math

import cv2
import numpy as np

from Kernel import Kernel
from LinearFilter1Channel import LinearFilter


class KernelBlobs:
    """
    This class finds all the blob locations
    """

    def __init__(self, img, kernelSize, percentOfCircle, stride, neighborCheck):
        """
        :param img: Thresholded per black and white image
        :param kernelSize: The length of the kernel
        :param percentOfCircle: The proportion of a blob's areas compared to a circle
        """

        # store in object
        self.image = img
        self.kernelSize = kernelSize
        self.percent = percentOfCircle

        # Now we have a kernel of just ones
        self.kernel = Kernel("mean", self.kernelSize, 0)

        # The rough area we are trying to hit
        self.thresh = self.darkMatterRequirement()

        # Create image made of points that pass the test
        shapeImg = self.image.shape
        self.resultImage = np.ones(shapeImg) * 255

        #correlation operation
        linFilter = LinearFilter(self.image, self.kernel, self.thresh)

        #print info about correlation
        linFilter.printLinearFilterContents("Disposable1")
        #perform correlation - get a list of possible points
        blobList = linFilter.correlationKernelDarkness(stride)

        print(f"blob points under consideration: {len(blobList)}")
        print(blobList)

        cv2.imwrite("outputImg/blobPoints.png", linFilter.finalImage)

        # cut down the list to just the important ones
        self.listOfCenters = self.minPixelInKernel(numPixelCheck=neighborCheck, stride=stride, listPoints=blobList,
                                                   image=linFilter.finalImage)

        print(f"Total blob centers determined: {len(self.listOfCenters)}")
        print(self.listOfCenters)

    def minPixelInKernel(self, numPixelCheck, stride, listPoints, image):
        """
        :param numPixelCheck: int - number of pixels checked to the right and left e.g. 3 _ _ _ . _ _ _
        :param stride: int - stride of the kernel (also how far apart points are)
        :param listPoints: list of coordinates - points underconsideration
        :param image: nparray - points are black and everything is white
        :return: list of min pixels
        """

        finalPoints = []

        for point in listPoints:

            isDone = False

            # extract info
            currXPos = point[0]
            currYPos = point[1]
            pointVal = image[currYPos][currXPos]

            # compare to neighboring pixels
            for xDir in range(-numPixelCheck, numPixelCheck + 1):
                if isDone:
                    break
                for yDir in range(-numPixelCheck, numPixelCheck + 1):
                    # print(positiveDirection, negativeDirection)
                    if xDir == 0 and yDir == 0:
                        continue

                    xShift = xDir * stride
                    yShift = yDir * stride

                    # the point is not the smallest (darkest)

                    if pointVal >= image[currYPos + yShift][currXPos + xShift]:
                        # delete oneself from image
                        image[currYPos][currXPos] = 255
                        isDone = True
                        break

            # done checking neighboring points
            if not isDone:
                finalPoints.append(point)
        # done going through all points
        return finalPoints


    def darkMatterRequirement(self):
        """
        Determines the amount of black required in kernel to call a note a blob
        :return: How much a mean kernel is expected to return
        """
        # whip out your pens and pencils time for some simple geometry
        areaOfSquare = self.kernelSize ** 2
        areaOfCircle = math.pi * ((self.kernelSize/2) ** 2)
        difSquareCircle = areaOfSquare - areaOfCircle

        proportionCircle = areaOfCircle / areaOfSquare
        proportionDifSquareCircle = difSquareCircle / areaOfSquare

        weightSum = 255 * proportionDifSquareCircle + 0 * proportionCircle

        # lower the threshold an arbitrary amount
        weightSum = weightSum * self.percent

        return weightSum
