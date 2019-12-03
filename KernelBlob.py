import numpy as np
import cv2
import math

from FolderHandler import FolderHandler
from Kernel import Kernel
from LinearFilter1Channel import LinearFilter


class KernelBlobs:
    """
    This class finds all the blob locations
    """

    def __init__(self, img, kernelSize, percentOfCircle, stride):
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

        # Create image made of points that pass
        shapeImg = self.image.shape
        self.resultImage = np.ones(shapeImg) * 255

        cv2.imwrite("disposable.png", self.resultImage)

        #correlation operation
        linFilter = LinearFilter(self.image, self.kernel, self.thresh)

        #print info about correlation
        linFilter.printLinearFilterContents("Disposable1")
        #perform correlation - get a list of possible points
        blobList = linFilter.correlationKernelDarkness(stride)

        print(f"blob points under consideration: {len(blobList)}")
        print(blobList)

        cv2.imwrite("disposable2.png", linFilter.finalImage)

        #TODO: Find the darkest point in each cluster

        self.listOfCenters = []

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
