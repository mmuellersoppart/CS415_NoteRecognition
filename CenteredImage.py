import string
import numpy as np
import math
import cv2

class centeredImage:

    #
    # NOTE: pos refers to the location in a grid
    #       index refers to a location in an array
    #

    def __init__(self, srcImg = np.ones((100, 100))):
        """
        :param type: string of type of operation the kernel will perform
        :param kernel: the kernel itself (square of any odd dimension)
        :param kernelSize: could be either num of rows, cols, or diagonal
        :param simple: does this only use one kernel
        """

        #extract info
        self.image = srcImg
        srcImgShape = srcImg.shape
        self.height = srcImgShape[0]
        self.width = srcImgShape[1]

        if len(srcImg.shape) == 2:
            self.channels = 1
        else:
            self.channels = srcImgShape[2]

        #form grid
        Offsets = self.findCenter(srcImgShape)

        self.offsetLeft = Offsets[0]
        self.offsetRight = Offsets[1]
        self.offsetUp = Offsets[2]
        self.offsetDown = Offsets[3]

        self.domainL = int(-1 * self.offsetLeft)
        self.domainR = int(self.offsetRight)
        self.rangeU = int(self.offsetUp)
        self.rangeD = int(-1 * self.offsetDown)


    def findCenter(self, shape):
        """
        centers the image. The first quadrant may end up being bigger than the rest (if even)
        :param shape: [] - list of dimensions of srcImg
        :param centerOffsets: [] - with values for how far the img stretches from center
        :return: Changes centerOffset values
        """

        centerOffsets = []

        rows = shape[0]
        columns = shape[1]

        # determine distances from center on the x
        if columns % 2 is 1:  # isOdd
            xDistL = columns // 2
            xDistR = xDistL
        else:  # perfect center cannot be found
            xDistL = columns / 2 - 1
            xDistR = columns / 2

        centerOffsets.append(xDistL)
        centerOffsets.append(xDistR)

        # same idea for y
        if rows % 2 is 1:
            yDistU = rows // 2
            yDistD = yDistU
        else:
            yDistU = rows / 2
            yDistD = rows / 2 - 1

        centerOffsets.append(yDistU)
        centerOffsets.append(yDistD)

        return centerOffsets

    def drawGrid(self):
        """
        :param color: [B, G, R]
        :return: Modifies image so one can see the grid
        """
        self.drawHorizonalLine(0)
        self.drawVerticalLine(0)


    def drawHorizonalLine(self, yPos):
        """
                :param xPos: int - Using centered coordinate
                :param color: [B, G, R]
                :return: modify src image with vert
                """

        yIndex = self.yPosToYIndex(yPos)
        yIndex = int(yIndex)
        print(f"yIndex: {yIndex}")
        self.image[yIndex, :] = 155


    def drawVerticalLine(self, xPos):
        """
        :param xPos: int - Using centered coordinate
        :param color: [B, G, R]
        :return: modify src image with vert
        """

        xIndex = self.xPosToXIndex(xPos)
        xIndex = int(xIndex)
        print(f"xIndex: {xIndex}")
        self.image[:, xIndex] = 155

    #
    # Convert to 2D array functions
    #

    def xPosToXIndex(self, xPos):
        """
        ob - changed from center coordinates to a usable array coordinates
        :param xPos: int - Using centered coordinates
        :return: xIndex in the array
        """

        xIndex = xPos + self.offsetLeft

        return int(xIndex)

    def yPosToYIndex(self, yPos):
        """
        ob - changed from center coordinates to a usable array coordinates
        :param yPos: int - using centered coordinates
        :return: yIndex in array
        """
        if yPos >= 0:
            return self.offsetUp - yPos
        else:
            return int(abs(yPos) + self.offsetUp)

    def returnEveryCoordinate(self):
        allCordinates = []

        for xPos in range(self.domainL, self.domainR + 1):
            for yPos in range(self.rangeD, self.rangeU + 1):
                point = [xPos, yPos]
                allCordinates.append(point)

        return allCordinates

    def getPixelVal(self, xPos, yPos):
        return [self.xPosToXIndex(xPos), self.yPosToYIndex(yPos)]