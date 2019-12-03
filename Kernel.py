import string
import numpy as np
import math


class Kernel:

    def __init__(self, type="", kernelSize=1, constant1=0):
        """
        :param type: string of type of operation the kernel will perform
        :param kernel: the kernel itself (square of any odd dimension)
        :param kernelSize: could be either num of rows, cols, or diagonal
        :param simple: does this only use one kernel
        """

        self.type = type

        if kernelSize % 2 == 0:
            kernelSize = 3
            raise ValueError
        else:
            self.kernelSize = kernelSize


        if type is "gaussian" and constant1 <= 0:
            print("Error: gaussian needs a sigma greater than 0")
            self.constant1 = 1
            raise ValueError
        else:
            self.constant1 = constant1

        # incase additional constants need to be specified (e.g. gaussian:sigma)
        self.kernel = self.makeKernel()

    def makeKernel(self):
        """
        in - type, size
        :return: np.ndarray
        """
        kernel = None
        filterType = self.type

        if filterType is "mean":
            # fill a matrix with only ones
            kernel = np.full((self.kernelSize, self.kernelSize), 1 / (self.kernelSize * self.kernelSize))
        #    print(kernel)
        elif filterType is "gaussian":
            kernel = self.calculateGaussianKernel()
        elif filterType is "median":
            # work will be done in the convolution
            kernel = np.ones((self.kernelSize, self.kernelSize))
        elif filterType is "sharpen":
            kernelSharpen = np.zeros((self.kernelSize, self.kernelSize))
            centerIndex = int((self.kernelSize - 1) / 2)
            kernelSharpen[centerIndex, centerIndex] = 2

            kernelMean = np.full((self.kernelSize, self.kernelSize), 1 / (self.kernelSize * self.kernelSize))

            kernel = np.subtract(kernelSharpen, kernelMean)
            # print(kernel)
        else:
            kernel = []

        return kernel

    def makeXDistMatrix(self):
        """
        :return: matrix with column values the distance column is from center
        [-2 -1 0 1 2]
        [-2 -1 0 1 2]
        [-2 -1 0 1 2]
        """

        # incase size 1
        if self.kernelSize == 1:
            return np.full((1, 1), 0)

        xDistMatrix = []
        maxDistFromZero = (self.kernelSize - 1) // 2
        allDistances = list(range(-maxDistFromZero, maxDistFromZero + 1))
        # print(allDistances)

        for dist in range(len(allDistances)):
            xDistMatrix.append(allDistances)

        return np.array(xDistMatrix)

    def makeYDistMatrix(self):
        """
        :return: matrix with column values the distance column is from center
        [2 2 2 2 2]
        [1 1 1 1 1]
        [0 0 0 0 0]
        [-1 -1 -1 -1 -1]
        [-2 -2 -2 -2 -2]
        """

        # incase size 1
        if self.kernelSize == 1:
            return np.full((1, 1), 0)

        yDistMatrix = []
        maxDistFromZero = (self.kernelSize - 1) // 2
        allDistances = list(range(-maxDistFromZero, maxDistFromZero + 1))

        for dist in allDistances:
            row = [dist for x in range(len(allDistances))]
            # print(f"printing row: {row}")
            yDistMatrix.append(np.array(row))

        return np.array(yDistMatrix)

    def calculateGaussianKernel(self):
        """
        :return:self.kernel
        """
        # preparation
        xPosMatrix = self.makeXDistMatrix()
        yPosMatrix = self.makeYDistMatrix()
        sigma = self.constant1

        finalMatrix = np.empty((self.kernelSize, self.kernelSize))

        for xIndex in range(self.kernelSize):
            for yIndex in range(self.kernelSize):
                finalMatrix[xIndex][yIndex] = Kernel.gaussianFunction(sigma, xPosMatrix[xIndex][yIndex],
                                                                      yPosMatrix[xIndex][yIndex])

        #print(finalMatrix)
        return finalMatrix

    def flipH(self):
        """
        :return: a matrix that has been flipped horizontally. left values on right and right on left
        """
        self.kernel = np.flip(self.kernel, 1)

    def flipV(self):
        """
        :return: a matrix that has been flipped horizontally. left values on right and right on left
        """
        self.kernel = np.flip(self.kernel, 0)

    def flipVandH(self):
        """
        :return: (for convolution) flips a matrix vertically and then horizontally
        """
        self.flipV()
        self.flipH()


    @staticmethod
    def gaussianFunction(sigma=0.0, xAxisPos=1, yAxisPos=1):
        """
        :param sigma: how spread out the distribution is. Can extend beyond kernel.
        :param xAxisPos: how many cols from center (pos and neg)
        :param yAxisPos: how many rows from center (pos and neg)
        :return: float value for point in kernel
        """

        firstTerm = 1 / (2 * math.pi * math.pow(sigma, 2))
        secondTerm = math.e
        secondTermPower = (-1) * ((math.pow(xAxisPos, 2) + math.pow(yAxisPos, 2)) / (2 * math.pow(sigma, 2)))
        return firstTerm * math.pow(secondTerm, secondTermPower)
