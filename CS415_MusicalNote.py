import numpy as np
import pathlib
import cv2
import time
import statistics as stats
from FolderHandler import FolderHandler
import sklearn
from sklearn.cluster import KMeans
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

    #apply binary threshold
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

    while lines is not None and len(lines) > 12:
        houghVotes = houghVotes + 20
        print(f"lower threshold to {houghVotes}")
        lines = cv2.HoughLines(imgCanny, rho=1, theta=np.pi / 180, threshold=houghVotes)
        if houghVotes > 1000:
            print("ERROR: We cannot get rid of the lines!")
            break

    # ********************
    # Remove Unwanted Lines
    print("3. Remove Unwanted Lines\n")
    fewLinesStart = time.time()
    # ********************

    #filter out unwanted lines

    rhos = []
    thetas = []
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            rhos.append(rho)
            thetas.append(theta)

    #get the mode of the lines
    thetasCopy = thetas.copy()
    for index1 in range(len(thetasCopy)):
        thetasCopy[index1] = round(thetasCopy[index1], 2)

    thetaMode = stats.mode(thetasCopy)

    #search for lines that deviate too much (20 degrees greater than the mode)
    badLines = [i for i, x in enumerate(thetas) if abs(x - thetaMode) > .35]

    #remove the lines that are too crooked
    for index2 in sorted(badLines, reverse=True):
        del rhos[index2]
        del thetas[index2]

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
            cv2.line(imgCanny, (x1, y1), (x2, y2), 170, 2)

    print()

    # ********************
    # Find y-index of the 5 lines
    print("4. y-index of lines\n")
    fewLinesStart = time.time()
    # ********************

    rhos = np.array(rhos)
    #select the anchors of each line to the left
    kmeans1 = KMeans(n_clusters=5)
    kmeans1.fit(rhos.reshape(-1, 1))

    lineAnchors = []

    # where the 5 ones are
    for cluster in kmeans1.cluster_centers_:
        lineAnchors.append(cluster[0])
        imgCanny = applyColorKernel(imgCanny, 1, int(cluster[0]), 9)

    lineAnchors.sort(reverse=True) #bottom line at the top of the list

    # write initial lined image
    wrkSpace1.writeImageToOutput(imgCanny, imgNameNoExtension + "_ClusterDots" + ".png")

    # ********************
    # Begin blob detection
    print("5. blob detection\n")
    blobStart = time.time()
    # ********************

    #gameplan

    #approximate the size of the note


    #average space between lines
    averageSpacing = calcAverageSpacing(lineAnchors)

def calcAverageSpacing(list):
    numSpaces = len(list) - 1

    totalSpace = 0
    for index in range(numSpaces):
        totalSpace = totalSpace + (list[index] - list[index + 1])

    avgSpace = int(totalSpace/numSpaces)

    if avgSpace % 2 == 0:
        avgSpace = avgSpace - 1

    return avgSpace

def applyColorKernel(workingImg, xPos, yPos, size):
    """
    :param maxKernel: object , kernel type max
    :param workingImg: img - where the kernel will be applied to
    :param kernel: max kernel - (a kernel with only 1s)
    :param xPos: int - xPos on matrix coordinate (not in terms of theta or rho)
    :param yPos: int - yPos on matrix coordinate (not in terms of theta or rho)
    :return: bool True for max and F for not max
    """

    kernelMaxDistFromOrigin = int((size - 1) / 2)
    allDistFromOrigin = list(range(-kernelMaxDistFromOrigin, kernelMaxDistFromOrigin + 1))

    # go through all the values around the point (determined by kernel size)
    for distY in allDistFromOrigin:
        for distX in allDistFromOrigin:
            try:
                workingImg[yPos + distY][xPos + distX] = 200
            except IndexError:
                pass

    return workingImg

main()
