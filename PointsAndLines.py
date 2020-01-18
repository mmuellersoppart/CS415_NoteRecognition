# Name - Points and Lines
# Author - Marlon Mueller-Soppart
# Date - 01/18/2020
# ob - This file helps find a point in relation to the 5 line in a staff

def centerOfLine(pt, threshImg):
    """
    ob - find the center of the line (at least confirm point is on black pixel)
    :param pt: Int - y-coordinate of an anchor points
    :param threshImg: threshholded image
    :return: Int - y-coordinate of the center of the closest line
    """

    # start on closest line
    onLine = findClosestLine(pt, threshImg)

    bottomLinePixel = 0
    topLinePixel = 0

    # find bottom()
    currLinePixel = onLine
    while threshImg[currLinePixel][0] == 0:
        currLinePixel -= 1
    bottomLinePixel = currLinePixel + 1

    # find top()
    currLinePixel = onLine
    while threshImg[currLinePixel][0] == 0:
        currLinePixel += 1
    topLinePixel = currLinePixel - 1

    # print(
    #     f"We started with {topLinePixel} {onLine} {bottomLinePixel}\n We ended with {int((topLinePixel + bottomLinePixel) / 2)}")

    return int((topLinePixel + bottomLinePixel) / 2)


def findClosestLine(pt, img):
    """
    ob - find a black line closest to point
    :param pt: [Int, Int] - yCor of point
    :param img: np array
    :return: Int - The first point found that is on black (value = 0)
    """

    imgHeight = img.shape[0]
    searchDepth = 0
    row = pt[1]
    col = pt[0]

    while searchDepth < imgHeight:
        if img[row + searchDepth][col] == 0:
            return row + searchDepth
        elif img[row - searchDepth][col] == 0:
            return row - searchDepth

        searchDepth += 1

    return -1


def travelRightUntilNote(pt, threshImg, blobCenterX):
    """
    ob - #y coordinate of a line at a certain note
    :param pt: [Int, Int] - yCoordinate
    :param threshImg: np array
    :param blobCenterX: Int - x index of note
    :return: Int - y val of line at x index
    """

    imgWidth = threshImg.shape[1]

    while pt[0] < blobCenterX and pt[0] < imgWidth:
        # move to the right
        pt[0] = pt[0] + 1

        # make sure point is on line
        pt[1] = findClosestLine(pt, threshImg)

    return pt


def linePositions(anchorList, threshImg, blobCenter):
    """
    :param anchorList: [[Int, Int],...]
    :param threshImg:
    :param blobCenter: [Int, Int]
    :return: [yCor, yCor,...] of all the lines
    """

    linePositions = []

    for anchor in anchorList:
        anchorL = [0, anchor]
        yVal = travelRightUntilNote(anchorL, threshImg, blobCenter)
        linePositions.append(yVal[1])

    linePositions.sort()

    return linePositions


def determineNote(linePositions, blobCenter):
    """
    :param linePositions: [Ints] - y coordinates of lines on column
    :param blobCenter: [Int, Int] - coordinate of blob
    :return: string - note name for music 21
    """

    blobY = blobCenter[1]

    # edge cases
    if blobY < linePositions[0]:
        return "B''4"

    if blobY > linePositions[4]:
        return "DD4"

    lowestNote = 0

    noteDict = {0: "A''4", 1: "G4", 2: "F4", 3: "E4", 4: "D4", 5: "C4", 6: "BB4", 7: "AA4", 8: "GG4", 9: "FF4"}
    pos2NumDict = {"top": 0, "middle": 1, "bottom": 2}
    top = 0
    bottom = 0

    for index in range(1, 5):
        if blobY <= linePositions[index]:
            bottom = linePositions[index]
            top = linePositions[index - 1]
            note = lowestNote + pos2NumDict[notePosBetweenLines(top, bottom, blobY, .5)]
            return noteDict[note]
        else:
            lowestNote += 2


def notePosBetweenLines(top, bottom, point, tolerance):
    """
    ob - is the note on a line or in the middle
    :param top: line closer to top of image (smaller value)
    :param bottom: line closer to bottom (bigger value)
    :param point: coordinate in question
    :param tolerance: double - 0 middle takes up everything, 1 no middle
    :return: 0 (top), 1 (middle), 2 (bottom)
    """
    totalDistance = bottom - top
    halfDistance = totalDistance / 2

    boundary1 = halfDistance * tolerance

    if point < top + boundary1:
        return "top"
    elif point < top + boundary1 + 2 * (halfDistance - boundary1):
        return "middle"
    else:
        return "bottom"
