import pytest
import cv2

import PointsAndLines

@pytest.fixture(autouse=True)
def setup():
    print("*Setup*")


@pytest.fixture()
def Image1():
    # convert 3 channels to 1 channel
    testImg = cv2.imread("srcImg/IMG_1810.jpg")
    testImgBW = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
    # rescale
    scale_percent = 60  # percent of original size
    width = int(testImgBW.shape[1] * scale_percent / 100)
    height = int(testImgBW.shape[0] * scale_percent / 100)
    dim = (width, height)
    testImgBW = cv2.resize(testImgBW, dim, interpolation=cv2.INTER_AREA)

    # apply smoothing
    testImgBW = cv2.GaussianBlur(testImgBW, (25, 25), 0)

    # adaptive threshold
    # testImgBWAdap = cv2.adaptiveThreshold(testImgBW, maxValue=threshMaxVal, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                       thresholdType=cv2.THRESH_BINARY, blockSize=threshBlockSize, C=2)

    # apply binary threshold
    ret, testImgBWAdap = cv2.threshold(testImgBW, 170, 255, cv2.THRESH_BINARY)

    return testImgBWAdap


blob1 = [117, 208]
anchors1 = [[0, 278], [0, 146], [0, 322], [0, 190], [0, 234]]

def test_imageSetup(Image1):
    if len(Image1.shape) == 2:
        assert True

def test_ClosestLine(Image1):
    #determine bottom of line
    bottomPixel = 0
    currPixel = 322
    while Image1[currPixel][0] == 0:
        currPixel += 1
    bottomPixel = currPixel - 1

    print(bottomPixel)

    testResult = PointsAndLines.findClosestLine([0, 400], Image1)

    print(testResult)

    assert bottomPixel == testResult

def test_TravelRight(Image1):
    result = PointsAndLines.travelRightUntilNote([0, 322], Image1, 117)
    assert abs(result[1] - 322) < 10

    result2 = PointsAndLines.travelRightUntilNote([0, 90], Image1, 117)
    assert (abs(result2[1] - 143) < 10)

def test_LinePositions(Image1):
    print(PointsAndLines.linePositions(anchors1, Image1, [117, 230]))
    assert 5 == len(PointsAndLines.linePositions(anchors1, Image1, [117, 230]))

def test_DetermineNote(Image1):
    point1 = 150
    assert "middle" == PointsAndLines.notePosBetweenLines(100, 200, point1, .5)

    point2 = 310
    assert "top" == PointsAndLines.notePosBetweenLines(300, 400, point2, .5)

    point3 = 310
    assert "middle" == PointsAndLines.notePosBetweenLines(300, 400, point3, .1)


