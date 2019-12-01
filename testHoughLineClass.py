import pytest
import numpy as np
import pathlib
import cv2
import math
import sys
import random
import time
import bisect


from FolderHandler import FolderHandler
from CenteredImage import centeredImage
from houghLineCustom import houghLineCustom

@pytest.fixture()
def instantiateCentered():
    img1 = cv2.imread('srcImg/IMG_1801.jpg')
    img1Centered = centeredImage(img1)
    allPixels = img1Centered.returnEveryCoordinate()

    return img1Centered

def test_makeClass(instantiateCentered):
    img1 = cv2.imread('srcImg/IMG_1801.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    hough1 = houghLineCustom(img1)