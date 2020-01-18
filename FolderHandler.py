"""
Author: Marlon Mueller-Soppart
Date: 01/18/2020
Objective: Easily access paths of all the directories in the project
"""

import pathlib
import sys
import cv2

class FolderHandler:
    """
    makes folders and reads and rights things in its folders
    """

    def __init__(self):
        #
        # get orientated in computer
        #

        # get paths where file is located
        self.projectPath = pathlib.Path.cwd()

        # locate two important files (they cannot be renamed)
        self.srcImgPath = pathlib.Path.cwd().joinpath('srcImg')
        self.outputImgPath = pathlib.Path.cwd().joinpath('outputImg')
        self.inputFilePath = pathlib.Path.cwd().joinpath('inputFiles')

        # check is there is a folder for output files
        if self.srcImgPath.is_dir():
            pass
        else:  # if there isn't, make one
            self.srcImgPath.mkdir()

        # check is there is a folder for output files
        if self.outputImgPath.is_dir():
            for f in self.outputImgPath.iterdir():
                if f.is_file():
                    f.unlink()
        else:  # if there isn't, make one
            self.outputImgPath.mkdir()

        # check is there is a folder for output files
        if self.inputFilePath.is_dir():
            pass
        else:  # if there isn't, make one
            self.inputFilePath.mkdir()

    def returnFile(self, fileName):
        # open the file named in command one of the command line argument
        try:
            f = open(self.inputFilePath.joinpath(fileName), "r")
            return f
        except:
            print("ERROR: user input file not found or not included in arg")
            return

    def writeImageToOutput(self, img, fileName):
        cv2.imwrite(str(self.outputImgPath.joinpath(fileName)), img)