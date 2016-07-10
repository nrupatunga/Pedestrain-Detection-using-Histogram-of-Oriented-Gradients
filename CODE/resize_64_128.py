import os
import glob
import numpy as np
import random
import pdb
import shutil
import cv2
from tqdm import tqdm

ImagePath = "F:\\ME-2016\\MSVC-WS\\HOG\\HOG_DETECTOR\\HOG_DETECTOR\\hard_negs"
dstdir = "F:\\ME-2016\\MSVC-WS\\HOG\\HOG_DETECTOR\\HOG_DETECTOR\\hard_negs_64_128"

def main():
    for (dir_path, dir_names, file_names) in os.walk(ImagePath):
        for img in tqdm(file_names):
            srcfile = ImagePath + os.sep + img
            Img = cv2.imread(srcfile)
            resizedImg = cv2.resize(Img, (64, 128))
            dstfile = dstdir + os.sep + 'iter_two_' + img
            cv2.imwrite(dstfile, resizedImg)

if __name__ == '__main__':
    main()
