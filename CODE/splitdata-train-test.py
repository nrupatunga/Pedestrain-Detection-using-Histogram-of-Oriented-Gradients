import os
import glob
import numpy as np
import random
import re
import pdb
import shutil
from tqdm import tqdm

ImagePath = "D:\\WindowsTestBench\\HOG\\HOG_DETECTOR\\HOG_DETECTOR\\neg_full"

train_dir = "D:\\WindowsTestBench\\HOG\\HOG_DETECTOR\\HOG_DETECTOR\\neg_train"
test_dir = "D:\\WindowsTestBench\\HOG\\HOG_DETECTOR\\HOG_DETECTOR\\neg_test"

def main():
    splitdata(ImagePath, "jpg", 90, 10)

def splitdata(dirImg, filetype, train_per, test_per ):
    '''This funcions splits the data into train and test data.
    dir : path of directory where we can find the images
    filetype: type of the files to be split (eg. jpg)
    train_per: percentage of train data
    test_per: percentage of test data'''

    pathname = dirImg + os.sep + "*." + filetype;
    filenamelist = (glob.glob(pathname))
    num_samples = len(filenamelist)
    #pdb.set_trace()

    train_num = int(train_per*num_samples/100)
    test_num = int(test_per*num_samples/100)

    count = 0
    for (dir_path, dir_names, file_names) in os.walk(dirImg):
        assert isinstance(file_names, object)
        random.shuffle(file_names)
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for img in tqdm(file_names):
                srcfile = dirImg + os.sep + img
                if count < train_num:
                    dstdir = train_dir
                    shutil.copy(srcfile, dstdir)
                else:
                    dstdir = test_dir
                    shutil.copy(srcfile, dstdir)
                count += 1

    print 'Done splitting, please check train.txt and test.txt'

if __name__ == '__main__':
    main()
