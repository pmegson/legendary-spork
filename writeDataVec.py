import cv2
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
import HoG
from progress.bar import Bar


import sys
import os

def makeDataVec(baseDir, train):

    os.chdir(baseDir)
    if train:
        posDirName = "C:/users/pmegson/Documents/openCVTutorial/HoG/INRIAPerson/96X160H96/Train/pos/"
    else:
        posDirName = "C:/users/pmegson/Documents/openCVTutorial/HoG/INRIAPerson/70X134H96/Test/pos/"

    dirNames = [ baseDir + "neg/", posDirName]
    
    truthVal = 0
    trainSamples = np.array([[]])
    trainFeatures = []
    f = open('trainingData.csv','w+')
    
    for dirName in dirNames:
        os.chdir(dirName)
        fileList = os.listdir(dirName)

        bar = Bar('Analyzing Image', max = len(fileList))

        for imgPath in fileList:
            try:
                vec = HoG.makeHoG(imgPath)
            except:
                print(filename)
            f.write(str(truthVal))
            f.write(",")
            vec.tofile(f,sep=",")
            f.write("\n")

            bar.next()
            
            
            ##if trainSamples.size == 0:
            ##    trainSamples = np.array([HoG.makeHoG(filename)])
            ##else:
            ##    vec = np.array([HoG.makeHoG(filename)])
            ##    trainSamples = np.append(trainSamples,vec,axis=0)

            #if (count%50 == 0) or (count%len(fileList)==0):
            #    print(str(count)+"/"+str(len(fileList)))

            #count += 1
            ##trainFeatures += [truthVal]
        
        truthVal += 1   
        os.chdir('..')
        bar.finish()
        
    f.close()




def main(argv):

    testDir = "C:/users/pmegson/documents/openCVTutorial/HoG/INRIAPerson/Test/"
    trainDir = "C:/users/pmegson/documents/openCVTutorial/HoG/INRIAPerson/Train/"

    if not argv:
        makeDataVec(trainDir, True)
        makeDataVec(testDir, False)
    elif argv[0] == "train":
        makeDataVec(trainDir, True)
    elif argv[0] == "test":
        makeDataVec(testDir, False)

    return

if __name__ == "__main__":
    main(sys.argv[1:])
