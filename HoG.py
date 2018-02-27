import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys

def createHistogram(magBlock,angleBlock):
    # Function to create a histogram of gradients given an 8x8 block
    hist = np.zeros([9])
    angleBlock = np.mod(angleBlock,180) / 20
    
    for i in range(len(angleBlock)):
        for j in range(len(angleBlock[i])):
            angle = angleBlock[i,j]
            rem = angle%1
            left = int(np.floor(angle))
                
            if (angle < 8.0):
                right = int(np.ceil(angle))
            else:
                right = 0

            hist[left] += (1-rem)*magBlock[i,j]
            hist[right] += rem*magBlock[i,j]
            
    return hist

def makeHoG(imgpath):
    # Step 1: Perform sobel derivative on WHOLE image (to cache results)
    # NB: is it faster to do an image fft and multiply than use opencv's default sobel?
    # Step 2: iterate over 8x8 blocks, creating one HoG per box
    # Step 3: normalize HoGs by iterating over (overlapping) 16x16 blocks,
    #           creating a 36x1 feature vector of from the HoGs of all the sub bins
    # Step 4: Roll up all the individual 36x1 vectors in one giant output vector

    # magic numbers based on commonly used examples.
    # EPS is for norm step to avoid div by zero, BLOCKSIZE and HISTSIZE are given from the algorithm description
    EPS = 0.0001
    BLOCKSIZE = 8
    HISTSIZE = 9

    # Read image, rescale to 64x128
    img = cv2.imread(imgpath)
    height, width = img.shape[0:2]
    if width != 64 or height != 128:
        xmargin = int((width - 64)/2)
        ymargin = int((height - 128)/128)
        img = img[ymargin:ymargin+128,xmargin:xmargin+64]
        width = 64
        height = 128

    # Perform derivatives
    img = np.float32(img) / 255.0
    gx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=1)
    gy = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=1)
    mag, angle =  cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # need to correct for color channels. According to the tutorial, we select the maximum gradient
    # on each pixel, and assign the corresponding direction.
    newMag = mag[:,:,0]
    newAng = angle[:,:,0]
    argmat = np.argmax(mag,2)
    for i in range(height):
        for j in range(width):
            newMag[i,j] = mag[i,j,argmat[i,j]]
            newAng[i,j] = angle[i,j,argmat[i,j]]

    mag = newMag
    angle = newAng

    # Subdivide image into bins, then create histograms of gradients over each bin
    numBlocksX = int(width/BLOCKSIZE)
    numBlocksY = int(height/BLOCKSIZE)
    HistList = np.empty([numBlocksY,numBlocksX,HISTSIZE])
    for row in range(numBlocksY):
        for col in range(numBlocksX):
            magBlock = mag[row*BLOCKSIZE:(row+1)*BLOCKSIZE-1, col*BLOCKSIZE:(col+1)*BLOCKSIZE-1]
            angleBlock = angle[row*BLOCKSIZE:(row+1)*BLOCKSIZE-1, col*BLOCKSIZE:(col+1)*BLOCKSIZE-1]
            HistList[row,col,:] = createHistogram(magBlock,angleBlock)

    # Normalization step, including rolling up into final vector
    finalVector = np.array([])
    for row in range(numBlocksY-1):
        for col in range(numBlocksX-1):
            bigHist = np.concatenate((HistList[row,col,:],HistList[row+1,col,:],
                                      HistList[row,col+1,:],HistList[row+1,col+1,:]))
            normHist = bigHist/np.sqrt(np.dot(bigHist,bigHist)+EPS)
            finalVector = np.concatenate((finalVector,normHist))
            
    return finalVector

##def main(argv):
##
##    BLOCKSIZE = 8
##    HISTSIZE = 9
##
##    if (len(argv) == 1):
##        imgpath = argv[0]
##    else:
##        print("Please provide the filename of an image. Correct usage: python HoG.py [filename]")
##        
##    featureVec = makeHoG(imgpath,BLOCKSIZE,HISTSIZE)
##            
##    return featureVec
##
##
##
##if __name__ == "__main__":
##    main(sys.argv[1:])
