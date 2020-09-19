import cv2
import os
import matplotlib.pyplot as plt
import urllib.request as urlreq
import numpy as np
from pylab import rcParams
from scipy.spatial import Delaunay
import sys

from point_finder import PointFinder

lbfmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

LBFmodel = "models/lbfmodel.yaml"
haarcascade = "models/haarcascade.xml"

def getTriangulation(pf):
    # detect face with haar cascade
    face = pf.detectFace()

    # find landmarks in the face with lbf model
    landmarks, convex = pf.detectLandmarks(face)

    # calculate delaunay triangulation. note: this is just 
    # indecies of vertecies not actually 
    # the vertex positions themselves
    delaunayTriangles = Delaunay(landmarks)
    triangluation = delaunayTriangles.simplices

    return triangluation, landmarks, pf.getImage(), convex

def applyMask(b, m):
    h = b.shape[0]
    w = b.shape[1]

    rgb = np.zeros((h, w, 4), np.uint8)

    for y in range(0, h):
        for x in range(0, w):
            if(m[y, x][0] == 255):
                rgb[y, x][0] = b[y, x][0]
                rgb[y, x][1] = b[y, x][1]
                rgb[y, x][2] = b[y, x][2]
                rgb[y, x][3] = 255
            else:
                rgb[y, x][3] = 0
    
    return rgb
def overlay(b, o):
    h = o.shape[0]
    w = o.shape[1]

    for y in range(h):
        for x in range(w):
            if(o[y, x][3] > 252 and not(o[y, x][0] > 250 or o[y, x][1] > 250 or o[y, x][2] > 250)):
                b[y, x] = o[y, x]

if __name__ == "__main__":

    pic1 = "2.jpg"
    pic2 = "1.jpg"
    
    pf1 = PointFinder(pic1)
    pf2 = PointFinder(pic2)

    tri1, landmarks, image, convex = getTriangulation(pf1)
    tri2, landmarks2, image1, convex = getTriangulation(pf2)

    for point in convex:
        print(point)

    b, g, r = cv2.split(image1)

    outputImage = np.zeros((image.shape[0], image.shape[1], 4), np.uint8)
    outputImage[:, :, 0] = b
    outputImage[:, :, 1] = g
    outputImage[:, :, 2] = r
    outputImage[:, :, 3] = 255

    count = 0

    for tri in tri1:
        first = np.float32([landmarks[tri[0]], landmarks[tri[1]], landmarks[tri[2]]])
        second = np.float32([landmarks2[tri[0]], landmarks2[tri[1]], landmarks2[tri[2]]])

        r1 = cv2.boundingRect(first)
        r2 = cv2.boundingRect(second)

        (x, y, w, h) = r1
        (x1, y1, w1, h1) = r2
        
        firstCropped = np.array([[first[0][0] - x, first[0][1] - y], [first[1][0] - x, first[1][1] - y], [first[2][0] - x, first[2][1] - y]], np.int32)
        secondCropped = np.array([[second[0][0] - x1, second[0][1] - y1], [second[1][0] - x1, second[1][1] - y1], [second[2][0] - x1, second[2][1] - y1]], np.int32)

        try:
            affine = cv2.getAffineTransform(np.float32(firstCropped), np.float32(secondCropped))
        except TypeError as t:
            print("errored on")
            print(type(first))
            print(type(second))

        # warp triangle from first image to shape of second
        cropped = image[y: y + h, x: x + w]
        mask = np.zeros_like(cropped)

        cv2.fillConvexPoly(mask, firstCropped, 255)

        masked = applyMask(cropped, mask)

        warped = cv2.warpAffine(masked, affine, (w1, h1))

        #add triangle to full image
        area = outputImage[y1: y1 + h1, x1: x1 + w1]
        
        overlay(area, warped)

        outputImage[y1: y1 + h1, x1: x1 + w1] = area
        
        count += 1

    # save image

    plt.imshow(outputImage)
    cv2.imwrite('og.png', masked)
    cv2.imwrite('warped.png', warped)
    cv2.imwrite('output.jpg', cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR))

    plt.show(block=True)