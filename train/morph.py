
import numpy as np
import cv2
import sys
import timeit
import os
from scipy.spatial import Delaunay

from point_finder import PointFinder

def getTriangulation(pf):
    # detect face with haar cascade
    face = pf.detectFace()

    # find landmarks in the face with lbf model
    landmarks, convex, num_points = pf.detectLandmarks(face)

    # calculate delaunay triangulation. note: this is just 
    # indecies of vertecies not actually 
    # the vertex positions themselves
    delaunayTriangles = Delaunay(landmarks)
    triangluation = delaunayTriangles.simplices

    return triangluation, landmarks, pf.getImage(), convex

def morphTriangle(image, image1, outputImage, first, second, alpha) :
    r1 = cv2.boundingRect(np.float32(first))
    r2 = cv2.boundingRect(np.float32(second))

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

    cropped = image1[y1: y1 + h1, x1: x1 + w1]
    mask = np.zeros_like(cropped)

    cv2.fillConvexPoly(mask, np.int32(secondCropped), (1.0, 1.0, 1.0), 16, 0)

    cropped = image[y: y + h, x: x + w]
    warped = cv2.warpAffine(cropped, affine, (w1, h1))

    # remove "* ( 1 - mask )" to show lines
    outputImage[y1: y1 + h1, x1: x1 + w1] = outputImage[y1: y1 + h1, x1: x1 + w1] * ( 1 - mask ) + warped * mask

def morph(first, second, triangulation, image, image1, alpha):
    points = []

    # Compute weighted average point coordinates
    for i in range(0, len(first)):
        x = ( 1 - alpha ) * first[i][0] + alpha * second[i][0]
        y = ( 1 - alpha ) * first[i][1] + alpha * second[i][1]
        points.append((x,y))

    b, g, r = cv2.split(image)
    b1, g1, r1 = cv2.split(image1)
    
    outputImage = np.zeros(image.shape, dtype = image.dtype)

    outputImage[:, :, 0] = ((1 - alpha) * b) + (alpha * b1)
    outputImage[:, :, 1] = ((1 - alpha) * g) + (alpha * g1)
    outputImage[:, :, 2] = ((1 - alpha) * r) + (alpha * r1)

    for tri in triangulation:
        x,y,z = tri
        
        x = int(x)
        y = int(y)
        z = int(z)
        
        tri = [first[x], first[y], first[z]]
        tri_new = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morphTriangle(image, image1, outputImage, tri, tri_new, alpha)

    return np.uint8(outputImage)

if __name__ == "__main__":
    filename = 'test1.jpg'
    filename1 = 'test2.jpg'

    pf = PointFinder(filename)
    pf1 = PointFinder(filename1)

    # Read array of corresponding points
    triangulation, _, _, convex = getTriangulation(pf)
    _, _, _, convex1 = getTriangulation(pf1)

    image = np.float32(cv2.imread(filename))
    image1 = np.float32(cv2.imread(filename1))

    points1 = convex
    points2 = convex1

    height,width,layers= image.shape
    
    video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v') , 20,(width, height))

    for alpha in range(0, 10):
        video.write(morph(points1, points2, triangulation, image, image1, alpha/10))
    
    video.release()