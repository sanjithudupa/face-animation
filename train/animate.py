import numpy as np
import cv2
import sys
import timeit
import os
from scipy.spatial import Delaunay
import json

def getTriangulationFromFile(filename):
    triangluation_str = open("positions/triangulation/" + filename + ".txt", "r").read()
    convex_str = open("positions/convex/" + filename + ".txt", "r").read()

    triangulation = np.array([line.split(' ') for line in triangluation_str.split("\n")], np.int32)
    convex = np.array([line.split(' ') for line in convex_str.split("\n")], np.int32)

    return triangulation, convex

def getImage(viseme): 
    for key, value in viseme_groups.items(): 
         if viseme in value: 
             fname = key
             return fname
  
    return None

def morphTriangle(image, image1, outputImage, first, next1, second, alpha) :
    r1 = cv2.boundingRect(np.float32(first))
    r2 = cv2.boundingRect(np.float32(second))
    r3 = cv2.boundingRect(np.float32(second))

    (x, y, w, h) = r1
    (x1, y1, w1, h1) = r2
    (x2, y2, w2, h2) = r3

    firstCropped = np.array([[first[0][0] - x, first[0][1] - y], [first[1][0] - x, first[1][1] - y], [first[2][0] - x, first[2][1] - y]], np.int32)
    secondCropped = np.array([[second[0][0] - x1, second[0][1] - y1], [second[1][0] - x1, second[1][1] - y1], [second[2][0] - x1, second[2][1] - y1]], np.int32)
    nextCropped = np.array([[next1[0][0] - x2, next1[0][1] - y2], [next1[1][0] - x2, next1[1][1] - y2], [next1[2][0] - x2, next1[2][1] - y2]], np.int32)

    try:
        affine = cv2.getAffineTransform(np.float32(firstCropped), np.float32(secondCropped))
        # affine1 = cv2.getAffineTransform(np.float32(secondCropped), np.float32(firstCropped))
    except TypeError as t:
        print("errored on")
        print(type(first))
        print(type(second))

    # part to replace on second image
    cropped = image1[y1: y1 + h1, x1: x1 + w1]
    mask = np.zeros_like(cropped)


    # mask for part to replace on second image
    cv2.fillConvexPoly(mask, np.int32(secondCropped), (1.0, 1.0, 1.0), 16, 0)


    # # part to replace on second image
    # cropped1 = image[y: y + h, x: x + w]
    # mask1 = np.zeros_like(cropped1)


    # # mask for part to replace on second image
    # cv2.fillConvexPoly(mask1, np.int32(firstCropped), (1.0, 1.0, 1.0), 16, 0)

    # warped portion of first image
    cropped = image[y: y + h, x: x + w]
    warped = cv2.warpAffine(cropped, affine, (w1, h1))
    # warped1 = cv2.warpAffine(cropped1, affine1, (w1, h1))

    # warped = (1.0 - alpha) * warped + alpha * warped1

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
        tri1 = [second[x], second[y], second[z]]
        tri_new = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morphTriangle(image, image1, outputImage, tri, tri1, tri_new, alpha)

    return np.uint8(outputImage)

if __name__ == "__main__":

    with open("../research/viseme_groups.json") as f:
        viseme_groups = json.load(f)['viseme_groups']

    start = 'r'
    end = 'n'

    main_image = getImage(start)
    main_image1 = getImage(end)

    # Read array of corresponding points
    triangulation, convex = getTriangulationFromFile(start)
    triangulation1, convex1 = getTriangulationFromFile(end)

    image = np.float32(cv2.imread("positions/pictures/" + main_image + ".jpg"))
    image1 = np.float32(cv2.imread("positions/pictures/" + main_image1 + ".jpg"))

    points1 = convex
    points2 = convex1

    height,width,layers= image.shape
    
    video = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v') , 20,(width, height))

    for alpha in range(0, 10):
        # if(main_image == main_image1):
        video.write(morph(points1, points2, triangulation, image, image1, alpha/10))
        # else:
        #     video.write(morphDifferent(points1, points2, triangulation, triangulation1, image, image1, alpha/10))
    
    video.release()