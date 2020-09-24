import cv2
import os
import matplotlib.pyplot as plt
import urllib.request as urlreq
import numpy as np
from pylab import rcParams
from scipy.spatial import Delaunay

lbfmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

LBFmodel = "models/lbfmodel.yaml"
haarcascade = "models/haarcascade.xml"

def downloadModels():
    if (haarcascade[7:] in os.listdir(os.curdir + "/models")):
        print("Haar Cascade Already Downloaded")
    else:
        print("Haar Cascate not found. Downloading...")

        urlreq.urlretrieve(haarcascade_url, haarcascade)

        print("Finished Downloading")
    
    if (LBFmodel[7:] in os.listdir(os.curdir + "/models")):
        print("LBFModel Already Downloaded")
    else:
        print("LBF Model not found. Downloading...")

        urlreq.urlretrieve(lbfmodel_url, LBFmodel)

        print("Finished Downloading")

def getTriangulation(pic):
    def detectFace():
        image_grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

        detector = cv2.CascadeClassifier(haarcascade)
        faces = detector.detectMultiScale(image_grayscale)

        #draw face
        if(len(faces) > 0):
            face = faces[0]
            (x,y,w,d) = face

            rectImage = np.zeros((image.shape[0], image.shape[1], 4), np.uint8)
            rectImage[:, :, 3] = 0
            cv2.rectangle(rectImage, (x,y), (x+w, y+d), (255, 255, 255), 2)

            return np.array([face])
        
        print("Couldn't find a face")
        exit(-1)

    def detectLandmarks(face):
        image_grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)


        landmark_detector  = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(LBFmodel)

        _, landmarks = landmark_detector.fit(image_grayscale, face)

        landmarkArr = []
        points = []
        count = 0
        for landmark in landmarks:
            for x,y in landmark[0]:
                landmarkArr.append([x , y])
                points.append((x,y))

                plt.text(x, y, str(count))
            count += 1
        
        return landmarkArr, np.array(points, np.int32)

    image = cv2.cvtColor(cv2.imread(pic), cv2.COLOR_BGR2RGB)

    # image_grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

    # detect face with haar cascade
    face = detectFace()

    # find landmarks in the face with lbf model
    landmarks, convex = detectLandmarks(face)

    # calculate delaunay triangulation. note: this is just 
    # indecies of vertecies not actually 
    # the vertex positions themselves
    delaunayTriangles = Delaunay(landmarks)
    triangluation = delaunayTriangles.simplices

    return triangluation, landmarks, image, convex

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
    # downloadModels()
    pic1 = "2.jpg"
    pic2 = "1.jpg"

    # pic = cv2.imread(pic1)

    # plt.imshow(pic)

    # _, _, image, landmarks, = getTriangulation(pic1)

    # plt.imshow(image)

    # count = -1
    # for landmark in landmarks:
    #     count += 1
    #     for x,y in landmark:
    #         plt.text(x, y, count)
        

    tri1, landmarks, image, convex = getTriangulation(pic1)
    tri2, landmarks2, image1, convex = getTriangulation(pic2)

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