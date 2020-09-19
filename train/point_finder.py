import cv2
import os
import numpy as np
import urllib.request as urlreq

class PointFinder:
    def __init__(self, img):
        print("made point finder")
        self.create(image=img)
  
    def create(self, image):
        self.image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        
        haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
        lbfmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

        self.LBFmodel = "models/lbfmodel.yaml"
        self.haarcascade = "models/haarcascade.xml"

        self.downloadModels(haarcascade_url, lbfmodel_url)

        self.image_grayscale = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)

    def detectFace(self):
        detector = cv2.CascadeClassifier(self.haarcascade)
        faces = detector.detectMultiScale(self.image_grayscale)

        #draw face
        if(len(faces) > 0):
            face = faces[0]
            (x,y,w,d) = face

            # rectImage = np.zeros((image.shape[0], image.shape[1], 4), np.uint8)
            # rectImage[:, :, 3] = 0
            # cv2.rectangle(rectImage, (x,y), (x+w, y+d), (255, 255, 255), 2)

            return np.array([face])
        
        print("Couldn't find a face")
        exit(-1)

    def detectLandmarks(self, face):
        landmark_detector  = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(self.LBFmodel)

        _, landmarks = landmark_detector.fit(self.image_grayscale, face)

        landmarkArr = []
        points = []
        for landmark in landmarks:
            for x,y in landmark[0]:
                landmarkArr.append([x , y])
                points.append((x,y))
        
        return landmarkArr, np.array(points, np.int32)

    def downloadModels(self, haarcascade_url, lbfmodel_url):
        if (self.haarcascade[7:] in os.listdir(os.curdir + "/models")):
            print("Haar Cascade Already Downloaded")
        else:
            print("Haar Cascate not found. Downloading...")

            urlreq.urlretrieve(haarcascade_url, self.haarcascade)

            print("Finished Downloading")
        
        if (self.LBFmodel[7:] in os.listdir(os.curdir + "/models")):
            print("LBFModel Already Downloaded")
        else:
            print("LBF Model not found. Downloading...")

            urlreq.urlretrieve(lbfmodel_url, self.LBFmodel)

            print("Finished Downloading")
    
    def getImage(self):
        return self.image