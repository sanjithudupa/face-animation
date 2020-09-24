import cv2
import os
import numpy as np
import urllib.request as urlreq

class PointFinder:
    # these are points on the corners

    def __init__(self, img):
        print("made point finder")

        # percentages of where the cornerpoints should be
        self.corners = [(0, 0.85), (0.99, 0.75), (0, 0), (0, 0.5), (0, 0.99), (0.5, 0.99), (0.99, 0.99), (0.99, 0.5), (0.99, 0), (0.5, 0)]
        self.otherEdges = [(0, 0), (0, 0.5), (0, 1), (0.5, 0), (1, 0), (0.5, 1), (1, 1), (1, 0.5)]
        # self.pfdir = pfdir
        self.create(image=img)
  
    def create(self, image):
        try:
            self.image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        except SystemError as _:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # positions of actual corners in image
        self.corners = [(x * self.image.shape[1], y * self.image.shape[0]) for (x, y) in self.corners]
        self.otherEdges = [(x * self.image.shape[1], y * self.image.shape[0]) for (x, y) in self.otherEdges]

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

        imcop = self.image_grayscale.copy()

        landmarkArr = []
        points = []
        count = 0
        for landmark in landmarks:
            for x,y in landmark[0]:
                landmarkArr.append([x , y])
                points.append((x,y))
                imcop = cv2.putText(imcop, str(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                count += 1 

        for x, y in self.otherEdges:
            points.append((x, y))

        
        
        # cv2.imshow("hi:", imcop)
        # cv2.waitKey(0)

        self.num_points = len(points)
        
        
        return landmarkArr, np.array(points, np.int32), self.num_points

    def downloadModels(self, haarcascade_url, lbfmodel_url):
        if (self.haarcascade[7:] in os.listdir("models")):
            print("Haar Cascade Already Downloaded")
        else:
            print("Haar Cascate not found. Downloading...")

            urlreq.urlretrieve(haarcascade_url, self.haarcascade)

            print("Finished Downloading")
        
        if (self.LBFmodel[7:] in os.listdir("models")):
            print("LBFModel Already Downloaded")
        else:
            print("LBF Model not found. Downloading...")

            urlreq.urlretrieve(lbfmodel_url, self.LBFmodel)

            print("Finished Downloading")
    
    def getImage(self):
        return self.image