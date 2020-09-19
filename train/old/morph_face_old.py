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

# def getTriangles(pic):
def getTriangulation(pic):
    def detectFace():
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
        quit()
    def detectLandmarks(face):
        landmark_detector  = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(LBFmodel)

        _, landmarks = landmark_detector.fit(image_grayscale, face)

        landmarkArr = []
        points = []
        for landmark in landmarks:
            for x,y in landmark[0]:
                # cv2.circle(image, (x, y), 1, (255, 255, 255), 1)
                landmarkArr.append([x , y])
                points.append((x,y))
        
        return landmarkArr, np.array(points, np.int32)

    image = cv2.cvtColor(cv2.imread(pic), cv2.COLOR_BGR2RGB)

    image_copy = image.copy()
    image_grayscale = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    face = detectFace()
    landmarks, convex = detectLandmarks(face)

    delaunayTriangles = Delaunay(landmarks)
    triangluation = delaunayTriangles.simplices

    # plt.triplot(np.array(landmarks)[:,0], np.array(landmarks)[:,1], triangluation)

    # plt.imshow(image)

    return triangluation, landmarks, image, convex
    
    # ts, ls = getTriangulation()

    # triangles = []

    # for t in ts:
    #     triangle = [ls[t[0]], ls[t[1]], ls[t[2]]]
    #     triangles.append(triangle)
    
    # print(triangles[0])
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image
def addAlpha(img, a):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * a
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def lerp(a, b, t):
    xm = (1-t) * a[0] + t * b[0]
    ym = (1-t) * a[1] + t * b[1]

    return [xm, ym]
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
    
    # rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    return rgb
def overlay(b, o):
    h = o.shape[0]
    w = o.shape[1]

    for y in range(h):
        for x in range(w):
            if(o[y, x][3] > 252 and not(o[y, x][0] > 250 or o[y, x][1] > 250 or o[y, x][2] > 250)):
                b[y, x] = o[y, x]

    
downloadModels()

pic1 = "2.jpg"
pic2 = "1.jpg"

tri1, landmarks, image, convex = getTriangulation(pic1)
tri2, landmarks2, image1, convex = getTriangulation(pic2)

"""THIS IS FOR GETTING MASK THING IF I NEED IT FOR SEAMLESS CLONE GET FROM HERE"""
convexhull = cv2.convexHull(convex)
face_mask = np.zeros_like(image)
face_mask = cv2.fillConvexPoly(face_mask, convexhull, (255, 255, 255))

no_face_mask = cv2.bitwise_not(face_mask)
no_face = cv2.bitwise_and(image, no_face_mask)

# outputImage = np.zeros((image.shape[0], image.shape[1], 4), np.uint8)
# # outputImage = addAlpha(image.copy(), 255)
# outputImage[:, :, 3] = 0

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
    
    # calculate affine transform
    # try:
    #     affine = cv2.getAffineTransform(first, second)
    # except TypeError as t:
    #     print("errored on")
    #     print(type(first))
    #     print(type(second))
    
    # # perform affine transform

    # draw triangle
    # cv2.line(image, tuple(first[0]), tuple(first[1]), (255, 255, 255), 3)
    # cv2.line(image, tuple(first[1]), tuple(first[2]), (255, 255, 255), 3)
    # cv2.line(image, tuple(first[2]), tuple(first[0]), (255, 255 ,255), 3)

    r1 = cv2.boundingRect(first)
    r2 = cv2.boundingRect(second)

    (x, y, w, h) = r1
    (x1, y1, w1, h1) = r2

    cropped = image[y: y + h, x: x + w]
    mask = np.zeros_like(cropped)

    print(cropped.shape)
    
    firstCropped = np.array([[first[0][0] - x, first[0][1] - y], [first[1][0] - x, first[1][1] - y], [first[2][0] - x, first[2][1] - y]], np.int32)
    cv2.fillConvexPoly(mask, firstCropped, 255)
    print(mask.shape)

    cropped = cv2.resize(cropped, mask.shape[1::-1])

    masked = applyMask(cropped, mask)

    secondCropped = np.array([[second[0][0] - x1, second[0][1] - y1], [second[1][0] - x1, second[1][1] - y1], [second[2][0] - x1, second[2][1] - y1]], np.int32)

    try:
        affine = cv2.getAffineTransform(np.float32(firstCropped), np.float32(secondCropped))
    except TypeError as t:
        print("errored on")
        print(type(first))
        print(type(second))

    warped = cv2.warpAffine(masked, affine, (w1, h1))

    #reconstruct final
    
    area = outputImage[y1: y1 + h1, x1: x1 + w1]
    # area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

    # _, mask_triangles_designed = cv2.threshold(area_gray, 0, 255, cv2.THRESH_BINARY)
    # warped = cv2.bitwise_and(warped, warped, mask=mask_triangles_designed)

    # area = cv2.add(area, warped)
    overlay(area, warped)
    # cv2.imshow("warped", area)
    # cv2.waitKey(0)

    outputImage[y1: y1 + h1, x1: x1 + w1] = area

    # outputImage[y1: y1 + h1, x1: x1 + w1] = area

    # final[y1: y1 + h1, x1: x1 + w1] = area

    # print(area.shape)
    
    # outputImage[y: y + h, x: x + w] = warped


    # r2 = cv2.boundingRect(second)



    # # cropped_image = image.copy()[y:y+h, x:x+w]
    # mask = np.zeros((image.shape[0], image.shape[1], 1), dtype = "uint8")
    # triCnt = np.array(tuple(first))
    # print(type(triCnt))
    # cv2.drawContours(image, [triCnt], 0, (0,255,0), -1)

    # cv2.fillPoly(mask, np.array([landmarks[tri[0]], landmarks[tri[1]], landmarks[tri[2]]]), (255, 255, 255), 8)
    # warp_dst = cv2.warpAffine(image, affine, (image.shape[1], image.shape[0]))


    # image_new = cv2.warpAffine(image, affine, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # cv2.imwrite('newimage.jpg', mask)

    # tri1Cropped = []
    # tri2Cropped = []

    # for i in range(0, 3):
    #     tri1Cropped.append(((first[0][i][0] - r1[0]),(first[0][i][1] - r1[1])))
    #     tri2Cropped.append(((second[0][i][0] - r2[0]),(second[0][i][1] - r2[1])))

    #     imageCropped = image[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    count += 1

#overlay image

# output_face_mask = np.zeros_like(image1)
# head_mask = cv2.fillConvexPoly(output_face_mask, convexhull, 255)
# head_mask = cv2.bitwise_not(head_mask)

# img2_head_noface = cv2.bitwise_and(image1, image1, mask=output_face_mask)
# result = cv2.add(img2_head_noface, outputImage)

# # final = output = cv2.seamlessClone(outputImage, image, blank, center, cv2.NORMAL_CLONE)

# b, g, r = cv2.split(image)
# alpha_channel = np.zeros(b.shape, dtype=b.dtype)

# final = cv2.merge((b, g, r, alpha_channel))

# b, g, r = cv2.split(image)

# final = np.zeros((image.shape[0], image.shape[1], 4), np.uint8)
# final[:, :, 0] = b
# final[:, :, 1] = g
# final[:, :, 2] = r
# final[:, :, 3] = 255

# # print(outputImage.shape)
# # print(final.shape)
# # final = cv2.add(final, outputImage)
# overlay(final, outputImage)
(x, y, w, h) = cv2.boundingRect(convexhull)
center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
# outputImage = cv2.rectangle(outputImage, (x, y), (x + h, y + h), (255, 0, 0), 3)

# seamlessclone = cv2.seamlessClone(outputImage, image, face_mask, center_face, cv2.NORMAL_CLONE)
# sem = cv2.seamlessClone(image1, image, face_mask, center_face, cv2.NORMAL_CLONE)

plt.imshow(outputImage)
cv2.imwrite('og.png', masked)
cv2.imwrite('warped.png', warped)

# cv2.waitKey(0)
cv2.imwrite('out2.png', cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGRA))

# print(first)
# print(second)

# plt.axis("off")
# plt.title('Face Detection')
plt.show(block=True)