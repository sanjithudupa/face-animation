import cv2 
from scipy.spatial import Delaunay
import json

from point_finder import PointFinder

vid = cv2.VideoCapture(0) 

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

with open("../research/viseme_groups.json") as f:
  viseme_groups = json.load(f)['viseme_groups']

need_pics = list(viseme_groups.keys())
pictures = []

need_tri = []

for needed in need_pics:
    need_tri.extend(list(viseme_groups[needed]))

print(need_pics)
print(need_tri)

trianglulations = []

cur_pic = 0

while(True): 
    ret, frame = vid.read()

    picture = need_tri[cur_pic]

    cv2.imshow('Camera Feed', cv2.putText(frame.copy(), picture, (0, 25), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0), 2, cv2.LINE_AA)) 
    
    if cv2.waitKey(1) & 0xFF == ord('e'): 
        pf = PointFinder(frame)
        triangluation, landmarks, image, convex, = getTriangulation(pf)
        if picture in need_pics:
            cv2.imwrite("positions/pictures/" + picture + ".jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        triangulation_str = "\n".join([" ".join(str(dimension) for dimension in point) for point in triangluation])
        convex_str = "\n".join([" ".join(str(dimension) for dimension in point) for point in convex])

        traingulation_file = open("positions/triangulation/" + picture + ".txt","w") 
        traingulation_file.write(triangulation_str)
        traingulation_file.close()

        convex_file = open("positions/convex/" + picture + ".txt","w") 
        convex_file.write(convex_str)
        convex_file.close()

        print(triangulation_str)
        cur_pic += 1
  
vid.release() 
cv2.destroyAllWindows() 