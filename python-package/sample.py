import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')
# img = cv2.imread("./t1.png")
img = cv2.imread("../../Cat-faces-dataset/dataset-part1/cat_0.png")
faces = app.get(img)
print(len(faces))
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)