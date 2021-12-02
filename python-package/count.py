import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# 顔認識器の設定
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(64, 64))

# 編集ファイルを開く
path_r = '/home/babaamata/workspace/Cat-faces-dataset/dataset-part1/'

# カウント変数
image_count = 0
face_count = 0

files = os.listdir(path_r)
for file in files:
  if (file.find('.png')!=-1):
    img = cv2.imread(path_r + file.replace('\n', ''))
    faces = app.get(img)
    image_count = image_count + 1
    face_count = face_count + len(faces)
    if len(faces) > 0:
      rimg = app.draw_on(img, faces)
      cv2.imwrite("./output/part1/" + str(face_count)+ "_" + file + ".jpg", rimg)

print(image_count)
print(face_count)
