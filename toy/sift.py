import cv2
import os
import numpy as np

test_pic = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png'
test_path = '/Users/zora/Github/CellDetector/overlay_data/'
image = cv2.cvtColor(cv2.imread(os.path.join(test_path, test_pic)), cv2.COLOR_BGR2GRAY)
# image: color: either 0 or 255

# Not available in opencv3
#sift = cv2.xfeatures2d.SIFT_create()
#kp, des = sift.detectAndCOmpute(image, None)
#img = cv2.drawKeypoints(image, kp, image)
#cv2.simshow('sift', img)