import numpy
import cv2
import os

test_path = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
test_pic = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png'
test_mask = '0adbf56cd182f784ca681396edc8b847b888b34762d48168c7812c79d145aa07.png'
# purple
test_path_2 = '3934a094e8537841e973342c7f8880606f7a2712b14930340d6f6c2afe178c25'
test_pic_2 = '3934a094e8537841e973342c7f8880606f7a2712b14930340d6f6c2afe178c25.png'

background = cv2.imread('/Users/zora/Github/CellDetector/test_data/stage1_train/%s/images/%s' %(test_path, test_pic))
foreground = cv2.imread('/Users/zora/Github/CellDetector/test_data/stage1_train/%s/masks/%s' %(test_path, test_mask))
# cv2.imshow('bg', background)#.astype(float)/255)
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
# background = background.astype(float)
# foreground = foreground.astype(float)

outImage = cv2.add(foreground, background)
cv2.imshow("outImg", outImage)
cv2.imshow("background", background)
cv2.imshow("foreground", foreground)
cv2.waitKey(0)
