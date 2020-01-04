from build import libnvjpeg
import numpy as np
import cv2
import time


pic_path = '../imgs/720P.jpg'
decoder = libnvjpeg.py_NVJpegDecoder()
pic = decoder.imread(pic_path)

print('nvjpeg')
st = time.time()
iters = 100
for i in range(iters):
    pic = decoder.imread(pic_path)
st = time.time() - st
print(st/iters, "sec.")
print(pic.shape)

#cv2.imshow("", pic)
#cv2.waitKey(0)

print('opencv')
st = time.time()
for i in range(iters):
    pic = cv2.imread(pic_path)
st = time.time() - st
print(st/iters, "sec.")

