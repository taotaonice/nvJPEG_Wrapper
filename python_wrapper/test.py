from build import libnvjpeg
import numpy as np


pic_path = '../imgs/720P.jpg'
decoder = libnvjpeg.NVJpegDecoder()
pic = decoder.imread(pic_path)
