"""CI 用：產生一段沒有人的合成影片（repo 不附真實影片）。用法：python make_test_clip.py out.mp4"""
import sys

import cv2
import numpy as np

w = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
for i in range(60):
    f = np.full((360, 640, 3), 230, np.uint8)
    cv2.circle(f, (80 + i * 8, 180), 30, (50, 50, 200), -1)
    w.write(f)
w.release()
