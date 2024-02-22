import os
import sys

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import cv2

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

from IPython.display import display, Javascript, Image

from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL
import io
import html
import time

#Get Python and OpenCV Version

print('OpenCV-Python Lib Version:', cv2.__version__)
print('Python Version:',sys.version)

from google.colab import drive
drive.mount('/content/drive')

#Metal1 = cv2.imread('/content/drive/MyDrive/Colab Notebooks/Training Images/Metal/Metal_1.jpg', cv2.IMREAD_UNCHANGED)
Metal1 = cv2.imread('/content/drive/MyDrive/Colab Notebooks/Training Images/Metal/Metal_1.jpg', cv2.IMREAD_UNCHANGED)

plt.figure(figsize=(20,10))
#Note: matplotlib uses RGB format so had to convert BGR-to-RGB
plt.imshow(cv2.cvtColor(Metal1,cv2.COLOR_BGR2RGB))
plt.title('RGB Image',color='c')

Metal1_Gray = cv2.cvtColor(Metal1, cv2.COLOR_BGR2RGB)
#Metal1_Canny = cv2.Canny(Metal1_Gray, 50, 200)
#ret, Metal1_Binary = cv2.threshold(Metal1_Gray, 25, 255, cv2.THRESH_BINARY)
#contours, _ = cv2.findContours(Metal1_Binary, 1, 2)

def houghCircleDetector(img_path):
  img = cv2.imread(img_path)

  new = cv2.convertScaleAbs(img, alpha = 1.5, beta = 10)
  new = cv2.medianBlur(new, 5)

  new1 = new[0:,0:200].copy()
  new2 = new[0:,1100:].copy()

  img_edge1 = cv2.Canny(new1, 50, 200)
  img_edge2 = cv2.Canny(new2, 50, 200)

  circles1 = cv2.HoughCircles(img_edge1,cv2.HOUGH_GRADIENT,1,minDist=20,param1=10,param2=34)
  circles1 = np.uint16(np.around(circles1))
  circles2 = cv2.HoughCircles(img_edge2,cv2.HOUGH_GRADIENT,1,minDist=20,param1=10,param2=34)
  circles2 = np.uint16(np.around(circles2))

  for val in circles1[0,:]:
      cv2.circle(new1,(val[0],val[1]),val[2],(255,0,0),2)
  for val in circles2[0,:]:
      cv2.circle(new2,(val[0],val[1]),val[2],(255,0,0),2)

  plt.subplot(121), plt.imshow(cv2.cvtColor(new1,cv2.COLOR_BGR2RGB)), plt.title('Canny')
  plt.subplot(122), plt.imshow(cv2.cvtColor(new2,cv2.COLOR_BGR2RGB)), plt.title('Result')

  return

def detectShapes(img_path):
  img = cv2.imread(img_path)

  new = cv2.convertScaleAbs(img, alpha = 1.5, beta = 10)
  new = cv2.medianBlur(new, 5)

  new1 = new[0:,0:200].copy()
  new2 = new[0:,1100:].copy()

  img_edge1 = cv2.Canny(new1, 50, 200)
  img_edge2 = cv2.Canny(new2, 50, 200)

  circles1 = cv2.HoughCircles(img_edge1,cv2.HOUGH_GRADIENT,1,minDist=20,param1=10,param2=34)
  circles1 = np.uint16(np.around(circles1))
  circles2 = cv2.HoughCircles(img_edge2,cv2.HOUGH_GRADIENT,1,minDist=20,param1=10,param2=34)
  circles2 = np.uint16(np.around(circles2))

  for val in circles1[0,:]:
      cv2.circle(new1,(val[0],val[1]),val[2],(255,0,0),2)
  for val in circles2[0,:]:
      cv2.circle(new2,(val[0],val[1]),val[2],(255,0,0),2)

  _,new_binary1 = cv2.threshold(new1,25,255,cv2.THRESH_BINARY_INV)
  _,new_binary2 = cv2.threshold(new2,25,255,cv2.THRESH_BINARY_INV)

  contours,_ = cv2.findContours(new_binary1.copy(),1,2)
  for num,cnt in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt)
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    # print(num, approx)
    if len(approx) > 10:
      cv2.putText(new_binary1,"Circle",(int(x+w/2),int(y+h/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
      cv2.drawContours(new_binary1,[cnt],-1,(0,255,0),2)

    plt.figure(figsize=(20,10))
    plt.imshow(cv2.cvtColor(new_binary1,cv2.COLOR_BGR2RGB)), plt.title('Result')

    return

#Note: matplotlib uses RGB format so had to convert BGR-to-RGB

detectShapes('/content/drive/MyDrive/Colab Notebooks/Training Images/Metal/Metal_1.jpg')
#plt.imshow()
#plt.title('Canny Edge Detection',color='c')

#ret, Metal1_Binary = cv2.threshold(Metal1_Gray, 25, 255, cv2.THRESH_BINARY_INV)

plt.imshow(Metal1_Binary, cmap = 'binary')
plt.imshow(Metal2_Binary, cmap = 'binary')
plt.imshow(Metal3_Binary, cmap = 'binary')
plt.imshow(Metal4_Binary, cmap = 'binary')
plt.imshow(Metal5_Binary, cmap = 'binary')
plt.imshow(Metal6_Binary, cmap = 'binary')
