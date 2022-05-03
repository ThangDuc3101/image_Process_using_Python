import cv2
import numpy as np
from matplotlib import pyplot as plt

# read the image
img = cv2.imread("Lenna.png")
img2 = img.copy()
# convert orther colorspace
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# thresholding
ret,imgThres=cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY)
ret,imgThresINV=cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY_INV)

# get width&height of image
h=img.shape[0]
w=img.shape[1]

# smoothing image
imgBlur=cv2.blur(img,(5,5))
imgMedianBlur=cv2.medianBlur(img,5)

# Image Gradient
imgLaplacian=cv2.Laplacian(imgGray,cv2.CV_64F)
# imgLaplacian=cv2.Laplacian(img,cv2.CV_64F)
imgSobelX=cv2.Sobel(imgGray,cv2.CV_64F,1,0,ksize=9)
imgSobelY=cv2.Sobel(imgGray,cv2.CV_64F,0,1,ksize=9)

# canny edge
imgCanny = cv2.Canny(img,100,255)

# contour
contours,hierarchy=cv2.findContours(imgThres,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,255,0),2)

# rect bounding
x,y,w,h = cv2.boundingRect(contours[0])
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# rotated rect
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)

# histogram
plt.plot(cv2.calcHist(img2,[0],None,[256],[0,256]))
plt.show()

# histogram equalization
equ = cv2.equalizeHist(img2)
cv2.show("Image with Histgram Eualization",equ)
# plt.plot(cv2.calcHist(img2,[0],None,[256],[0,256]))
# plt.show()

# show image
cv2.imshow("Image",img)
cv2.imshow("Image Gray",imgGray)
cv2.imshow("Image HSV",imgHSV)
cv2.imshow("Image RGB",imgRGB)
cv2.imshow("Image Thresholding",imgThres)
cv2.imshow("Image ThresholdingINV",imgThresINV)
cv2.imshow("Image Blur",imgBlur)
cv2.imshow("Image Median Blur",imgMedianBlur)
cv2.imshow("Image Gradient with Laplacian",imgLaplacian)
cv2.imshow("Image Gradient with SobelX",imgSobelX)
cv2.imshow("Image Gradient with SobelY",imgSobelY)
cv2.imshow("Image Canny Edge Detection",imgCanny)
cv2.imshow("rectangle bounding",img)
cv2.imshow("Image with rotate rectangle",img)


if(cv2.waitKey(0)==ord('q')):
    cv2.destroyAllWindow()