import numpy as np
import matplotlib as mpl
import cv2
import imutils

vid=cv2.VideoCapture("video.mp4")
y=1

template = cv2.imread("template.PNG")
#cv2.imshow("adi",template)

hsvtemp = cv2.cvtColor(template,cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsvtemp],[0],None,[180],[0,180]) 
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 

while True:
    b,frame=vid.read()
    if b==False:
        break
    else:
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
	dst = cv2.erode(dst, None, iterations=4) #Cleans up the image
	dst = cv2.dilate(dst, None, iterations=4)
	t,dst = cv2.threshold(dst,3,255,cv2.THRESH_BINARY)
	cv2.imshow("dst",dst) 
	cnts = cv2.findContours(dst.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	for c in cnts:
		if cv2.contourArea(c)>1000:
			#hull = cv2.convexHull(c)
			cv2.drawContours(frame,[c],-1,(0,255,0),3)
	cv2.imshow("hey",frame)
    	#if cv2.waitKey(1) & 0xFF==ord('q'):
        #	break
	cv2.waitKey(0)
cv2.destroyAllWindows()
