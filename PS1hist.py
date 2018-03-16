import numpy as np
import matplotlib as mpl
import cv2
import imutils

vid=cv2.VideoCapture("video.mp4")
y=1
whitelower = (122,122,122)
whiteupper= (222,222,222)
template = cv2.imread("template.PNG")
#cv2.imshow("adi",template)
kernel = np.ones((5,5),np.uint8)
hsvtemp = cv2.cvtColor(template,cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsvtemp],[0],None,[180],[0,180]) 
cv2.normalize(roi_hist,roi_hist,0,180,cv2.NORM_MINMAX)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 

while True:
	b,frame=vid.read()
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
	t,dst = cv2.threshold(dst,0,255,cv2.THRESH_BINARY) 
	dst = cv2.erode(dst, None, iterations=5) #Cleans up the image
	dst = cv2.dilate(dst, None, iterations=5)
	mask2 = cv2.inRange(frame,whitelower,whiteupper)
	mask2 = cv2.dilate(mask2, None, iterations=5)
	mask2 = cv2.erode(mask2, None, iterations=5)
	cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	dst = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, kernel)
	#cv2.imshow("dst",dst) 
	cnts = cv2.findContours(dst.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	for c in cnts:
		if cv2.contourArea(c)>2000:
			hull = cv2.convexHull(c)
			#epsilon = 0.001*cv2.arcLength(c,True)
    		#c = cv2.approxPolyDP(c,epsilon,True)
			cv2.drawContours(frame,[c],-1,(255,0,255),3)
	for c in cnts2:			
		x,y,w,h = cv2.boundingRect(c)
		if(y>50) and cv2.contourArea(c)>25:
			cv2.drawContours(frame,[c],-1,(255,0,255),-1)
	cv2.imshow("hey",frame)
    	if cv2.waitKey(1) & 0xFF==ord('q'):
        	break
	#cv2.waitKey(0)
cv2.destroyAllWindows()
