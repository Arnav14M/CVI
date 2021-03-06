import numpy as np
import matplotlib as mpl
import cv2
import imutils

vid=cv2.VideoCapture("video.mp4")
y=1
#whitelower = (130,130,130)
#whiteupper= (200,200,200)
whitelower = (10,0,130) #Dont mess with these values
whiteupper = (80,90,255)
t,frame = vid.read()
cv2.imwrite("checkrgb.jpeg",frame)
template = cv2.imread("template.PNG")
#cv2.imshow("adi",template)
kernel = np.ones((5,5),np.uint8)
hsvtemp = cv2.cvtColor(template,cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsvtemp],[0,2],None,[180,256],[0,180,0,256]) 
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 
while True:
	b,frame=vid.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#cv2.imshow("lkj",hsv)
	dst = cv2.calcBackProject([hsv],[0,2],roi_hist,[0,180,0,256],1)
	t,dst = cv2.threshold(dst,0,255,cv2.THRESH_BINARY) 
	arr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,arr,dst)
	dst = cv2.erode(dst, None, iterations=5) #Cleans up the image
	dst = cv2.dilate(dst, None, iterations=5)
	mask2 = cv2.inRange(hsv,whitelower,whiteupper)
	cv2.imshow("adw",mask2)
	mask2 = cv2.erode(mask2, None, iterations=2)
	cv2.imshow("wer",mask2)
	mask2 = cv2.dilate(mask2, None, iterations=5)
	cv2.imshow("wey",mask2)
	cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	cnts = cv2.findContours(dst.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	for c in cnts:
		if cv2.contourArea(c)>2000:
			#c = cv2.convexHull(c)
			epsilon = 0.005*cv2.arcLength(c,True)
    		#fesfw
			#c = cv2.approxPolyDP(c,epsilon,True)
			cv2.drawContours(frame,[c],-1,(255,0,255),3)
	for c in cnts2:			
		x,y,w,h = cv2.boundingRect(c)
		if(y>50) and cv2.contourArea(c)>1:
			cv2.drawContours(frame,[c],-1,(255,0,255),-1)
	cv2.imshow("hey",frame)
    	if cv2.waitKey(1) & 0xFF==ord('q'):
        	break
	#cv2.waitKey(0)
cv2.destroyAllWindows()
