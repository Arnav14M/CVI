#Import necessary libraries
import numpy as np
import matplotlib as mpl
import cv2
import imutils

y = 1

vid=cv2.VideoCapture("video.mp4")
whitelower = (130,130,130) #Ranges for detecting white lines in BGR
whiteupper= (200,200,200)
template = cv2.imread("template.PNG") #Template for recognising cannisters
kernel = np.ones((5,5),np.uint8) 
hsvtemp = cv2.cvtColor(template,cv2.COLOR_BGR2HSV) #HSV image of template
roi_hist = cv2.calcHist([hsvtemp],[0,2],None,[180,256],[0,180,0,256]) #Histogram of template's H and V values
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX) #Normalise histogram between 0 and 255

while True:
	b,frame=vid.read() #Read from video 
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Convert frame to HSV
	dst = cv2.calcBackProject([hsv],[0,2],roi_hist,[0,180,0,256],1) #Back project template's histogram over frame
	t,dst = cv2.threshold(dst,0,255,cv2.THRESH_BINARY) #dst stores the probability distribution of a pixel belonging to template
	arr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)) 
	cv2.filter2D(dst,-1,arr,dst) #Convolves dst with arr and gets a more smoother image
	dst = cv2.erode(dst, None, iterations=5) #Cleans up the image
	dst = cv2.dilate(dst, None, iterations=5) #Makes image smoother
	mask2 = cv2.inRange(frame,whitelower,whiteupper) #Creates a mask for the white lines
	mask2 = cv2.dilate(mask2, None, iterations=5) 
	mask2 = cv2.erode(mask2, None, iterations=5)
	#arr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
	#cv2.filter2D(mask2,-1,arr,mask2) #Adds more noise
	cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	cnts = cv2.findContours(dst.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

	for c in cnts:
		if cv2.contourArea(c)>2000:
			#c = cv2.convexHull(c)
			#epsilon = 0.005*cv2.arcLength(c,True)
			#c = cv2.approxPolyDP(c,epsilon,True)
			cv2.drawContours(frame,[c],-1,(255,0,255),3)
	for c in cnts2:			
		x,y,w,h = cv2.boundingRect(c)
		if(y>50) and cv2.contourArea(c)>100:  #To identify only the required white lines
			cv2.drawContours(frame,[c],-1,(255,0,255),-1)
	cv2.imshow("MainIm",frame)
    	if cv2.waitKey(1) & 0xFF==ord('q'): #Exit by pressing q
        	break
cv2.destroyAllWindows()
