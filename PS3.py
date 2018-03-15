#Code for PS3

import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",help="path to the (optional) video file")
args = vars(ap.parse_args())

if not args.get("video", False):
	cap = cv2.VideoCapture(0) 
else:
	cap = cv2.VideoCapture(args["video"]) 

r,h,c,w = 155,20,250,20
for i in xrange(5):
	ret,frame = cap.read()
check = 0
def draw_rect(event,x,y,flags,param):
	global check,r,c,h,w
	if(check == 0):
		if event == cv2.EVENT_LBUTTONDOWN:
			r,c = y,x 
		elif event == cv2.EVENT_LBUTTONUP:
			w = x-c
			h = y-r
			cv2.rectangle(frame,(c,r),(x,y),(90,255,0),1)  
			check = 1
			
cv2.namedWindow('img2')
cv2.setMouseCallback('img2',draw_rect)
while(1):
	cv2.imshow('img2',frame)
	cv2.waitKey(1)
	if(check==1):
		break
while(1):
	if check==0:
		cv2.imshow('img2',img)
	else:
		break

track_window = (c,r,w,h) 
roi = frame[r:r+h,c:c+w]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180]) 
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 
ret,frame = cap.read()
    
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) 

while(1):
	ret = False
    	ret,frame = cap.read()
    	if ret==True:
        	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        	dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) 
        	ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        	x,y,w,h = track_window
        	img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        	cv2.imshow('img2',img2)
        	k = cv2.waitKey(20) & 0xff
        	if k == ord('q'):
            		break
        	else:
            		cv2.imwrite(chr(k)+".jpg",img2)
    	elif(check==1):
        	break
	else:
		continue
cv2.destroyAllWindows()
cap.release()


