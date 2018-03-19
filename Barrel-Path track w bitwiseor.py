'''
Press 'space' to quit
Program aims to highlight orange white striped barrels and the white path.
'''

import cv2
import numpy as np


#orig o_lo = ( 150, 30, 0 )[::-1]
o_lo = ( 150, 0, 0 )[::-1]
#orig o_up = ( 255, 140, 80 )[::-1]
o_up = ( 255, 140, 80 )[::-1]

cam = cv2.VideoCapture("video.mp4")

while cam.isOpened():

    grabbed, frame  = cam.read()    #Gets frame(img) and grabbed
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv = cv2.flip( hsv, 1 )
    masko = cv2.inRange( frame, o_lo, o_up )        #Orange Mask
    #masko = cv2.erode( masko, None, iterations = 1)
   # masko = cv2.dilate( masko, None, iterations = 1 )

    maskw = cv2.inRange( frame, (140,130,150)[::-1] ,(255,255,255)[::-1] )      #White mask
   # maskw = cv2.erode( maskw, None, iterations = 1)
   # maskw = cv2.dilate( maskw, None, iterations = 1 )

    mask = cv2.bitwise_or(maskw,masko)      #Combined Mask

    mask = cv2.erode( mask, None, iterations = 1)
    mask = cv2.dilate( mask, None, iterations = 2 )

    contours = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )[-2]
    # center = None

    for c in contours:
        if cv2.contourArea( c ) > 200:
            #x,y,w,h = cv2.boundingRect( c )
            #cv2.rectangle( frame, ( x, y ), ( x+w, y+h ), (0,0,255)[::-1], 2)      #For Straight rectangles

            rect = cv2.minAreaRect( c )             #For skewed rectangles
            points = cv2.boxPoints( rect )
            points = np.int0( points )
            cv2.drawContours( frame, [ points ], 0, (0,0,255)[::-1])


    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(10) == 32:
        break  # space to quit

cam.release()
cv2.destroyAllWindows()
