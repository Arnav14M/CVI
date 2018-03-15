import numpy as np
import matplotlib as mpl
import cv2
import imutils

orangelower=(0,69,200)
orangeupper=(80,165,255)

whiteu=(222,222,222)
whitel=(122,122,122)

vid=cv2.VideoCapture("vid2.avi")
y=1


d={(0,50):[],(50,100):[],(100,150):[],(150,200):[],(200,250):[],(250,300):[],(300,350):[],(350,400):[],(400,450):[],(450,500):[],(500,550):[],(550,600):[]}

while True:
    d={(0,50):[],(50,100):[],(100,150):[],(150,200):[],(200,250):[],(250,300):[],(300,350):[],(350,400):[],(400,450):[],(450,500):[],(500,550):[],(550,600):[]}
    b,frame=vid.read()
    if not b:
        break
    frame = imutils.resize(frame, width=600)
    mask=cv2.inRange(frame,orangelower,orangeupper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    mask2 = cv2.inRange(frame,whitel,whiteu)
    mask2 = cv2.dilate(mask2, None, iterations=3)
    mask = cv2.erode(mask, None, iterations=5)
    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        for c in cnts:      
            x,y,w,h = cv2.boundingRect(c)
            midx=x+h/2
            for i in d:
                if midx>i[0] and midx<i[1]:
                    if not d[i]:
                        d[i]=[y,y+h]
                    else:
                        if y<d[i][0]:
                            d[i][0]=y
                        if y+h>d[i][1]:
                            d[i][1]=y+h
    for i in d:
        if d[i]:
            cv2.rectangle(frame,(i[0],d[i][0]),(i[1],d[i][1]),(0,0,0),2)

    for c in cnts2:
        x,y,w,h = cv2.boundingRect(c)
        if y>50:
            cv2.drawContours(frame, [c], -1, (255,255,255), -1)

    #for c in cnts2:
        #x,y,w,h = cv2.boundingRect(c)
        #if w<15 and h<30:
            #cv2.circle(frame,(x-w/2,y-h/2),(w+h)/2,(0,0,255),-1)
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),-1)
##            if x>300:
##                cv2.line(frame,(x-w/4,y-h/4),(x+w+w/4,y+h+h/4),(0,0,0),7)
##            else:
##                cv2.line(frame,(x+w+w/4,y-h/4),(x-w/4,y+h+h/4),(0,0,0),7)
    cv2.imshow('fr',frame)
    cv2.imshow('mas',mask2)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
#webcam.release()
cv2.destroyAllWindows()
