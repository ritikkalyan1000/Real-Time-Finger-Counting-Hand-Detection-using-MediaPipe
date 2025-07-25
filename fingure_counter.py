import cv2
import mediapipe as mp

import numpy as np


cap = cv2.VideoCapture(0)
cap.set(3,1200)
cap.set(4,1200)

mphands = mp.solutions.hands
hand = mphands.Hands()

mpdraw = mp.solutions.drawing_utils

specs = mpdraw.DrawingSpec(thickness=1 ,circle_radius = 4,color= (255,255,255))

while True:
    _,img = cap.read()
    
    img = cv2.flip(img,1)
    list = []
    count = 0
    finger_tip = [8,12,16,20]
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    results = hand.process(imgrgb)
    
    if(results.multi_hand_landmarks):
        
        # print(results.multi_hand_landmarks)
        for idx,handlm in enumerate(results.multi_hand_landmarks):
            mpdraw.draw_landmarks(img,handlm,mphands.HAND_CONNECTIONS,specs,specs)
            h,w ,_ = img.shape
            
            for id,val in enumerate(handlm.landmark):  # one by one each points of the hand
                cx = int(val.x*w)
                cy = int(val.y*h)
                cv2.putText(img,f"{id}",(cx,cy-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                
                list.append([id,cx,cy])
        
        if(list[8][2]<list[6][2]):
            count = count + 1
            cv2.putText(img,f"{list[8][0]}",(list[8][1],list[8][2]-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        if(list[12][2]<list[10][2]):
            count = count + 1
            cv2.putText(img,f"{list[12][0]}",(list[12][1],list[12][2]-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        if(list[16][2]<list[14][2]):
            count = count + 1
            cv2.putText(img,f"{list[16][0]}",(list[16][1],list[16][2]-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        if(list[20][2]<list[18][2]):
            count = count + 1
            cv2.putText(img,f"{list[20][0]}",(list[20][1],list[20][2]-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            
        thumb_two_x = list[2][1]
        thumb_two_y = list[2][2]
        thumb_four_x = list[4][1]
        thumb_four_y = list[4][2]
        
        length = np.hypot(thumb_four_x-thumb_two_x , thumb_four_y-thumb_two_y)
        
        if(length > 110):
            count = count+1
            
        print(length)
        if(results.multi_handedness[idx].classification[0].label == "Left"):
            # if(length > 120):
            #     count = count+1
                # cv2.putText(img,f"{list[4][0]}",(list[4][1],list[4][2]-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
                cv2.putText(img,"left",(list[4][1],list[4][2]-40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
        if(results.multi_handedness[idx].classification[0].label == "Right"):
            # if(list[4][1] < list[2][1]):
                    
                # count = count+1
                # cv2.putText(img,f"{list[4][0]}",(list[4][1],list[4][2]-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
                # cv2.putText(img,f"{list[4][1]}",(list[4][1]-20,list[4][2]-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
                # cv2.putText(img,f"{list[2][1]}",(list[4][1]-70,list[4][2]),cv2.FONT_HERSHEY_COMPLEX,0.5,(200,0,255),1)
                cv2.putText(img,"right",(list[4][1],list[4][2]-40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
                
        cv2.rectangle(img,(50,50),(250,250),(255,255,255),-1)
        cv2.putText(img,f"{count}",(100,200),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,4,(255,0,0),4)  
        # print(count)
                    
                    
        
        
    cv2.imshow("window1",img)
    
    if(cv2.waitKey(1) & 0xff ==ord("x")):
        break

