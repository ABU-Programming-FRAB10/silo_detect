import cv2
import numpy as np
from util import get_limits, read_fps, change_brightness, crop
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('WIN_20231024_13_51_16_Pro.mp4')
# fps=cap.get(cv2.CAP_PROP_FPS)
#lower_range=np.array([])
#upper_range=np.array([])
red = [0,0,255]
lower_purple = np.array([000,188,48])
upper_purple = np.array([255,255,218])
frame_count = 0
grahp = []

lower_red = np.array([132,140,000])
upper_red = np.array([255,255,255])



def purple(img):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_purple_1 = 0
    count_purple_2 = 0
    count_purple_3 = 0

    fps = read_fps(frame_count)
    grahp.append(fps)
    for c in cnts: 
        x=100
        if cv2.contourArea(c)>x:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            if x > 170 and y < 170:
                count_3 += 1
            if 170 > x > 83 and y < 170:
                count_2 += 1
            if  x <= 83 and y < 170:
                count_1 += 1
            if  x <= 83 and y < 480:
                count_purple_1 += 1
            if  170 > x > 83 and y < 480:
                count_purple_2 += 1
            if x > 170 and y < 480:
                count_purple_3 += 1
            # cv2.putText(frame,("DETECT-redball"),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"purple_{3} : {count_3}",(180,30),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
    cv2.putText(frame,f"purple_{2} : {count_2}",(100,30),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)    
    cv2.putText(frame,f"purple_{1} : {count_1}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
    
    return count_purple_1,count_purple_2,count_purple_3            
    #         cv2.putText(frame,("DETECT-purpleball"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    # cv2.putText(frame,f"total : {count}",(200,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    # cv2.putText(frame,f"fps : {fps:.2f}",(300,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

def Red(img):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # lowerLimit, upperLimit = get_limits(color=[0,0,255])
    # print(lowerLimit, upperLimit)
    mask=cv2.inRange(hsv,lower_red,upper_red)
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(mask1, cnts, -1, (140, 135, 000), 2) 


    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_red_1 = 0
    count_red_2 = 0
    count_red_3 = 0
    fps = read_fps(frame_count)
    grahp.append(fps)
    for c in cnts:
            
        x=100
        if cv2.contourArea(c)>x:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            if x > 170 and y < 170:
                count_3 += 1
            if 170 > x > 83 and y < 170:
                count_2 += 1
            if  x <= 83 and y < 170:
                count_1 += 1
            if  x <= 83 and y < 480:
                count_red_1 += 1
            if  170 > x > 83 and y < 480 :
                count_red_2 += 1
            if x > 170 and y < 480:
                count_red_3 += 1
            # cv2.putText(frame,("DETECT-redball"),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"red_{3} : {count_3}",(180,60),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
    cv2.putText(frame,f"red_{2} : {count_2}",(100,60),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)    
    
    cv2.putText(frame,f"red_{1} : {count_1}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
    return count_red_1,count_red_2,count_red_3
    # cv2.putText(frame,f"red_{n} : {count}",(180,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    # cv2.putText(frame,f"fps : {fps:.2f}",(270,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

try:
    while True:
        ret,frame=cap.read()
        frame=cv2.resize(frame,(640,480))
        frame = crop(frame, 200, 0, 300, 480)
        cv2.line(frame, (0, 175 ), (frame.shape[1], 175), (255,255,255), thickness=2)
        cv2.line(frame, (83, 0 ), (83, frame.shape[1]), (255,255,255), thickness=2)
        cv2.line(frame, (170, 0 ), (170, frame.shape[1]), (255,255,255), thickness=2)
        # frame = change_brightness(frame, value=30)

        # hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # mask=cv2.inRange(hsv,lower_red,upper_red)
        # _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
        # cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(mask1, cnts, -1, (140, 135, 000), 2) 
        count = 0
        Red(frame)
        purple(frame)
        count_red_1,count_red_2,count_red_3 = Red(frame)
        count_purple_1,count_purple_2,count_purple_3 = purple(frame)
        frame_count += 1
        fps = read_fps(frame_count)
        grahp.append(fps)
        print(f"purple{count_purple_1,count_purple_2,count_purple_3}, red{count_red_1,count_red_2,count_red_3}")
        # cv2.putText(frame,f"total_{1} : {count_red_1+count_purple_1}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1)
        # cv2.putText(frame,f"total_{2} : {count_red_2+count_purple_2}",(100,80),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1)
        # cv2.putText(frame,f"total_{3} : {count_red_3+count_purple_3}",(180,80),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1)

        cv2.imshow("FRAME",frame)
            
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
    plt.plot(grahp)
    plt.xlabel("frame")
    plt.ylabel("fps")
    plt.title("fps")
    plt.savefig('line_plot.png')
    plt.show()
    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    plt.plot(grahp)
    plt.xlabel("frame")
    plt.ylabel("fps")
    plt.title("fps")
    plt.savefig('line_plot.png')
    plt.show()
