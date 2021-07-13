##Here Color Detection and tracking are used in order to achieve the objective.
##The color marker is detected and a mask is produced.
##It includes the further steps of morphological operations on the mask produced which are Erosion and Dilation.
##Erosion reduces the impurities present in the mask and dilation further restores the eroded main mask.

##Motivation
##The initial motivation was a need for a dustless class room for the students to study in.
##I know that there are many ways like touch screens and more but what about the schools which can’t afford it to buy such huge large screens and teach on them like a T.V.
##So, I thought why not can a finger be tracked, but that too at a initial level without deep learning.
##Hence it was OpenCV which came to the rescue for these computer vision projects.






import numpy as np
from collections import deque
import cv2

#cv2.namedWindow("Color Detectors")
def setValues(x):
    print("")

#Creating trackbars needed for adjusting the marker color

#We are using HSV instead of RGB format because working in image and video frames in HSV is much more convinient and easier
# use trackbar to select ur marker color, it is blue right now
##cv2.createTrackbar("Upper Hue", "Color Detectors", 153, 180, setValues)
##cv2.createTrackbar("Lower Hue", "Color Detectors", 255, 255, setValues)
##cv2.createTrackbar("Upper Saturation", "Color Detectors", 255, 255, setValues)
##cv2.createTrackbar("Lower Saturation", "Color Detectors", 64, 189, setValues)
##cv2.createTrackbar("Upper Value", "Color Detectors", 72, 255, setValues)
##cv2.createTrackbar("Lower Value", "Color Detectors", 49, 255, setValues)

#Giving different arrays to handle color points of diff color
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

#assigning index values
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

kernel = np.ones((5,5),np.uint8)   #to properly detect the marker
#OpenCV blurs an image by applying what's called a Kernel.
#A Kernel tells you how to change the value of any given pixel by combining it with different amounts of the neighboring pixels.
#The kernel is applied to every pixel in the image one-by-one to produce the final image (this operation known as a convolution).

#paint window
colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255)]
paint_window = np.zeros((471,636,3),dtype=np.uint8) + 255
#cv2.imshow("Paint Window",paint_window)
paint_window = cv2.rectangle(paint_window, (40,1), (140,65), (0,0,0), 2)  #clear box
cv2.putText(paint_window, "CLEAR", (64,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
paint_window = cv2.rectangle(paint_window, (160,1), (255,65), (255,0,0), -1, cv2.LINE_AA)
cv2.putText(paint_window, "BLUE", (185,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
paint_window = cv2.rectangle(paint_window, (275,1), (370,65), (0,255,0), -1, cv2.LINE_AA)
cv2.putText(paint_window, "GREEN", (298,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
paint_window = cv2.rectangle(paint_window, (390,1), (485,65), (0,0,255), -1)
cv2.putText(paint_window, "RED", (420,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
paint_window = cv2.rectangle(paint_window, (505,1), (595,65), (0,255,255), -1)
cv2.putText(paint_window, "YELLOW", (519,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
#cv2.imshow("Paint window",paint_window)

cv2.namedWindow("Paint",cv2.WINDOW_AUTOSIZE)
colorIndex = 0

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    #flipping the frame to prevent lateral inversion
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    '''u_hue = cv2.getTrackbarPos("Upper Hue", "Color Detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color Detectors")
    u_sat = cv2.getTrackbarPos("Upper Saturation", "Color Detectors")
    l_sat = cv2.getTrackbarPos("Lower Saturation", "Color Detectors")
    u_val = cv2.getTrackbarPos("Upper Value", "Color Detectors")
    l_val = cv2.getTrackbarPos("Lower Value", "Color Detectors")

    upper_hsv = np.array([u_hue, u_sat, u_val])    #if webcam find the hsv value of the object btw upper and lower hsv range, it will identify that as marker marker colour blue
    lower_hsv = np.array([l_hue, l_sat, l_val])'''

    lower_hsv = np.array([110, 50, 50])    #if webcam find the hsv value of the object btw upper and lower hsv range, it will identify that as marker marker colour blue
    upper_hsv = np.array([130, 255, 255])

    lower_hsv = np.array([90, 50, 70])    #if webcam find the hsv value of the object btw upper and lower hsv range, it will identify that as marker marker colour blue
    upper_hsv = np.array([128, 255, 255])


    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)  #clear box
    cv2.putText(frame, "CLEAR", (64,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), -1, cv2.LINE_AA)    #blue
    cv2.putText(frame, "BLUE", (185,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), -1, cv2.LINE_AA)     #green
    cv2.putText(frame, "GREEN", (298,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), -1)           #red
    cv2.putText(frame, "RED", (420,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.rectangle(frame, (505,1), (595,65), (0,255,255), -1)         #yellow
    cv2.putText(frame, "YELLOW", (519,33), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    #removing noise so that marker could properly be traced
    mask = cv2.inRange(hsv,lower_hsv,upper_hsv)
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.dilate(mask,kernel,iterations=1)
    
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #continuous boundary points
    center = None      #center of the circle enclosing our marker
    
    #since we can have multiple marker color objects, hence multiple contours we will be getting
    #we will sort the contours by size and will fix the contour with the largest area as our marker
    if(len(contours)>0):
        cnt = sorted(contours, key = cv2.contourArea, reverse=True)[0]
        #Now we want the radius of the enclosing circle around the largest found contour
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)
        #Now drawing the circle around the contour
        cv2.circle(frame,(int(x),int(y)),int(radius),(0,0,0),2)
        M = cv2.moments(cnt)   #centroid of an arbitary shape is the avg of all points of the shape
                               #image moment is a particular weighted average of image pixel intensities, using which we can find radius,area,etc
        center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))

        #checking if any button above the screen is clicked or hovered
        if(center[1] <= 65):    #if y ccordinate of marker's center is less than 65, since from y=65 there are color blocks in our paint window
            if(40 <= center[0] <= 140):     #lies in clear block
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paint_window[67:,:,:] = 255
                
            elif(160 <= center[0] <= 255):
                colorIndex = 0  #blue
            elif(275 <= center[0] <= 370):
                colorIndex = 1  #green
            elif(390 <= center[0] <= 485):
                colorIndex = 2  #red        
            elif(505 <= center[0] <= 595):
                colorIndex = 3 #yellow
        else:
            if(colorIndex == 0):
                bpoints[blue_index].appendleft(center)
            elif(colorIndex == 1):
                gpoints[green_index].appendleft(center)
            elif(colorIndex == 2):
                rpoints[red_index].appendleft(center)
            elif(colorIndex == 3):
                ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if(points[i][j][k-1] is None or points[i][j][k] is None):
                    continue
                cv2.line(frame, points[i][j][k-1], points[i][j][k], colors[i], 2)
                cv2.line(paint_window, points[i][j][k-1], points[i][j][k], colors[i], 2)
    cv2.imshow("Tracking",frame)
    cv2.imshow("Paint",paint_window)
    cv2.imshow("mask",mask)
    
    k = cv2.waitKey(1) & 0xff
    if(k==27):
        break
    
cap.release()
cv2.destroyAllWindows()






