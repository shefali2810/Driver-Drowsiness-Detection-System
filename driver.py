# Import the necessary packages 
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream
#import matplotlib.pyplot as plt
import matplotlib.animation as animate
from matplotlib import style 
import imutils 
import dlib
import time 
import argparse 
import cv2 
from playsound import playsound
from scipy.spatial import distance as dist
import os 
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import pygame

style.use('fivethirtyeight')

#all eye and mouth aspect ratio to be recorded with time
ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
ts=[]
total_ts=[]

# Threshold for EAR value, below which it will be regared as an eye blink
EAR_THRESHOLD = 0.2

# Consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 5

# Threshold for MAR value
MAR_THRESHOLD = 27

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0

# Creating the dataset 
def assure_path_exists(path):
    print("Verifying Directory.....")
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print("Creating Directory.....")
        os.makedirs(dir)


pygame.mixer.init()
pygame.mixer.music.load("sound files/alarm.mp3")

print("Loading the front face predictor.....")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Grab the indexes of the facial landamarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print(lstart, lend)
print(rstart, rend)
print(mstart, mend)

assure_path_exists("dataset_rpcam/")

# Now start the video stream and allow the camera to warm-up
print("Starting Camera.....")
vs = VideoStream(usePiCamera=True).start()
time.sleep(1)

count_sleep = 0
count_yawn = 0 

while True: 
    # Extract a frame 
    frame = vs.read()
    # Resize the frame 
    frame = imutils.resize(frame, width = 500)
    # Convert the frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces 
    rects = detector(frame, 1)

    # Now loop over all the face detections and apply the predictor 
    for (i, rect) in enumerate(rects): 
        shape = predictor(gray, rect)
        # Convert it to a (68, 2) size numpy array 
        shape = face_utils.shape_to_np(shape)

        # Draw a rectangle over the detected face 
        (x, y, w, h) = face_utils.rect_to_bb(rect) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)    
        # Put a number 
        cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend] 
        mouth = shape[mstart:mend]
        
        # Compute the EAR for both the eyes 
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Take the average of both the EAR
        EAR = (leftEAR + rightEAR) / 2.0
        
        #live datawrite in csv
        ear_list.append(EAR)
        # print(ear_list)
        

        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        
        # Compute the convex hull for both the eyes and then visualize it
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        # Draw the green contours 
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR/10)


        # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
        # Thus, count the number of frames for which the eye remains closed 
        if EAR < EAR_THRESHOLD: 
            FRAME_COUNT += 1

            # Change the contour color for eyes to red
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            # When eyes are closed for a longer period
            if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                count_sleep += 1
                cv2.putText(frame, "CLOSED EYES ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite("dataset_rpcam/frame_sleep%d.jpg" % count_sleep, frame)
                if pygame.mixer.music.get_busy() == False:
                    pygame.mixer.music.play()
                # playsound('sound files/alarm.mp3')
                
        else:
            # if FRAME_COUNT >= CONSECUTIVE_FRAMES:
            #     playsound('sound files/alarm.mp3',0)
                # winsound.PlaySound('sound files/warning.mp3', winsound.SND_ASYNC | winsound.SND_ALIAS )
            FRAME_COUNT = 0

        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(MAR), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check if the person is yawning
        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            # playsound('sound files/alarm.mp3',0)
            # playsound('sound files/warning_yawn.mp3',0)
            # Change the contour color for lips to red
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
            # Display Alert
            cv2.putText(frame, "YAWNING ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite("dataset_rpcam/frame_yawn%d.jpg" % count_yawn, frame)
            if pygame.mixer.music.get_busy() == False:
                    pygame.mixer.music.play()
            # playsound('sound files/alarm.mp3')
            # winsound.PlaySound('sound files/alarm.mp3', winsound.SND_ASYNC | winsound.SND_ALIAS )
            # winsound.PlaySound('sound files/warning_yawn.mp3', winsound.SND_ASYNC | winsound.SND_ALIAS )

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
        
    if key == ord('q'):
        break