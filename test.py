# Import the necessary packages 
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream
import matplotlib.pyplot as plt
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
#import winsound

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
CONSECUTIVE_FRAMES = 20

# Threshold for MAR value
MAR_THRESHOLD = 30

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0

# Alarm controls
ALARM_CTRL = False
WARN_CTRL = False
YAWN_WARN_CTRL = False

# Counters for alarms
ALARM_CNT = 0
WARN_CNT = 0
YAWN_WARN_CNT = 0

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

# Now start the video stream and allow the camera to warm-up
print("Starting Camera.....")
vs = VideoStream(usePiCamera=False).start()
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
				if not ALARM_CTRL:
					ALARM_CTRL = True
					playsound('sound files/alarm.mp3',0)
				# ALARM_CTRL = True
				# winsound.PlaySound('sound files/alarm.mp3', winsound.SND_ASYNC | winsound.SND_ALIAS )
				
				# Display Alert
				cv2.putText(frame, "CLOSED EYES ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else: 
			if FRAME_COUNT >= CONSECUTIVE_FRAMES:
				if not WARN_CTRL:
					WARN_CTRL = True
				#playsound('sound files/warning.mp3',0)
				# winsound.PlaySound('sound files/warning.mp3', winsound.SND_ASYNC | winsound.SND_ALIAS )
			FRAME_COUNT = 0

		cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MAR: {:.2f}".format(MAR), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Check if the person is yawning
		if MAR > MAR_THRESHOLD:
			count_yawn += 1

			# Change the contour color for lips to red
			cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 

			# Display Alert
			cv2.putText(frame, "YAWNING ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
			# if not ALARM_CTRL:
			if not ALARM_CTRL:
				ALARM_CTRL = True
				#playsound('sound files/alarm.mp3',0)

			if not YAWN_WARN_CTRL:
				YAWN_WARN_CTRL = True
				#playsound('sound files/warning_yawn.mp3',0)
			# winsound.PlaySound('sound files/alarm.mp3', winsound.SND_ASYNC | winsound.SND_ALIAS )
			# winsound.PlaySound('sound files/warning_yawn.mp3', winsound.SND_ASYNC | winsound.SND_ALIAS )
	if YAWN_WARN_CTRL:
		YAWN_WARN_CNT = YAWN_WARN_CNT + 1
	
	if ALARM_CTRL:
		ALARM_CNT = ALARM_CNT + 1

	if WARN_CTRL:
		WARN_CNT = WARN_CNT + 1

	if ALARM_CNT > 60:
		ALARM_CNT = 0
		ALARM_CTRL = False

	if WARN_CNT > 60:
		WARN_CNT = 0
		WARN_CTRL = False

	if YAWN_WARN_CNT > 60:
		YAWN_WARN_CNT = 0
		YAWN_WARN_CTRL = False

	cv2.imshow("Output", frame)
	key = cv2.waitKey(1) & 0xFF
	# winsound.PlaySound(None, winsound.SND_ASYNC)
	# ALARM_CTRL = False
	
	if key == ord('q'):
		break
