#pip install virtualenv
#virtualenv mypython

#attivare tesivenv
#tesivenv\Scripts\activate

# pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np

mp_drawing=mp.solutions.drawing_utils	#drawing utilities
mp_pose=mp.solutions.pose				#pose estimation model di mediapipe
'''
# VIDEO FEED
cap = cv2.VideoCapture(0)
while cap.isOpened():
	ret, frame = cap.read()
	cv2.imshow('Mediapipe Feed', frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):	#0xFF è la variabile che fa lo store del tasto premuto, in questo caso quando premiamo q viene chiuso il feed video
		break

cap.release()
cv2.destroyAllWindows()
'''

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
	while cap.isOpened():
		ret, frame = cap.read()

		# Recolor image to RGB
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image.flags.writeable = False

		# Make detection
		results = pose.process(image)

		# Recolor back to BGR
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# Render detections
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))	#color=(b,g,r)

		cv2.imshow('Mediapipe Feed', image)

		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
