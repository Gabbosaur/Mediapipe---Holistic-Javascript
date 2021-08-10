import os
import cv2
import mediapipe as mp
import numpy as np
import math_module

def __extract_keypoints(results):
	pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
	return pose


def run(filename):
    directory = "C:/Users/Orphe/Desktop"
    videoP = os.path.join(directory,filename)
    cap = cv2.VideoCapture(videoP)
    if cap.isOpened()== False:
        print("Error opening video stream or file")
        raise TypeError

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    land_list = []

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_pose=__extract_keypoints(results)
        land_list.append(frame_pose)

    X = np.array(land_list, dtype="object")
    print("polso X: ",X[0][16*4])




run("1.mp4")
run("1flipped.mp4")