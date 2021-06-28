import cv2
import mediapipe as mp
import numpy as np
import sys
import pathlib
import os
import pickle


#funzione privata perchè ha il doppio underscore davanti
def __extract_keypoints(results):
	pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
	return pose

def createAnnotation(nomeCartella):
	mp_drawing = mp.solutions.drawing_utils
	mp_pose = mp.solutions.pose

	pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

	PROJECT_PATH=pathlib.Path(__file__).parent.resolve() #restituisce il path del progetto

	ALL_ES_PATH=os.path.join("data\\exercise\\",nomeCartella)
	SPEC_ES_PATH=os.path.join(PROJECT_PATH,ALL_ES_PATH)

	list_subfolders_with_paths = [f.path for f in os.scandir(SPEC_ES_PATH) if f.is_dir()]

	for directory in list_subfolders_with_paths:
		#directory=FULL_VIDEO_PATH
		print("sto processando i video in: "+ directory )
		for file in os.listdir(directory):
			filename = os.fsdecode(file)

			if filename.endswith(".mp4"):
				# print(os.path.join(directory, filename))
				#nome_video=filename
				videoP=os.path.join(directory,filename)
				cap = cv2.VideoCapture(videoP)

				if cap.isOpened()== False:
					print("Error opening video stream or file")
					raise TypeError

				frame_width = int(cap.get(3))
				frame_height = int(cap.get(4))

				outVideo=directory
				dirOutputPKL=os.path.join(outVideo,"annotated_PKL")
				dirOutputAnnotatedVideo=os.path.join(outVideo,"annotated_VIDEO")


				try:
					os.makedirs(dirOutputPKL)
					os.makedirs(dirOutputAnnotatedVideo)
				except:
					#print("cartelle per video e pkl annotated già presenti")
					pass

				#inputflnm = filename
				inflnm, inflext = filename.split('.')
				out_filename = f'{dirOutputAnnotatedVideo}\{inflnm}_annotated.{inflext}'
				out_filename_landmark = f'{dirOutputPKL}\{inflnm}_annotated.pkl'

				out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))
				land_list=[]
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
					out.write(image)

					frame_pose=__extract_keypoints(results)
					land_list.append(frame_pose)

				with open(out_filename_landmark, 'wb') as outfile:
					pickle.dump(land_list, outfile, pickle.HIGHEST_PROTOCOL)


			else:
				print("file non mp4")


	pose.close()
	cap.release()
	out.release()
