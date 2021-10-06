
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import winsound
from threading import Thread

#nostri moduli
import math_module

"""
# VIDEO FEED
cap = cv2.VideoCapture(0)
while cap.isOpened():
	ret, frame = cap.read()
	cv2.imshow('Mediapipe Feed', frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):	#0xFF è la variabile che fa lo store del tasto premuto, in questo caso quando premiamo q viene chiuso il feed video
		break

cap.release()
cv2.destroyAllWindows()
"""
def __extract_keypoints(results):
	pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
	return pose

def playBeep(freq):
    play_thread = Thread(target=lambda: winsound.Beep(freq,200))
    play_thread.start()

def alzateLaterali_live(num_rep):

	mp_drawing = mp.solutions.drawing_utils  # drawing utilities
	mp_pose = mp.solutions.pose  # pose estimation model di mediapipe

	rep_counter=0 # contatore di ripetizioni anche se è sbagliato, ma deve superare i 30° di soglia
	flag_discesa=0
	flag_salita=0
	flag_fine_es=0
	flag_es_valido=0

	flag_scritta_partenza = 0
	flag_scritta_alzata = 0
	flag_scritta_tpose = 0
	flag_scritta_fine = 0

	flag_beep_tpose = 0
	flag_inizio_esercizio = 0


	record_movimento = []
	tutte_le_rep = []
	record = 0

	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	n_frame = 0

	out = cv2.VideoWriter('outputAlzateLaterali.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (640,480))

	## Setup mediapipe instance
	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		while cap.isOpened():
			start_time = time.time()
			n_frame=n_frame+1

			ret, frame = cap.read()

			# Recolor image to RGB
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False

			# Make detection
			results = pose.process(image)

			# Recolor back to BGR
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			# Extract landmarks
			try:
				landmarks = results.pose_landmarks.landmark

				cv2.putText(image,"BODY FOUND",
							(int(width/2)-25,15),
							cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA
				)
				# print(landmarks)

				#status box
				cv2.rectangle(image, (0,0), (250,170), (245,117,16), -1)

				"""
				ciclo ogni landmarks
				for lndmrk in mp_pose.PoseLandmark:
					print(lndmrk)

				accesso ai lendmark
				landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x oppure .y  .z  .visibility
				"""

				# get coordinates
				shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
				elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
				wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
				hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

				shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
				elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
				wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
				hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

				# calculate angles
				angle_elbow_left = math_module.calculate_angle(shoulder_left, elbow_left, wrist_left)
				angle_elbow_right = math_module.calculate_angle(shoulder_right, elbow_right, wrist_right)
				angle_shoulder_left = math_module.calculate_angle(hip_left, shoulder_left, elbow_left)
				angle_shoulder_right = math_module.calculate_angle(hip_right, shoulder_right, elbow_right)

				# Visualize angle
				cv2.putText(image,str(math.trunc(angle_elbow_left)) + "deg",
					tuple(np.multiply(elbow_left, [width, height-25]).astype(int)),
					cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA
				)

				cv2.putText(image,str(math.trunc(angle_elbow_right)) + "deg",
					tuple(np.multiply(elbow_right, [width-25, height-25]).astype(int)),
					cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA
				)

				cv2.putText(image,str(math.trunc(angle_shoulder_left)) + "deg",
							tuple(np.multiply(shoulder_left, [width, height-25]).astype(int)),
							cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA
				)

				cv2.putText(image,str(math.trunc(angle_shoulder_right)) + "deg",
					tuple(np.multiply(shoulder_right, [width, height-25]).astype(int)),
					cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA
				)

				#controlli per gli stati della tpose
				#################################################da main chiamamo questa funz, gli passiamo medel, e il numero di ripetizioni
				tollerance = 20

				if (angle_shoulder_left < 20 and angle_shoulder_right < 20 and angle_elbow_left >= (180 - tollerance) and angle_elbow_left <= (180 + tollerance) and angle_elbow_right >= (180 - tollerance) and angle_elbow_right <= (180 + tollerance)):
					record = 1
					if (flag_discesa == 0):
						flag_salita = 1
					# cv2.putText(image,"posizione di" + "\n" + "partenza riconosciuta",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA)
				else:
					if (angle_shoulder_left >= 80 or angle_shoulder_right >= 80):
						if flag_inizio_esercizio == 1:
							flag_scritta_tpose = 1

					if (flag_salita == 1):

						if (angle_shoulder_left >= 30 or angle_shoulder_right >= 30):
							flag_es_valido = 1
							# cv2.putText(image,"soglia validita' es superata",
							# 		(10,100),
							# 		cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA
							# )

							if (flag_discesa == 0):
								flag_salita = 0
								flag_discesa = 1

						else:
							None
					else:
						None

				# vincolo quando le braccia ritornano giù
				if (angle_shoulder_left < 15 and angle_shoulder_right < 15 and flag_es_valido==1 and flag_discesa==1):
					flag_discesa = 0
					# flag_salita = 0
					rep_counter = rep_counter+1
					flag_fine_es = 1
					record = 0

					flag_scritta_partenza = 0
					flag_scritta_alzata = 0
					flag_scritta_tpose = 0
					flag_scritta_fine = 1


				# if(flag_salita==1 or flag_discesa==1):
				# 	frame_pose=__extract_keypoints(results)
				# 	record_movimento.append(frame_pose)
				# 	print(".. salvando record movimento ..")

				if(record == 1):
					frame_pose=__extract_keypoints(results)
					record_movimento.append(frame_pose)
					print(".. salvando record movimento ..")
				else:
					if(flag_fine_es == 1):
						flag_fine_es=0


						'''
						PREDICTION IN REAL TIME, ma troppo lento --> usare i threads ???
						#calcolo feature
						print("... sto calcolando le features ...")
						feature_X=math_module.calculate_feature_alzateLaterali(record_movimento)
						print("featureX: " + feature_X)

						X=math_module.conversione_dataset_al(feature_X)

						prediction = model.predict(X) # array di probabilità. for example: [0 0.3 0.4 0.3] e prendiamo con argmax l'indice con il valore più alto
						print("prediction: " + prediction)
						print("valore predetto: " + str(actions[np.argmax(prediction)])) #prediction
						'''

						tutte_le_rep.append(record_movimento)
						record_movimento = []
						#tolgo return
						print("Fine esercizio.")

						if(rep_counter==num_rep):
							#testo:es completato
							#aspetta 2 secondi
							print("Chiusura Webcam.")
							cap.release()
							cv2.destroyAllWindows()
							return tutte_le_rep
							#passi le classificazioni a main con un return
						#return record_movimento
			except:
				pass


			# Reps data
			cv2.putText(image, 'REPS',
						(15,15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
			cv2.putText(image, str(rep_counter),
						(10,63),
						cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

			# Flag per le scritte laterali
			if (flag_salita == 0 and flag_scritta_partenza == 0):
				cv2.putText(image,"Posizione partenza",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA)
			else:
				if flag_scritta_partenza == 0:
					playBeep(500)
				flag_scritta_partenza = 1
				flag_inizio_esercizio = 1
				cv2.putText(image,"Posizione partenza",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,cv2.LINE_AA)



			if (flag_discesa == 0 and flag_scritta_alzata == 0):
				cv2.putText(image,"Fase di salita",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA)
			else:
				if flag_inizio_esercizio == 1:
					if flag_scritta_alzata == 0:
						playBeep(500)
					flag_scritta_alzata = 1 # flag_discesa diventa 1 quando supera i 30 gradi
					cv2.putText(image,"Fase di salita",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,cv2.LINE_AA)


			if (flag_scritta_tpose == 0):
				cv2.putText(image,"Posizione t-pose",(10,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA)
			else:
				if flag_inizio_esercizio == 1:
					if flag_beep_tpose == 0:
						playBeep(500)
						flag_beep_tpose = 1
					cv2.putText(image,"Posizione t-pose",(10,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,cv2.LINE_AA)

				else:
					cv2.putText(image,"Posizione t-pose",(10,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA)


			if (flag_scritta_fine == 0):
				cv2.putText(image,"Fase di discesa (fine " + str(rep_counter+1) + " rep)",(10,160),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA)
			else:
				if flag_inizio_esercizio == 1:
					playBeep(1000)
					cv2.putText(image,"Fase di discesa (fine " + str(rep_counter+1) + " rep)",(10,160),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,cv2.LINE_AA)
					flag_scritta_fine = 0 # inizializzo a 0
					flag_beep_tpose = 0
					flag_inizio_esercizio = 0


			# Render detections
			mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
				mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
				mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
			)  # color=(b,g,r)


			# Save video
			if ret==True:
				# write the flipped frame
				out.write(image)

			# else:
			# 	break



			cv2.imshow("Mediapipe Feed", image)
			cv2.setWindowProperty("Mediapipe Feed", cv2.WND_PROP_TOPMOST, 1)

			if cv2.waitKey(10) & 0xFF == ord("q"):
				break

			# print("--- n_frame: " + str(n_frame) + " --- " + "%s seconds ---" % (time.time() - start_time))

		cap.release()
		out.release()
		cv2.destroyAllWindows()

