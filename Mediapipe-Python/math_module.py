import numpy as np
import math
from matplotlib import pyplot as plt

def calculate_angle(a, b, c):
	a = np.array(a)  # First
	b = np.array(b)  # Mid
	c = np.array(c)  # End

	radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
		a[1] - b[1], a[0] - b[0]
	)
	angle = np.abs(radians * 180.0 / np.pi)

	if angle > 180.0:
		angle = 360 - angle

	return angle

def schiena_dritta(x):
	dist=[]
	'''
	print(len(x))#numero di frame
	print(len(x[0]))#132
	print(x[0])
	'''

	for i in range(len(x)):
		#x[i][1]= asse y del naso per ogni frame
		#x[i][93]= asse y del marker 23(anca sinistra) 
		#x[i][97]= asse y del marker 24(anca destra)
		d=x[i][1]-((x[i][93] - x[i][97])/2)
		dist.append(d)

	dist_max=max(dist)
	numerator=0

	for i in range(len(dist)):
		numerator= numerator+(dist[i]/dist_max)

	########plot
	x=list(range(0,len(dist)))
	x=np.array(x)

	plt.xlabel("numero frame")
	plt.ylabel("distanza")
	plt.plot(x,dist,label="schiena")
	plt.legend(loc="best")

	axes = plt.gca()
	axes.set_ylim([0,1])
	plt.show()
	#######fine plot

	return numerator/len(dist)

def braccia_tese(x):
	dist_destra=[]
	dist_sinistra=[]
	'''
	print(len(x))#numero di frame
	print(len(x[0]))#132
	print(x[0])
	'''

	for i in range(len(x)):
		#x[i][48]= asse x del marker 12(spalla destra) 
		#x[i][49]= asse y del marker 12(spalla destra) 
		#x[i][64]= asse x del marker 16(polso destra)
		#x[i][65]= asse y del marker 16(polso destra)

		#x[i][44]= asse x del marker 11(spalla sinistra) 
		#x[i][45]= asse y del marker 11(spalla sinistra)
		#x[i][60]= asse x del marker 15(polso sinistra)
		#x[i][61]= asse y del marker 15(polso sinistra)

		#formula distanza tra 2 punti
		d_destra=math.sqrt((pow((x[i][48]-x[i][64]),2)+pow((x[i][49]-x[i][65]),2)))
		d_sinistra=math.sqrt((pow((x[i][44]-x[i][60]),2)+pow((x[i][45]-x[i][61]),2)))
		dist_destra.append(d_destra)
		dist_sinistra.append(d_sinistra)

	print("dist destra: "+str(dist_destra))
	print("dist sinistra: "+str(dist_sinistra))
	dist_max_destra=max(dist_destra)
	dist_max_sinistra=max(dist_sinistra)
	numerator_destra=0
	numerator_sinistra=0

	'''
	dist_s=0
	for j in range(len(dist_destra)):
		dist_s=dist_destra[j]+dist_s
	dist_media=dist_s/len(dist_destra)
	print("dist media: "+str(dist_media/dist_max_destra))
	'''


	for i in range(len(dist_destra)):
		numerator_destra= numerator_destra+(dist_destra[i]/dist_max_destra)
		numerator_sinistra= numerator_sinistra+(dist_sinistra[i]/dist_max_sinistra)

	coeff_destra=numerator_destra/len(dist_destra)
	coeff_sinistra=numerator_sinistra/len(dist_sinistra)

	print(coeff_destra)
	print(coeff_sinistra)

	########plot
	x=list(range(0,len(dist_sinistra)))
	x=np.array(x)

	plt.xlabel("numero frame")
	plt.ylabel("distanza")
	plt.plot(x,dist_sinistra,label="braccio sinistro")
	plt.plot(x,dist_destra,label="braccio destro")
	plt.legend(loc="best")

	axes = plt.gca()
	axes.set_ylim([0,1])
	plt.show()
	#######fine plot

	return min(coeff_destra,coeff_sinistra)



def mov_braccia_simmetrico(x):
	distanzaX=0
	distanzaY=0
	for i in range(len(x)):
		com_x=max(x[i][48],x[i][44]) - min((x[i][48],x[i][44])) #x del centro di massa
		#dist_X=abs(x[i][64]-x[i][60])

		dist_X=abs(abs(x[i][64]-com_x)-abs(x[i][60]-com_x))
		dist_Y=abs(x[i][65]-x[i][61])

		distanzaX=distanzaX+dist_X
		print("d:"+str(distanzaX))
		distanzaY=distanzaY+dist_Y
	print("distanza su asse x:"+str(distanzaX))
	print("distanza su asse y:"+str(distanzaY))
	return

def angolo_massimo_spalla(x):
	max_angolo_destro=0
	max_angolo_sinistro=0
	for i in range(len(x)):
		#siccome l'anca e la spalla non sono sulla stessa asse verticale(stessa x) utiliziamo il valore di x della spalla anche per la x dell'anca
		ang_destro=calculate_angle([x[i][48],x[i][97]],[x[i][48],x[i][49]],[x[i][56],x[i][57]])
		ang_sinistro=calculate_angle([x[i][44],x[i][93]],[x[i][44],x[i][45]],[x[i][52],x[i][53]])
		if ang_sinistro>max_angolo_sinistro:
			max_angolo_sinistro=ang_sinistro
		if ang_destro>max_angolo_destro:
			max_angolo_destro=ang_destro

	return max_angolo_destro,max_angolo_sinistro

def calculate_feature_alzateLaterali(X):
	list_feature_X=[]

	#for i in range(len(X)):
	for i in range(90,91):
		coeff_schiena=schiena_dritta(X[i])
		print("coeff schiena: "+ str(coeff_schiena))

		coeff_braccia=braccia_tese(X[i])
		print("coeff braccia: "+str(coeff_braccia))

		coeff_mov_braccia=mov_braccia_simmetrico(X[i])

		angolo_spalla_destra,angolo_spalla_sinistra=angolo_massimo_spalla(X[i])
		print("max angolo spalla destra: "+str(angolo_spalla_destra))
		print("max angolo spalla sinistra: "+str(angolo_spalla_sinistra))

		list_feature_X.append([coeff_schiena,coeff_braccia,coeff_mov_braccia,angolo_spalla_destra,angolo_spalla_sinistra])
	
	return np.array(list_feature_X)