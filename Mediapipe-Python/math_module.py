import numpy as np
import math
import sklearn.metrics as metrics
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#x_schiena=[]
#y_schiena=[]


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
	asse_x=list(range(0,len(dist)))
	asse_x=np.array(asse_x)
	'''
	plt.xlabel("numero frame")
	plt.ylabel("distanza")
	plt.plot(asse_x,dist,label="schiena")
	plt.legend(loc="best")

	axes = plt.gca()
	axes.set_ylim([0,1])
	plt.show()
	#######fine plot
	'''
	global x_schiena,y_schiena
	#x_schiena=[],y_schiena=[]
	x_schiena=asse_x
	y_schiena=dist

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
	'''
	print("dist destra: "+str(dist_destra))
	print("dist sinistra: "+str(dist_sinistra))
	'''

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
	'''
	print(coeff_destra)
	print(coeff_sinistra)
	'''
	########plot
	asse_x=list(range(0,len(dist_sinistra)))
	asse_x=np.array(asse_x)
	'''
	plt.xlabel("numero frame")
	plt.ylabel("distanza")
	plt.plot(asse_x,dist_sinistra,label="braccio sinistro")
	plt.plot(asse_x,dist_destra,label="braccio destro")
	plt.legend(loc="best")

	axes = plt.gca()
	axes.set_ylim([0,1])
	plt.show()
	#######fine plot
	'''
	global x_b,y_b_sinistro,y_b_destro
	
	x_b=asse_x
	y_b_sinistro=dist_sinistra
	y_b_destro=dist_destra

	return min(coeff_destra,coeff_sinistra)

def braccia_tese_angolo(x):
	somma_angolo_gomito_destro=0
	somma_angolo_gomito_sinistro=0
	angolo_gomito_destro=[]
	angolo_gomito_sinistro=[]

	for i in range(len(x)):
		angolo_gomito_destro.append(calculate_angle([x[i][48],x[i][49]], [x[i][56],x[i][57]], [x[i][64],x[i][65]]))
		angolo_gomito_sinistro.append(calculate_angle([x[i][44],x[i][45]], [x[i][52],x[i][53]], [x[i][60],x[i][61]]))

		somma_angolo_gomito_destro=somma_angolo_gomito_destro+angolo_gomito_destro[i]
		somma_angolo_gomito_sinistro=somma_angolo_gomito_sinistro+angolo_gomito_sinistro[i]

	angolo_medio_gomito_destro=somma_angolo_gomito_destro/(180*len(x))
	angolo_medio_gomito_sinistro=somma_angolo_gomito_sinistro/(180*len(x))

	########plot
	asse_x=list(range(0,len(angolo_gomito_destro)))
	asse_x=np.array(asse_x)
	'''
	plt.xlabel("numero frame")
	plt.ylabel("distanza")
	plt.plot(asse_x,angolo_gomito_destro,label="angolo gomito destro")
	plt.plot(asse_x,angolo_gomito_sinistro,label="angolo gomito sinistro")
	plt.legend(loc="best")

	axes = plt.gca()
	axes.set_ylim([0,200])
	plt.show()
	#######fine plot
	'''
	global x_ang,y_angolo_gomito_sinistro,y_angolo_gomito_destro
	
	x_ang=asse_x
	y_angolo_gomito_sinistro=angolo_gomito_sinistro
	y_angolo_gomito_destro=angolo_gomito_destro


	return angolo_medio_gomito_destro,angolo_medio_gomito_sinistro

# Ritorna la somma delle distanze tra un braccio e l'altro, più è basso il valore meglio è
def mov_braccia_simmetrico(x):
	distanzaX=0
	distanzaY=0
	for i in range(len(x)):
		com_x=(x[i][48]+x[i][44])/2 #x del centro di massa
		#dist_X=abs(x[i][64]-x[i][60])

		dist_X=abs(abs(x[i][64]-com_x)-abs(x[i][60]-com_x))
		dist_Y=abs(x[i][65]-x[i][61])

		distanzaX=distanzaX+dist_X
		#print("x:"+str(distanzaX))
		distanzaY=distanzaY+dist_Y
		#print("y:"+str(distanzaY))
	return distanzaX,distanzaY

def angolo_massimo_spalla(x):
	max_angolo_destro=0
	max_angolo_sinistro=0
	ang_spall_d=[]
	ang_spall_s=[]

	for i in range(len(x)):
		#siccome l'anca e la spalla non sono sulla stessa asse verticale(stessa x) utiliziamo il valore di x della spalla anche per la x dell'anca
		ang_destro=calculate_angle([x[i][48],x[i][97]],[x[i][48],x[i][49]],[x[i][56],x[i][57]])
		ang_sinistro=calculate_angle([x[i][44],x[i][93]],[x[i][44],x[i][45]],[x[i][52],x[i][53]])
		ang_spall_d.append(ang_destro)
		ang_spall_s.append(ang_sinistro)
		if ang_sinistro>max_angolo_sinistro:
			max_angolo_sinistro=ang_sinistro
		if ang_destro>max_angolo_destro:
			max_angolo_destro=ang_destro

	global x_spalla,angolo_destro_spalla,angolo_sinistro_spalla
	asse_x=list(range(0,len(ang_spall_d)))
	
	x_spalla=np.array(asse_x)
	angolo_destro_spalla=ang_spall_d
	angolo_sinistro_spalla=ang_spall_s

	return max_angolo_destro,max_angolo_sinistro

def print_graph():
	#fig, axs = plt.subplots(2, 2)
	fig=plt.figure()
	gs = fig.add_gridspec(2, 2)
	
	ax1=plt.subplot(gs[0,0])
	ax2=plt.subplot(gs[0,1])
	ax3=plt.subplot(gs[1,0])
	ax4=plt.subplot(gs[1,1])

	ax1.plot(x_schiena, y_schiena,label="schiena")
	ax1.set_title("schiena")
	ax1.set_xlabel("numero frame")
	ax1.set_ylabel("distanza")
	ax1.legend(loc="best")
	ax1.set_ylim([0,1])


	ax2.plot(x_b, y_b_sinistro,label="braccio sinistro")
	ax2.plot(x_b, y_b_destro,label="braccio destro")
	ax2.set_title("estensione braccia")
	ax2.set_xlabel("numero frame")
	ax2.set_ylabel("dist spalla gomito")
	ax2.legend(loc="best")
	ax2.set_ylim([0,1])


	ax3.plot(x_ang, y_angolo_gomito_sinistro,label="angolo gomito sinistro")
	ax3.plot(x_ang, y_angolo_gomito_destro,label="angolo gomito destro")
	ax3.set_title("angolo dei gomiti")
	ax3.set_xlabel("numero frame")
	ax3.set_ylabel("angolo gomito")
	ax3.legend(loc="best")
	ax3.set_ylim([0,200])
	
	ax4.plot(x_spalla, angolo_sinistro_spalla,label="angolo spalla sinistra")
	ax4.plot(x_spalla, angolo_destro_spalla,label="angolo spalla destra")
	ax4.set_title("angolo delle spalle")
	ax4.set_xlabel("numero frame")
	ax4.set_ylabel("angolo spalla")
	ax4.legend(loc="best")
	ax4.set_ylim([0,200])



	fig.tight_layout()
	plt.show()

def calculate_feature_alzateLaterali(X):
	list_feature_X=[]

	'''
	0-->0-52 braccia piegate
	1-->53-95 braccia asimmetriche
	2-->96-141 no 90 gradi
	3-->142-187 ok
	'''
	'''
	import itertools as it
	a=10
	b=70
	c=110
	d=180
	for i in it.chain(range(a, a+1), range(b, b+1),range(c,c+1),range(d,d+1)):
	'''
	for i in range(len(X)):
		
		coeff_schiena=schiena_dritta(X[i])
		#print("coeff schiena: "+ str(coeff_schiena))

		coeff_braccia=braccia_tese(X[i])
		#print("coeff braccia: "+str(coeff_braccia))

		coeff_angolo_medio_braccio_destro,coeff_angolo_medio_braccio_sinistro=braccia_tese_angolo(X[i])
		#print("angolo medio destro:"+str(coeff_angolo_medio_braccio_destro))
		#print("angolo medio sinistro:"+str(coeff_angolo_medio_braccio_sinistro))

		coeff_simm_x,coeff_simm_y=mov_braccia_simmetrico(X[i])
		#print("coefficiente simmetria asse x: "+str(coeff_simm_x))
		#print("coefficiente simmetria asse x: "+str(coeff_simm_y))

		angolo_spalla_destra,angolo_spalla_sinistra=angolo_massimo_spalla(X[i])
		#print("max angolo spalla destra: "+str(angolo_spalla_destra))
		#print("max angolo spalla sinistra: "+str(angolo_spalla_sinistra))

		#print_graph()

		list_feature_X.append([coeff_schiena,coeff_braccia,coeff_angolo_medio_braccio_destro,coeff_angolo_medio_braccio_sinistro,coeff_simm_x,coeff_simm_y,angolo_spalla_destra,angolo_spalla_sinistra])
	
	return np.array(list_feature_X)


def confusionMatrix(y_test, y_pred, actions):
	print("\nCONFUSION MATRIX\n")

	# print(y_pred)
	# print(y_pred.argmax(axis=1))
	
	# Model Accuracy, how often is the classifier correct?
	# print("metrics.accuracy score 1D (non tiene conto dei null):\t",metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
	print("metrics.accuracy score normale (considera i null sbagliati) - Testing score:\t",metrics.accuracy_score(y_test, y_pred))

	matrix = [[0 for x in range(len(actions))] for y in range(len(actions))]
	errori = 0
	predNull = 0
	for i in range(0,len(y_test)):
		if 1 in y_pred[i]:
			if(np.argmax(y_pred[i]) != np.argmax(y_test[i])):
				matrix[np.argmax(y_test[i])][np.argmax(y_pred[i])]=matrix[np.argmax(y_test[i])][np.argmax(y_pred[i])] + 1
				errori=errori+1
			#print("valore predetto per campione "+ str(i)+ ": "+str(actions[np.argmax(y_pred[i])])) #prediction
			#print("valore effettivo per campione "+ str(i)+ ": "+str(actions[np.argmax(y_test[i])])+"\n") #valore effettivo
			else:
				matrix[np.argmax(y_test[i])][np.argmax(y_pred[i])]=matrix[np.argmax(y_test[i])][np.argmax(y_pred[i])] + 1
		else:
			errori+=1
			predNull+=1

	

	mat=pd.DataFrame(np.row_stack(matrix))
	col=[]
	for i in range(len(actions)):
		col.append(actions[i])

	mat.columns=col
	mat.index=actions

	print("numero campioni di test: "+str(len(y_pred))+"   campioni erroneamente classificati: "+str(errori) + "   campioni classificati nulli: " + str(predNull) + "\n")
	print(mat)

	# print("\nmetrics.confusion_matrix")
	# print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))




def oneHot_to_1D(y):

	y_final = []
	for i in range(0,len(y)):
		for j in range(0, 4):
			if y[i,j] == 1:
				y_final.append(j)
				break

	return y_final

def oneD_to_oneHot(y):
	y_final = []
	for i in range(0,len(y)):
		if y[i] == 0:
			y_final.append([1,0,0,0])
		elif y[i] == 1:
			y_final.append([0,1,0,0])
		elif y[i] == 2:
			y_final.append([0,0,1,0])
		else:
			y_final.append([0,0,0,1])
	return y_final



def conversione_dataset_al(x):
	X=pd.DataFrame(np.row_stack(x))
	X.columns=["coeff_schiena", "coeff_braccia", "coeff_angolo_medio_braccio_destro", "coeff_angolo_medio_braccio_sinistro", "coeff_simm_x", "coeff_simm_y", "angolo_spalla_destra", "angolo_spalla_sinistra"]
	return X

def split(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

	pd.DataFrame(X_train).to_csv('X_train_dec_tree.csv',index=False)
	pd.DataFrame(X_test).to_csv('X_test_dec_tree.csv',index=False)
	pd.DataFrame(y_train).to_csv('y_train_dec_tree.csv',index=False)
	pd.DataFrame(y_test).to_csv('y_test_dec_tree.csv',index=False)

	return X_train, X_test, y_train, y_test

def load_model(mod_file):

	loaded_model = pickle.load(open(mod_file, 'rb'))
	return loaded_model

def load_split_model(mod_file=None):
	X_train = pd.read_csv('X_train_dec_tree.csv')
	X_test=pd.read_csv('X_test_dec_tree.csv')
	y_train=pd.read_csv('y_train_dec_tree.csv')
	y_test=pd.read_csv('y_test_dec_tree.csv')

	#loaded_model=load_model(mod_file)

	#score = cross_val_score(loaded_model, X_train, y_train, cv=5)
	# print("cross val score DT:" + str(score))

	return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()#, loaded_model