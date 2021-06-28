# pip install virtualenv
# virtualenv mypython

# attivare tesivenv
# tesivenv\Scripts\activate

# pip install mediapipe opencv-python


import cv2
import mediapipe as mp
import numpy as np
import math

#nostri moduli
import Tpose_module
import annotateData_module

#Tpose_module.Tpose()

#crea i video e i file pkl annotati dai file mp4 presenti nelle sottocartelle di alzateLaterali
annotateData_module.createAnnotation("alzateLaterali")