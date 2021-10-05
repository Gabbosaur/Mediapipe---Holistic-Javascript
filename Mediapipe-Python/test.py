from playsound import playsound
import os
import pathlib

PROJECT_PATH=pathlib.Path(__file__).parent.resolve()
fileName = "Mediapipe-Python/audio/fineAlzateLaterali.mp3"
fileName2 = "Mediapipe-Python/audio/outcomeAlzateLaterali.mp3"
file = os.path.join(PROJECT_PATH, fileName)
fil2 = os.path.join(PROJECT_PATH, fileName2)


playsound(fileName)

playsound(fileName2)
# playsound("fineAlzateLaterali.mp3")



