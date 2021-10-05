import pygame
from pygame import mixer
import os
import pathlib


PROJECT_PATH=pathlib.Path(__file__).parent.resolve()

fileName = "audio/fineAlzateLaterali.mp3"
fileName2 = "Mediapipe-Python/audio/outcomeAlzateLaterali.mp3"
file = os.path.join(PROJECT_PATH, fileName)
fil2 = os.path.join(PROJECT_PATH, fileName2)

pygame.init()
mixer.init()
mixer.music.load("Mediapipe-Python/audio/fineAlzateLaterali.mp3")
mixer.music.play()


while pygame.mixer.music.get_busy():
	pygame.time.Clock().tick(10)


