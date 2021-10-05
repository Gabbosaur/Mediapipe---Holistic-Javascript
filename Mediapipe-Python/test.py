from threading import Thread
import winsound

def play():

    play_thread = Thread(target=lambda: winsound.Beep(500,1200))
    play_thread.start()


play()

print("we")