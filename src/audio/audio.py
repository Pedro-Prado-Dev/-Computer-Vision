import threading

from abc import ABC
import pyttsx3


ENGINE = pyttsx3.init()
voices = ENGINE.getProperty('voices')
ENGINE.setProperty('voice', voices[1].id)


class Speaker(ABC):
    @staticmethod
    def _speak(text: str):
        lock = threading.Lock()
        with lock:
            ENGINE.say(text)
            ENGINE.runAndWait()

    @staticmethod
    def speak(text):
        threading.Thread(target=Speaker._speak, args=text).start()
