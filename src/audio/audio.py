import threading
import weakref
from multiprocessing.dummy import Pool as ThreadPool

import pyttsx3


enable_speak = True


class Speaker:
    def __init__(self) -> None:
        self.ENGINE = pyttsx3.init()
        self.voices = self.ENGINE.getProperty('voices')
        self.ENGINE.setProperty('voice', self.voices[1].id)
        self.pool = ThreadPool(1)

    def _speak(self, text: str):
        global enable_speak

        enable_speak = False
        lock = threading.Lock()
        with lock:
            self.ENGINE.say(text)
            self.ENGINE.runAndWait()

            enable_speak = True

    def speak(self, text):
        if not enable_speak:
            return

        try:
            self.pool.apply_async(lambda: self._speak(text), ())
            self.pool.close()

            self.pool = ThreadPool(1)
        except Exception as e:
            print(e)
        