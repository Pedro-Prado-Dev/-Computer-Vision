import threading
from multiprocessing.dummy import Pool as ThreadPool

import pyttsx3


class Speaker:
    def __init__(self) -> None:
        self.ENGINE = pyttsx3.init()
        self.voices = self.ENGINE.getProperty('voices')
        self.ENGINE.setProperty('voice', self.voices[1].id)
        # self.thread_id = threading.Thread(target=lambda: self._speak(text))
        # self.thread_id.start()

        self.pool = ThreadPool(1)

    def _speak(self, text: str):
        lock = threading.Lock()
        with lock:
            self.ENGINE.say(text)
            self.ENGINE.runAndWait()

    def speak(self, text):
        try:
            self.pool.apply_async(lambda: self._speak(text), ())
            self.pool.close()

            self.pool = ThreadPool(1)
        except Exception as e:
            print(e)
            print("LIGEIRINHO E DATA = NONE, START_DATA, OPEN_DATA")
        