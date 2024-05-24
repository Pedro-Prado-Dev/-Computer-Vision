import os
import pathlib
import tempfile
import time

from abc import ABC

from pygame import mixer
from gtts import gTTS


class Speaker(ABC):
    @staticmethod
    def speak(text):
        speech = gTTS(text, lang="pt-BR")

        temp_filepath: pathlib.Path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tfile:
            temp_filepath = pathlib.Path(tfile.name)
            speech.write_to_fp(tfile)

            if temp_filepath.exists():
                print(f"Arquivo gerado no diretório: {temp_filepath}")

        mixer.init()
        mixer.music.load(pathlib.Path(tfile.name))
        mixer.music.play()

        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)

        mixer.music.stop()
        os.unlink(temp_filepath)