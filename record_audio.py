import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

fs = 16000
seconds = 1
label = "lewo" # zmieniaj gora prawo dol 
out_dir = f"data/raw/{label}"
os.makedirs(out_dir, exist_ok=True)

for i in range(60):
    print(f"Nagrywanie {i}")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(f"{out_dir}/{label}_{i}.wav", fs, audio)