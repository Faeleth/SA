import sounddevice as sd
import numpy as np
import joblib
import pandas as pd
import librosa
from collections import deque

# --- wczytanie modelu ---
data = joblib.load("model/voice_model.pkl")
model = data["model"]
le = data["le"]
scaler = data.get("scaler", None)

# --- ustawienia ---
FS = 16000
# Jak długo trwa jedno nagranie audio przed analizą
# Mniejsza wartość = szybsza odpowiedź, ale gorsze features
# Większa wartość = lepsze features, ale wolniej
CHUNK_DURATION = .1 
SILENCE_THRESHOLD = None # dod wyalenia zawsze none mpoki co None deaulkt spoko
# Liczba frames które muszą być "mową" aby zacząć nagrywanie
# Zapobiega szumowi tła
# Jeśli = 2, to pierwsze 2 chunks muszą być powyżej threshold
SUSTAIN_FRAMES = 2
# Ile frames ciszy czekamy zanim powiemy "koniec mowy"
# Jeśli = 3, czekamy 150ms ciszy (3 × 50ms)
# Jeśli za mało = przerwy w słowach będą rozcinane
# Jeśli za dużo = długi delay
MAX_SILENT_FRAMES = 2

audio_buffer = deque(maxlen=SUSTAIN_FRAMES)

# --- adaptacyjny prog ciszy ---
def estimate_noise_level(duration=1.0):
    """Oszacuj poziom szumu tla"""
    print("Calibrating... (stay quiet)")
    noise = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    return np.max(np.abs(noise.flatten()))

def set_silence_threshold():
    global SILENCE_THRESHOLD
    SILENCE_THRESHOLD = estimate_noise_level() * 2.5

# --- ekstrakcja cech - MUSI PASOWAC DO TRENINGU ---
def extract_features_from_audio(audio, sr):
    """Wyciagnij 52 cechy: mfcc_mean (13) + mfcc_std (13) + mfcc_delta_mean (13) + mfcc_delta_std (13)"""
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio buffer contains NaN/Inf")
    
    # Wspolczynniki MFCC
    mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Delty MFCC (predkosc zmian) - obsluga krotkiego audio
    try:
        mfcc_delta = librosa.feature.delta(mfcc, width=min(9, mfcc.shape[1]))
    except:
        # Zapas dla bardzo krotkiego audio
        mfcc_delta = np.zeros_like(mfcc)
    
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)
    
    # Polacz wszystkie cechy (razem 52)
    features = list(mfcc_mean) + list(mfcc_std) + list(mfcc_delta_mean) + list(mfcc_delta_std)
    return np.array(features)

# --- glowna funkcja nasluchu ---
def listen_command():
    """
    Nasluchuj komendy glosowej i zwroc rozpoznana komende.
    Zwrot: str (nazwa komendy) albo None
    """
    if SILENCE_THRESHOLD is None:
        set_silence_threshold()
    
    print("Listening for command...")
    chunk_samples = int(CHUNK_DURATION * FS)
    
    audio_chunks = []
    speech_detected = False
    silent_frames = 0
    
    MAX_SILENT_FRAMES = 3  # Poczekaj 150ms ciszy, by domknac wypowiedz
    
    while True:
        chunk = sd.rec(chunk_samples, samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        chunk = chunk.flatten()
        
        is_speech = np.max(np.abs(chunk)) > SILENCE_THRESHOLD
        
        if is_speech:
            speech_detected = True
            silent_frames = 0
            audio_chunks.append(chunk)
            print(".", end="", flush=True)
        elif speech_detected:
            silent_frames += 1
            audio_chunks.append(chunk)
            print(".", end="", flush=True)
            
            if silent_frames >= MAX_SILENT_FRAMES:
                print(" [Processing]")
                break
        
        # bezpieczny limit czasu: 5 sekund
        if len(audio_chunks) > (5 * FS) // chunk_samples:
            print(" [Timeout]")
            return None
    
    if not audio_chunks:
        return None
    
    try:
        audio_data = np.concatenate(audio_chunks)
        
        # Wyciagnij 52 cechy (MUSI PASOWAC DO TRENINGU!)
        features = extract_features_from_audio(audio_data, FS)
        
        # Przeskaluj, jesli jest scaler
        if scaler is not None:
            features = scaler.transform([features])[0]
        
        # Utworz dataframe z DOKLADNYMI nazwami kolumn z treningu
        n_mfcc = 13
        columns = (
            [f"mfcc_mean_{i}" for i in range(n_mfcc)] +
            [f"mfcc_std_{i}" for i in range(n_mfcc)] +
            [f"mfcc_delta_mean_{i}" for i in range(n_mfcc)] +
            [f"mfcc_delta_std_{i}" for i in range(n_mfcc)]
        )
        features_df = pd.DataFrame([features], columns=columns)
        
        # Predykcja z pewnoscia
        probabilities = model.predict_proba(features_df)[0]
        confidence = np.max(probabilities)
        pred_idx = np.argmax(probabilities)
        
        CONFIDENCE_THRESHOLD = 0.35
        
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"⚠ Low confidence ({confidence:.2f})")
            return None
        
        command = le.inverse_transform([pred_idx])[0]
        print(f"✓ {command} ({confidence:.2%})")
        return command
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
