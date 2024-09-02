import pyaudio
import wave
import threading
import time
import os
import whisper
import io
import datetime

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

# Initialize PyAudio and Whisper model
p = pyaudio.PyAudio()
model = whisper.load_model("base")

# Thread control variables
lock = threading.Lock()
stop_flag = False

# Function to record audio and store it in memory
def record_audio():
    global stop_flag

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while not stop_flag:
        frames = []
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Store audio in a BytesIO object
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        # Move buffer cursor to the beginning for reading
        audio_buffer.seek(0)

        # Pass buffer to the recognition thread
        with lock:
            recognize_audio(audio_buffer)

    stream.stop_stream()
    stream.close()

# Function to transcribe audio stored in memory
def recognize_audio(audio_buffer):
    try:
        # Transcribe the audio in memory
        text = model.transcribe(audio_buffer)
        print(f"Transcription at {datetime.datetime.now()}: {text['text']}")
    except Exception as e:
        print(f"Error in transcribing audio: {e}")
    finally:
        # Release memory
        audio_buffer.close()

# Start recording thread
audio_thread = threading.Thread(target=record_audio)

audio_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_flag = True
    audio_thread.join()
    p.terminate()
