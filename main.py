import pyaudio
import wave
import threading
import time
import os
import whisper
import glob
import datetime


# 设置参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_TIME = 5
OUTPUT_DIR = "recordings"

# 创建PyAudio对象
p = pyaudio.PyAudio()

# 创建语音识别模型实例
model = whisper.load_model("base")

# 创建线程锁和标志
lock = threading.Lock()
stop_flag = False

# 录制音频数据
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 5
    NUM_CHUNKS = int(RATE / CHUNK * RECORD_SECONDS)
    FILE_PREFIX = 'recording'
    FILE_SUFFIX = '.wav'
    PATH = os.path.join(os.getcwd(), 'recordings')

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index= 1,
                    frames_per_buffer=CHUNK)

    while True:
        frames = []
        for i in range(NUM_CHUNKS):
            data = stream.read(CHUNK)
            frames.append(data)

        filename = FILE_PREFIX + '_' + str(int(time.time())) + FILE_SUFFIX
        filepath = os.path.join(PATH, filename)

        wf = wave.open(filepath, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        time.sleep(0.01)

    stream.stop_stream()
    stream.close()
    p.terminate()

# 将录制的音频文件转换为文本
def recognize_audio():

    while not stop_flag:
        # 寻找最新的WAV文件
        newest_file = max(glob.glob(f"{OUTPUT_DIR}/*.wav"), key=os.path.getctime)
        # print("now reading" + newest_file)
        # print(int(time.time()))
        now = datetime.datetime.now()
        lock.acquire()
        try:
            text = model.transcribe(newest_file)
            # print("v2t time is " + now.strftime('%Y-%m-%d %H:%M:%S'))
            print(text['text'])
        except Exception as e:
            print(f"无法识别音频文件: {e}")
        finally:
            lock.release()

        time.sleep(5)

# 创建录制音频的线程
audio_thread = threading.Thread(target=record_audio)

# 创建识别音频的线程
recognize_thread = threading.Thread(target=recognize_audio)

# 启动线程
audio_thread.start()
recognize_thread.start()

# 等待程序结束
audio_thread.join()
recognize_thread.join()

# 关闭PyAudio对象
p.terminate()
