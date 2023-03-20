import pyaudio
import struct
import math
import wave
import numpy as np
import time

Threshold = 10

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 256
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2

TIMEOUT_LENGTH = 1

class atoolclass:
    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 100    
    
    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)
        self.data = bytearray()
        
        self.recording = 0#False
        

    def listen(self):
        # print("listen")
        while True:
    
            self.data = self.stream.read(chunk)
            rms_val = self.rms(self.data)
            rec = []
            if (rms_val >= Threshold): 
                print("recording beginning")            
                rec.append(self.data)
                current = time.time()
                end = current + TIMEOUT_LENGTH
                self.recording = 1#True
                print([_ for _ in self.data])
                print(np.shape(self.data))
                # break
                while self.recording:
                    # print("self.recording")
                    
                    current = time.time()          
                    if (self.rms(self.data) >= Threshold): 
                        # print("end = current + TIMEOUT_LENGTH")
                        end = current + TIMEOUT_LENGTH
                        
                    if (current > end):
                        self.recording = 0# False
                        # print("self.recording = 0# False")
                                            
                    rec.append(self.data)
                    self.data = self.stream.read(chunk)

                
                self.write(b''.join(rec))   
                break   
    
    def write(self, recording):
        print("self.write")
        filename = "input_audio.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        print('Returning to listening')    