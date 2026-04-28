from scipy.io.wavfile import write
import numpy as np

# sampling_rate means here 16k numbers
sampling_rate = 16000

# 16k * 2 = 32k samplings
duration = 2

# total evenly beep sound
time = np.linspace(0, duration, sampling_rate * duration)

# 440 hz
frequency = 440

# 0.1 for sound volume and other one is just oscillation
# [volume] * sin( oscillation )
audio = 0.1 * np.sin(2 * np.pi * frequency * time)

write("tone.wav", sampling_rate, audio.astype(np.float32))
print(len(time))
print(audio)
print(len(audio))