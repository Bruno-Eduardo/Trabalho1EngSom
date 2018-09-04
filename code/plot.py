#from pathlib import Path
#
#for path in Path(enderecoDoggos).iterdir():
#    info = path.stat()
#    print(info.st_mtime)

import os
from pydub import AudioSegment
import pandas
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import librosa
import librosa.display

#enderecos de cada pasta de amostras de som
enderecoDoggos = '../doggos/'
enderecoSamples = '../samples/'

medias = []
desvios = []

print('Plot - Flatness')


#descobre a frequencia de amostragem de um arquivo e carrega no librosa
framerate = AudioSegment.from_mp3(enderecoDoggos+'dogbark2.wav').frame_rate
amostra, fs = (librosa.load(enderecoDoggos+'dogbark2.wav', sr=framerate))
#calcula dft através de dfft para plotar
transf1 = (librosa.stft(amostra, n_fft=2048, hop_length=128, win_length=1024, window='hann'))

#repete para um arquivo de violão
framerate = AudioSegment.from_mp3(enderecoSamples+'AC_GuitarMix120A-03.wav').frame_rate
amostra, fs = (librosa.load(enderecoSamples+'AC_GuitarMix120A-03.wav'   , sr=framerate))
transf2 = (librosa.stft(amostra, n_fft=2048*8, hop_length=128, win_length=1024*8, window='hann'))


plt.figure()
plt.subplot(2, 1, 1)
librosa.display.specshow(transf1, sr=fs, y_axis='log', x_axis='time', label='Doggo')
plt.subplot(2, 1, 2)
librosa.display.specshow(transf2, sr=fs, y_axis='log', x_axis='time', label='Violão')
plt.show()
