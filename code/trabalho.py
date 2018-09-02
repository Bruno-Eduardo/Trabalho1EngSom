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

print('Flatness')

i=0
for filename in os.listdir(enderecoDoggos):
    #descobre a frequencia de amostragem de cada arquivo e carrega no librosa
    framerate = AudioSegment.from_mp3(enderecoDoggos+filename).frame_rate
    amostra, fs = (librosa.load(enderecoDoggos+filename, sr=framerate))

    #calcula dft através de dfft, calcula o centroid/flatness e faz a media e o desvio
    transf = (librosa.stft(amostra, n_fft=2048, hop_length=128, win_length=1024, window='hann'))
    centroid = (librosa.feature.spectral_flatness(S=np.abs(transf), n_fft=2048,hop_length=512))
    medias.append(np.mean(centroid))
    desvios.append(np.std(centroid))
    i += 1

mediaDoggos  = np.mean ( np.array(medias))
desvioDoggos = np.mean ( np.array(desvios))

#repete o procedimento para os sons de instrumentos
medias = []
desvios = []

j=i-1;i=0
for filename in os.listdir(enderecoSamples):
    framerate = AudioSegment.from_mp3(enderecoSamples+filename).frame_rate
    amostra, fs = (librosa.load(enderecoSamples+filename, sr=framerate))

    transf = (librosa.stft(amostra, n_fft=2048, hop_length=128, win_length=1024, window='hann'))
    centroid = (librosa.feature.spectral_flatness(S=np.abs(transf), n_fft=2048,hop_length=512))
    medias.append(np.mean(centroid))
    desvios.append(np.std(centroid))
    i += 1

#calcula a media das medias e a media dos desvios
mediaSamples  = np.mean ( np.array(medias))
desvioSamples = np.mean ( np.array(desvios))

#calcula o p-valor
t, p = st.ttest_ind_from_stats(mediaDoggos, desvioDoggos, j, mediaSamples, desvioSamples, i)

print("Doggos: ",mediaDoggos, desvioDoggos)
print("Samples: ",mediaSamples, desvioSamples)
print("P-value", p)
