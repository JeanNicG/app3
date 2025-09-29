import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
import fonction as fc

# Lecture du fichier audio
fe, audio_signal = wavfile.read('note_guitare_lad.wav')
omega = np.pi / 1000
gain_db = -3
N = fc.get_filtre_order(omega, gain_db)
filtre = fc.reponse_impulsionnelle_filtre_RIF_temporel(N)
enveloppe = fc.get_enveloppe(audio_signal,filtre)
harmonique, fondamental  = fc.analyse_freq(audio_signal, fe)
note_dict = fc.note_dict(fondamental)
fc.composition_bethoven(harmonique, fe, enveloppe, note_dict)

# figures
plt.figure()
# Signal audio
plt.subplot(3,2,1)
plt.plot(audio_signal)
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal audio")
# Signal audio redresse
plt.subplot(3,2,2)
plt.plot(np.abs(audio_signal))
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal audio redresse")
# reponse impulsionnelle temporelle du filtre RIF
plt.subplot(3,2,3)
n = np.arange(-(N+1)//2, (N+1)//2)
plt.stem(n,filtre)
plt.ylim(0,0.002)
plt.xlabel("n")
plt.ylabel("h[n]")
plt.title("reponse impulsionnelle temporelle du filtre RIF")
# Enveloppe temporelle du signal
plt.subplot(3,2,4)
plt.plot(enveloppe)
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Enveloppe temporelle du signal")
# 32 premieres harmoniques
plt.subplot(3,2,5)
freqs = [h["frequence"] for h in harmonique]
gains = [h["gain"] for h in harmonique]
plt.stem(freqs,20*np.log10(gains))
plt.xlabel("frequence")
plt.ylabel("Amplitude (db)")
plt.title("32 premieres harmoniques")
plt.show()