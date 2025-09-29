import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
import guitarfunction as fc

# Lecture du fichier audio
fe, audio_signal = wavfile.read('note_guitare_lad.wav')
omega = np.pi / 1000
gain_db = -3
N = fc.get_filtre_order(omega, gain_db)
print(N)
filtre_temporel, filtre_frequenciel = fc.reponse_impulsionnelle_filtre_RIF_temporel(N,omega)
enveloppe = fc.get_enveloppe(audio_signal,filtre_temporel)
harmonique, fondamental  = fc.analyse_freq(audio_signal, fe)
note_dict = fc.note_dict(fondamental)
fc.composition_bethoven(harmonique, fe, enveloppe, note_dict)

fc.create_wav_sound(harmonique, fe, note_dict["LA#"], enveloppe, 4, "la_d.wav")

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
w, h = signal.freqz(filtre_temporel, worN=fe)
plt.plot(w, np.abs(h))
plt.xlabel("ω (rad/échantillon)")
plt.ylabel("Gain")
plt.axhline(10**(-3/20), linestyle='--', label='0.7079')
plt.axvline(omega, linestyle='--', label='wc')
plt.xlim(0,0.05)
plt.title("Réponse fréquentielle du filtre RIF")
plt.legend()
# Enveloppe temporelle du signal
plt.subplot(3,2,4)
plt.plot(enveloppe)
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Enveloppe temporelle du signal")
## 32 premieres harmoniques
plt.subplot(3,2,5)
freqs = [h["frequence"] for h in harmonique]
gains = [h["gain"] for h in harmonique]
plt.stem(freqs,20*np.log10(gains))
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (db)")
plt.title("32 premieres harmoniques")
# Signal audio redressé et enveloppe
plt.subplot(3,2,6)
plt.plot(np.abs(audio_signal))
plt.plot(enveloppe)
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal audio redressé et enveloppe")
plt.show()