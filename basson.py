import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import bassonFunction as bf
import guitarfunction as sf

fe, audio_data = wavfile.read('note_basson_plus_sinus_1000_hz.wav')
N = 6000
w0 = 1000
w1 = 40

sb_filter = bf.get_stopBand(w0, w1, fe, N)

enveloppe = sf.get_enveloppe(audio_data, sb_filter)
filtered_audio = bf.apply_filter(audio_data, sb_filter, 3)

filtered_audio = filtered_audio.astype(audio_data.dtype)
wavfile.write('filtered_audio.wav', fe, filtered_audio)

# Time domain plot
plt.figure()
plt.subplot(3, 2, 1)
plt.plot(np.arange(N), sb_filter)
plt.xlim(2900,3100)
plt.title("Réponse à l'impulsion h[n]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid()

# Réponse à une sinusoïde de 1000 Hz
plt.subplot(3, 2, 2)
t = np.arange(len(audio_data)) / fe
sinus_1000 = np.sin(2 * np.pi * 1000 * t)
sinus_filtered = bf.apply_filter(sinus_1000, sb_filter, 1)
plt.plot(t[:5000], sinus_1000[:5000], label='Sinus 1000 Hz (entrée)')
plt.plot(t[:5000], sinus_filtered[:5000], label='Sinus filtré (sortie)')
plt.xlim(0,0.02)
plt.title("Réponse à une sinusoïde de 1000 Hz")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# amplitude de la réponse en fréquence
freq = np.fft.fftfreq(N, d=1/fe)
freq_response = np.fft.fft(sb_filter)
plt.subplot(3, 2, 3)
plt.plot(freq[:N//2], 20*np.log10(np.abs(freq_response[:N//2])))
plt.xlim(800,1200)
plt.title("Amplitude de la réponse en fréquence")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.grid()

# phase de la réponse en fréquence
plt.subplot(3, 2, 4)
plt.plot(freq[:N//2], np.angle(freq_response[:N//2]))
plt.xlim(800,1200)
plt.title("Phase de la réponse en fréquence")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

# Tracer spectres d'amplitude avant filtrage
freq_audio = np.fft.fftfreq(len(audio_data), d=1/fe)
spectrum_before = np.fft.fft(audio_data)
spectrum_after = np.fft.fft(filtered_audio)

plt.subplot(3, 2, 5)
plt.plot(freq_audio[:len(audio_data)//2], 20*np.log10(np.abs(spectrum_before[:len(audio_data)//2])))
plt.xlim(800,1200)
plt.title("Spectre d'amplitude avant filtrage")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

# spectres d'amplitude après filtrage
plt.subplot(3, 2, 6)
plt.plot(freq_audio[:len(audio_data)//2], 20*np.log10(np.abs(spectrum_after[:len(audio_data)//2])))
plt.xlim(800,1200)
plt.title("Spectre d'amplitude après filtrage")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.subplots_adjust(
    hspace=0.5,  # vertical space
    wspace=0.4   # horizontal space
)
plt.show()