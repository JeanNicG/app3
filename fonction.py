import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

def reponse_impulsionnelle_filtre_RIF_temporel(N):
    h = np.ones(N+1) / (N+1)
    return h

def get_filtre_order(omega, gain_target_db):
    gain_target_linear = 10**(gain_target_db/20)

    best_N = None
    best_gain_diff = float('inf')

    for N in range(1, 1000, 1):
        h = reponse_impulsionnelle_filtre_RIF_temporel(N)
        
        # reponse impulsionnelle a la frequence omega (domaine frequenciel) TFSD
        H = 0+0j
        for k in range(len(h)):
            H += h[k] * np.exp(-1j * omega * k)

        # garde le gain le plus proche de -3db a la frequence normalise pi/1000
        gain = np.abs(H)
        gain_diff = abs(gain - gain_target_linear)
        if gain_diff < best_gain_diff:
            best_gain_diff = gain_diff
            best_N = N
    print(f"N optimal: {best_N}")
    return best_N

def get_enveloppe(audio_signal, filtre):
    audio_signal_redresse = np.abs(audio_signal)
    return np.convolve(audio_signal_redresse, filtre)

def analyse_freq(audio_data, fe):
    # Calcul de la FFT
    N = len(audio_data)
    signal_fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(N, 1/fe)

    # Calcul du gain et de la phase
    gain = np.abs(signal_fft)
    phase = np.angle(signal_fft)

    # Trouver la fréquence fondamentale (pic le plus eleve)
    N_positif = N // 2
    fondamental_idx = np.argmax(gain[1:N_positif]) + 1
    fondamental = freqs[fondamental_idx]

    # Extraire les 32 premières harmoniques
    harmoniques = []
    for h in range(1, 33):
        freq_target = h * fondamental
        idx = np.argmin(np.abs(freqs[:N_positif] - freq_target))
        harmoniques.append({
            'harmonique': h,
            'frequence': freqs[idx],
            'gain': gain[idx],
            'phase': phase[idx]
        })
    return harmoniques, fondamental 

def note_dict(la_d):
    note_freq_dict = { 
        "DO":2**(-10/12) * la_d,
        "DO#":2**(-9/12) * la_d,
        "RE":2**(-8/12) * la_d,
        "RE#":2**(-7/12) * la_d,
        "MI":2**(-6/12) * la_d,
        "FA":2**(-5/12) * la_d,
        "FA#":2**(-4/12) * la_d,
        "SOL":2**(-3/12) * la_d,
        "SO#":2**(-2/12) * la_d,
        "LA": 2**(-1/12) * la_d,
        "LA#":la_d,
        "SI": 2**(1/12) * la_d
    }
    return note_freq_dict

def get_sound(harmonique, fe, fondamental, enveloppe, duration):
    print("get_sound:", fondamental)
    t = np.linspace(0, duration, int(fe * duration))
    signal_synth = np.zeros(len(t))
    
    freq_ratio = fondamental / harmonique[0]["frequence"]

    for n in range(len(harmonique)):
        new_freq = harmonique[n]["frequence"] * freq_ratio
        signal_synth += harmonique[n]["gain"] * np.sin(2 * np.pi * new_freq * t + harmonique[n]["phase"])
    

    env_resized = np.interp(np.linspace(0, 1, len(signal_synth)), np.linspace(0, 1, len(enveloppe)), enveloppe)
    signal_synth *= env_resized
    signal_synth *= np.hamming(len(signal_synth))

    return signal_synth

def get_silence(fe, duration):
    t = np.linspace(0, duration, int(fe*duration))
    silence = np.zeros_like(t)
    return silence

def composition_bethoven(harmonique, fe, enveloppe, note_dict):
    sol = get_sound(harmonique, fe, note_dict["SOL"], enveloppe, 0.4)
    mi_b = get_sound(harmonique, fe, note_dict["RE#"], enveloppe, 1)
    fa = get_sound(harmonique, fe, note_dict["FA"], enveloppe, 0.4)
    re = get_sound(harmonique, fe, note_dict["RE"], enveloppe, 1)
    
    silence = get_silence(fe, 0.05)
    silence1 = get_silence(fe, 0.3)

    musique = np.concatenate((
        sol, silence, 
        sol, silence, 
        sol, silence, 
        mi_b, silence1, 
        fa, silence, 
        fa, silence, 
        fa, silence, 
        re, silence1
    ))
    
    musique = np.int16(np.array(musique) / np.max(np.abs(musique)) * 32767)
    wavfile.write("beethoven.wav", fe, musique)
    return musique