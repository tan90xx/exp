import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib import colors

def Drawspec(*param, rate = None, name=None, save=None, a=10, b=4, show=True, bar=False, retn=False):
    # ------------------dft parameters---------------------
    FFT=rate
    HOP=160
    WIN=512
    # ------------------------------------------------------
    n = len(param)
    if n > 1:
        fig, axs = plt.subplots(1, n, sharey=True, figsize=(a, b))
        images = []
        for col, signal in enumerate(param):
            ax = axs[col]
            if np.ndim(param[0]) == 2:
                Mag_signal_db = signal
            else:
                S_signal = librosa.stft(signal, n_fft=FFT, hop_length=HOP, win_length=WIN)  # D x T
                Mag_signal = np.abs(S_signal)
                Mag_signal_db = librosa.amplitude_to_db(Mag_signal)

            Y = np.arange(0,np.shape(Mag_signal_db)[0],1)
            X = np.arange(0,np.shape(Mag_signal_db)[1]/FFT,1/FFT)

            pcm = ax.pcolormesh(X, Y, Mag_signal_db, shading='auto', cmap="coolwarm")

            ax.set_xlim([0,np.shape(Mag_signal_db)[1]/FFT])
            images.append(pcm)
            ax.label_outer()
            ax.set_ylim([1, int(FFT / 2 + 1)])
            ax.set_xlabel('Time (s)')
            ax.grid(False)
            if name:
                ax.set_title("{}".format(name[col]))
            if col == 0:
                ax.set_ylabel('Frequency (Hz)')

        # Find the min and max of all colors for use in setting the color scale.
        if bar:
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            fig.colorbar(images[0], ax=axs, orientation='horizental', fraction=.05, label='Magnitude(dB)')

    else:
        plt.figure(figsize=(a, b))

        if np.ndim(param[0]) == 2:
            Mag_signal_db = param[0]
        else:
            S_signal = librosa.stft(param[0], n_fft=FFT, hop_length=HOP, win_length=WIN)  # D x T
            Mag_signal = np.abs(S_signal)
            Mag_signal_db = librosa.amplitude_to_db(Mag_signal)

        Y = np.arange(0,np.shape(Mag_signal_db)[0],1)
        X = np.arange(0,np.shape(Mag_signal_db)[1]/FFT*2,1/FFT*2)

        pcm = plt.pcolormesh(X, Y, Mag_signal_db, shading='auto', cmap="coolwarm")

        if name:
            plt.title("{}".format(name))
        plt.ylim([0, int(FFT / 2 + 1)])
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(False)
        plt.colorbar(pcm, label='Magnitude (dB)')
    
    if save:
        plt.savefig('{}.png'.format(save), dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    if retn:
        return Mag_signal_db