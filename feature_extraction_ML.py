import librosa
import numpy as np

FRAME_SIZE = 2048
HOP_LENGTH = 512
NUM_OF_CHUNKS = 113
SAMPLING_RATE = 22050
COUNT_PADDED = 0

# -- TIME DOMAIN FEATURES - AMPLITUDE -----------------------------------------------------------------------------------------
def compute_zero_crossing_rate(audio):
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    #return np.array([np.mean(chunk) for chunk in np.split(zcr, NUM_OF_CHUNKS)])
    return np.sum(zcr)

def zero_crossing_rate_extr(audio):
    audio['mean_zcr_AV'] = audio.apply(lambda row: compute_zero_crossing_rate(row.AV), axis=1)
    audio['mean_zcr_MV'] = audio.apply(lambda row: compute_zero_crossing_rate(row.MV), axis=1)
    audio['mean_zcr_PV'] = audio.apply(lambda row: compute_zero_crossing_rate(row.PV), axis=1)
    audio['mean_zcr_TV'] = audio.apply(lambda row: compute_zero_crossing_rate(row.TV), axis=1)




def amplitude_envelope(audio):
    """Calculate the amplitude envelope of a signal per frame"""
    return np.array([max(audio[i:i + FRAME_SIZE]) for i in range(0, len(audio), HOP_LENGTH)])


def stat_amplitude_envelope_extr(audio, stat_measure, type):
    audio[f'{type}_ae_AV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.AV)), axis=1)
    audio[f'{type}_ae_MV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.MV)), axis=1)
    audio[f'{type}_ae_PV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.PV)), axis=1)
    audio[f'{type}_ae_TV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.TV)), axis=1)


def stat_rms_extr(audio, stat_measure, type):
    audio[f'{type}_rms_AV'] = audio.apply(lambda row: stat_measure(librosa.feature.rms(row.AV)), axis=1)
    audio[f'{type}_rms_MV'] = audio.apply(lambda row: stat_measure(librosa.feature.rms(row.MV)), axis=1)
    audio[f'{type}_rms_PV'] = audio.apply(lambda row: stat_measure(librosa.feature.rms(row.PV)), axis=1)
    audio[f'{type}_rms_TV'] = audio.apply(lambda row: stat_measure(librosa.feature.rms(row.TV)), axis=1)


def stat_amplitude_envelope_perc_extr(audio, percentile=50):
    audio[f'perc_{percentile}_ae_AV'] = audio.apply(lambda row: np.percentile(amplitude_envelope(row.AV), percentile),
                                                    axis=1)
    audio[f'perc_{percentile}_ae_MV'] = audio.apply(lambda row: np.percentile(amplitude_envelope(row.MV), percentile),
                                                    axis=1)
    audio[f'perc_{percentile}_ae_PV'] = audio.apply(lambda row: np.percentile(amplitude_envelope(row.PV), percentile),
                                                    axis=1)
    audio[f'perc_{percentile}_ae_TV'] = audio.apply(lambda row: np.percentile(amplitude_envelope(row.TV), percentile),
                                                    axis=1)


def stat_rms_perc_extr(audio, percentile=50):
    audio[f'perc_{percentile}_rms_AV'] = audio.apply(lambda row: np.percentile(librosa.feature.rms(row.AV), percentile),
                                                     axis=1)
    audio[f'perc_{percentile}_rms_MV'] = audio.apply(lambda row: np.percentile(librosa.feature.rms(row.MV), percentile),
                                                     axis=1)
    audio[f'perc_{percentile}_rms_PV'] = audio.apply(lambda row: np.percentile(librosa.feature.rms(row.PV), percentile),
                                                     axis=1)
    audio[f'perc_{percentile}_rms_TV'] = audio.apply(lambda row: np.percentile(librosa.feature.rms(row.TV), percentile),
                                                     axis=1)


"""Find frequency at which the maximum amplitude occurs of a signal"""


# -- FREQUENCY DOMAIN FEATURES --------------------------------------------------------------------------------------------------------
def max_frequency(audio):
    fft_sig = np.absolute(np.fft.fft(audio))
    return np.argmax(fft_sig) * SAMPLING_RATE / len(fft_sig)


def max_frequency_extr(audio):
    audio[f'max_mag_freq_AV'] = audio.apply(lambda row: max_frequency(row.AV), axis=1)
    audio[f'max_mag_freq_MV'] = audio.apply(lambda row: max_frequency(row.MV), axis=1)
    audio[f'max_mag_freq_PV'] = audio.apply(lambda row: max_frequency(row.PV), axis=1)
    audio[f'max_mag_freq_TV'] = audio.apply(lambda row: max_frequency(row.TV), axis=1)


def total_energy(audio):
    return np.sum(audio ** 2)


def total_energy_extr(audio):
    audio[f'total_energy_AV'] = audio.apply(lambda row: total_energy(row.AV), axis=1)
    audio[f'total_energy_MV'] = audio.apply(lambda row: total_energy(row.MV), axis=1)
    audio[f'total_energy_PV'] = audio.apply(lambda row: total_energy(row.PV), axis=1)
    audio[f'total_energy_TV'] = audio.apply(lambda row: total_energy(row.TV), axis=1)


def set_len(audio, sec):
    global COUNT_PADDED
    num_of_samples = sec * SAMPLING_RATE
    audio_length = audio.shape[0]
    if audio_length > num_of_samples:
        audio = audio[:num_of_samples]
    elif audio_length < num_of_samples:
        audio = np.pad(audio, (0, num_of_samples - audio_length), 'constant')
        COUNT_PADDED += 1

    return audio


def set_len_extr(audio,sec=8):
    audio['AV'] = audio.apply(lambda row: set_len(row.AV,sec), axis=1)
    audio['MV'] = audio.apply(lambda row: set_len(row.MV,sec), axis=1)
    audio['PV'] = audio.apply(lambda row: set_len(row.PV,sec), axis=1)
    audio['TV'] = audio.apply(lambda row: set_len(row.TV,sec), axis=1)
    print(f'{COUNT_PADDED} audios have been padded')
    return audio
