import librosa
import numpy as np

FRAME_SIZE = 2048
HOP_LENGTH = 512
NUM_OF_CHUNKS = 113

SAMPLING_RATE = 22050

# -- TIME DOMAIN FEATURES - AMPLITUDE -----------------------------------------------------------------------------------------
def zero_crossing_rate_extr(audio):
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    return np.array([np.mean(chunk) for chunk in np.split(zcr, NUM_OF_CHUNKS)])

def amplitude_envelope(audio):
    """Calculate the amplitude envelope of a signal per frame"""
    return np.array([max(audio[i:i+FRAME_SIZE]) for i in range(0, len(audio), HOP_LENGTH)])

def stat_amplitude_envelope_extr(audio, stat_measure, type):
    audio[f'{type}_ae_AV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.AV)), axis = 1)
    audio[f'{type}_ae_MV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.MV)), axis = 1)
    audio[f'{type}_ae_PV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.PV)), axis = 1)
    audio[f'{type}_ae_TV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.TV)), axis = 1)

def stat_rms_extr(audio, stat_measure, type):
    audio[f'{type}_rms_AV'] = audio.apply(lambda row: stat_measure(librosa.feature.rms(row.AV)), axis = 1)
    audio[f'{type}_rms_MV'] = audio.apply(lambda row: stat_measure(librosa.feature.rms(row.MV)), axis = 1)
    audio[f'{type}_rms_PV'] = audio.apply(lambda row: stat_measure(librosa.feature.rms(row.PV)), axis = 1)
    audio[f'{type}_rms_TV'] = audio.apply(lambda row: stat_measure(librosa.feature.rms(row.TV)), axis = 1)

def stat_amplitude_envelope_perc_extr(audio, percentile=50):
    audio[f'perc_{percentile}_ae_AV'] = audio.apply(lambda row: np.percentile(amplitude_envelope(row.AV), percentile),axis=1)
    audio[f'perc_{percentile}_ae_MV'] = audio.apply(lambda row: np.percentile(amplitude_envelope(row.MV), percentile),axis=1)
    audio[f'perc_{percentile}_ae_PV'] = audio.apply(lambda row: np.percentile(amplitude_envelope(row.PV), percentile),axis=1)
    audio[f'perc_{percentile}_ae_TV'] = audio.apply(lambda row: np.percentile(amplitude_envelope(row.TV), percentile),axis=1)

def stat_rms_perc_extr(audio, percentile=50):
    audio[f'perc_{percentile}_rms_AV'] = audio.apply(lambda row: np.percentile(librosa.feature.rms(row.AV), percentile),axis=1)
    audio[f'perc_{percentile}_rms_MV'] = audio.apply(lambda row: np.percentile(librosa.feature.rms(row.MV), percentile),axis=1)
    audio[f'perc_{percentile}_rms_PV'] = audio.apply(lambda row: np.percentile(librosa.feature.rms(row.PV), percentile),axis=1)
    audio[f'perc_{percentile}_rms_TV'] = audio.apply(lambda row: np.percentile(librosa.feature.rms(row.TV), percentile),axis=1)

"""Find frequency at which the maximum amplitude occurs of a signal"""

# -- FREQUENCY DOMAIN FEATURES --------------------------------------------------------------------------------------------------------
def max_frequency(audio):
    fft_sig = np.absolute(np.fft.fft(audio))
    return np.argmax(fft_sig)*SAMPLING_RATE/len(fft_sig)

def max_frequency_extr(audio):
    audio[f'max_mag_freq_AV'] = audio.apply(lambda row: max_frequency(row.AV),axis=1)
    audio[f'max_mag_freq_MV'] = audio.apply(lambda row: max_frequency(row.MV),axis=1)
    audio[f'max_mag_freq_PV'] = audio.apply(lambda row: max_frequency(row.PV),axis=1)
    audio[f'max_mag_freq_TV'] = audio.apply(lambda row: max_frequency(row.TV),axis=1)


def extract(data):
    data['mean_zcrAV'] = data['AV'].map(zero_crossing_rate_extr)

    return data
