import librosa
import numpy as np

FRAME_SIZE = 2048
HOP_LENGTH = 512
NUM_OF_CHUNKS = 113


def zero_crossing_rate_extr(audio):
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    return np.array([np.mean(chunk) for chunk in np.split(zcr, NUM_OF_CHUNKS)])


def amplitude_envelope(signal, frame_size, hop_length):
    """Calculate the amplitude envelope of a signal per frame"""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])


def stat_amplitude_envelope_extr(audio, stat_measure, type):
    audio[f'{type}_AV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.AV,FRAME_SIZE,HOP_LENGTH)), axis = 1)
    audio[f'{type}_MV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.MV,FRAME_SIZE,HOP_LENGTH)), axis = 1)
    audio[f'{type}_PV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.PV,FRAME_SIZE,HOP_LENGTH)), axis = 1)
    audio[f'{type}_TV'] = audio.apply(lambda row: stat_measure(amplitude_envelope(row.TV,FRAME_SIZE,HOP_LENGTH)), axis = 1)


def extract(data):
    data['mean_zcrAV'] = data['AV'].map(zero_crossing_rate_extr)

    return data
