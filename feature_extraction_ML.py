import librosa
import numpy as np
from scipy.stats import skew, kurtosis

FRAME_SIZE = 2048
HOP_LENGTH = 512
NUM_OF_CHUNKS = 113
SAMPLING_RATE = 22050
COUNT_PADDED = 0


def compute_zero_crossing_rate(audio):
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    # return np.array([np.mean(chunk) for chunk in np.split(zcr, NUM_OF_CHUNKS)])
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


def onset_detection(audio):
    oenv = librosa.onset.onset_strength(y=audio, sr=22050)
    return len(librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False))


def onset_detection_extr(audio):
    audio['od_AV'] = audio.apply(lambda row: onset_detection(row.AV), axis=1)
    audio['od_MV'] = audio.apply(lambda row: onset_detection(row.MV), axis=1)
    audio['od_PV'] = audio.apply(lambda row: onset_detection(row.PV), axis=1)
    audio['od_TV'] = audio.apply(lambda row: onset_detection(row.TV), axis=1)


def skewness(audio):
    return skew(audio)


def skewness_extr(audio):
    audio['sk_AV'] = audio.apply(lambda row: skewness(row.AV), axis=1)
    audio['sk_MV'] = audio.apply(lambda row: skewness(row.MV), axis=1)
    audio['sk_PV'] = audio.apply(lambda row: skewness(row.PV), axis=1)
    audio['sk_TV'] = audio.apply(lambda row: skewness(row.TV), axis=1)


def kurtosis_(audio):
    return kurtosis(audio)


def kurtosis_extr(audio):
    audio['ku_AV'] = audio.apply(lambda row: kurtosis_(row.AV), axis=1)
    audio['ku_MV'] = audio.apply(lambda row: kurtosis_(row.MV), axis=1)
    audio['ku_PV'] = audio.apply(lambda row: kurtosis_(row.PV), axis=1)
    audio['ku_TV'] = audio.apply(lambda row: kurtosis_(row.TV), axis=1)


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


"""
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


def set_len_extr(audio, sec=8):
    audio['AV'] = audio.apply(lambda row: set_len(row.AV, sec), axis=1)
    audio['MV'] = audio.apply(lambda row: set_len(row.MV, sec), axis=1)
    audio['PV'] = audio.apply(lambda row: set_len(row.PV, sec), axis=1)
    audio['TV'] = audio.apply(lambda row: set_len(row.TV, sec), axis=1)
    print(f'{COUNT_PADDED} audios have been padded')
    return audio
"""


def set_len_extr(dataset, sec=8, augment=False):
    sample2_AV = 0
    sample2_MV = 0
    sample2_PV = 0
    sample2_TV = 0

    num_of_samples = sec * SAMPLING_RATE
    new_samples = []
    for index, row in dataset.iterrows():
        audio_length_AV = row.AV.shape[0]
        audio_length_MV = row.AV.shape[0]
        audio_length_PV = row.AV.shape[0]
        audio_length_TV = row.AV.shape[0]

        eligible_for_augmentation = (audio_length_AV > 2 * num_of_samples) and \
                                    (audio_length_MV > 2 * num_of_samples) and \
                                    (audio_length_PV > 2 * num_of_samples) and \
                                    (audio_length_TV > 2 * num_of_samples) and \
                                    (row.MURMUR == 'Present') and \
                                    augment

        if audio_length_AV > num_of_samples:
            first_half = row.AV[:num_of_samples]
            second_half = row.AV[num_of_samples: 2 * num_of_samples]
            row.AV = first_half
            if eligible_for_augmentation:
                sample2_AV = second_half
        elif audio_length_AV < num_of_samples:
            row.AV = np.pad(row.AV, (0, num_of_samples - audio_length_AV), 'constant')

        if audio_length_MV > num_of_samples:
            first_half = row.MV[:num_of_samples]
            second_half = row.MV[num_of_samples: 2 * num_of_samples]
            row.MV = first_half
            if eligible_for_augmentation:
                sample2_MV = second_half
        elif audio_length_MV < num_of_samples:
            row.MV = np.pad(row.MV, (0, num_of_samples - audio_length_MV), 'constant')

        if audio_length_PV > num_of_samples:
            first_half = row.PV[:num_of_samples]
            second_half = row.PV[num_of_samples: 2 * num_of_samples]
            row.PV = first_half
            if eligible_for_augmentation:
                sample2_PV = second_half
        elif audio_length_PV < num_of_samples:
            row.PV = np.pad(row.PV, (0, num_of_samples - audio_length_PV), 'constant')

        if audio_length_TV > num_of_samples:
            first_half = row.TV[:num_of_samples]
            second_half = row.TV[num_of_samples: 2 * num_of_samples]
            row.TV = first_half
            if  eligible_for_augmentation:
                sample2_TV = second_half
        elif audio_length_TV < num_of_samples:
            row.TV = np.pad(row.TV, (0, num_of_samples - audio_length_TV), 'constant')

        if eligible_for_augmentation:
            new_row = {'Patient_ID': row.Patient_ID,
                       'AV': sample2_AV,
                       'MV': sample2_MV,
                       'PV': sample2_PV,
                       'TV': sample2_TV,
                       'MURMUR': row.MURMUR}

            new_samples.append(new_row)

    for new_sample in new_samples:
        dataset = dataset.append(new_sample, ignore_index=True)

    return dataset

def calculate_split_frequency_bin(split_frequency, sample_rate, num_frequency_bins):
    """Infer the frequency bin associated to a given split frequency."""

    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / num_frequency_bins
    split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)


def band_energy_ratio(audio, split_frequency):
    """Calculate band energy ratio with a given split frequency."""
    spectrogram = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    split_frequency_bin = calculate_split_frequency_bin(split_frequency, SAMPLING_RATE, len(spectrogram[0]))
    band_energy_ratio = []

    # calculate power spectrogram
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T

    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)

    return np.array(band_energy_ratio)


def band_energy_ratio_extr(audio, stat_measure, type, split_frequency):
    audio[f'BER_{type}_AV'] = audio.apply(lambda row: stat_measure(band_energy_ratio(row.AV, split_frequency)), axis=1)
    audio[f'BER_{type}_MV'] = audio.apply(lambda row: stat_measure(band_energy_ratio(row.MV, split_frequency)), axis=1)
    audio[f'BER_{type}_PV'] = audio.apply(lambda row: stat_measure(band_energy_ratio(row.PV, split_frequency)), axis=1)
    audio[f'BER_{type}_TV'] = audio.apply(lambda row: stat_measure(band_energy_ratio(row.TV, split_frequency)), axis=1)


def spectral_centr(audio):
    return librosa.feature.spectral_centroid(y=audio, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]


def spectral_centroid_extr(audio, stat_measure, type):
    audio[f'SC_{type}_AV'] = audio.apply(lambda row: stat_measure(spectral_centr(row.AV)), axis=1)
    audio[f'SC_{type}_MV'] = audio.apply(lambda row: stat_measure(spectral_centr(row.MV)), axis=1)
    audio[f'SC_{type}_PV'] = audio.apply(lambda row: stat_measure(spectral_centr(row.PV)), axis=1)
    audio[f'SC_{type}_TV'] = audio.apply(lambda row: stat_measure(spectral_centr(row.TV)), axis=1)


def spectral_bandwidth(audio):
    return \
        librosa.feature.spectral_bandwidth(y=audio[10:-10], sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[
            0]


def spectral_bandwidth_extr(audio, stat_measure, type):
    audio[f'BW_{type}_AV'] = audio.apply(lambda row: stat_measure(spectral_bandwidth(row.AV)), axis=1)
    audio[f'BW_{type}_MV'] = audio.apply(lambda row: stat_measure(spectral_bandwidth(row.MV)), axis=1)
    audio[f'BW_{type}_PV'] = audio.apply(lambda row: stat_measure(spectral_bandwidth(row.PV)), axis=1)
    audio[f'BW_{type}_TV'] = audio.apply(lambda row: stat_measure(spectral_bandwidth(row.TV)), axis=1)


def auto_cor(audio):
    return librosa.autocorrelate(audio)


def autocorrelation_extr(audio, stat_measure, type):
    audio[f'AC_{type}_AV'] = audio.apply(lambda row: stat_measure(auto_cor(row.AV)), axis=1)
    audio[f'AC_{type}_MV'] = audio.apply(lambda row: stat_measure(auto_cor(row.MV)), axis=1)
    audio[f'AC_{type}_PV'] = audio.apply(lambda row: stat_measure(auto_cor(row.PV)), axis=1)
    audio[f'AC_{type}_TV'] = audio.apply(lambda row: stat_measure(auto_cor(row.TV)), axis=1)


def mfccs(audio, n_mfcc):
    mfccs = librosa.feature.mfcc(y=audio, n_mfcc=n_mfcc, sr=SAMPLING_RATE)
    return np.mean(mfccs.T, axis=0)


def mfccs_extr(audio, n_mfcc=13):
    for index, row in audio.iterrows():

        AV_mfcc_list = mfccs(row.AV, n_mfcc).tolist()
        count = 0
        for mfcc in AV_mfcc_list:
            audio.loc[index, f'AV_mfcc_{count + 1}'] = mfcc
            count += 1

        MV_mfcc_list = mfccs(row.MV, n_mfcc).tolist()
        count = 0
        for mfcc in MV_mfcc_list:
            audio.loc[index, f'MV_mfcc_{count + 1}'] = mfcc
            count += 1

        PV_mfcc_list = mfccs(row.PV, n_mfcc).tolist()
        count = 0
        for mfcc in PV_mfcc_list:
            audio.loc[index, f'PV_mfcc_{count + 1}'] = mfcc
            count += 1

        TV_mfcc_list = mfccs(row.TV, n_mfcc).tolist()
        count = 0
        for mfcc in TV_mfcc_list:
            audio.loc[index, f'TV_mfcc_{count + 1}'] = mfcc
            count += 1
