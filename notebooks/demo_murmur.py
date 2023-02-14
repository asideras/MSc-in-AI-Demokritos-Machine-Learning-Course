import gradio as gr 
import pathlib
import pandas as pd
import librosa
import numpy as np
import feature_extraction_ML as fe
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.preprocessing import StandardScaler
from pickle import load

scaler = load(open('scaler.pkl', 'rb'))
loaded_model = load(open('final_model.pkl', 'rb'))


results_file = r"../classifiers_results/feature_selection_rfe_results.txt"
selected_features_list = 'rfe.txt'
f = open(results_file, "w")

selected_feats=[]
with open(r'../important_features/'+selected_features_list, 'r') as fp:
    for line in fp:
        x = line[:-1]
        selected_feats.append(x)


def predict(AV_audio,MV_audio,PV_audio,TV_audio):
    global selected_features,scaler,loaded_model
    num_of_samples = 5 * 22050
    df = pd.DataFrame(columns=['AV', 'MV', 'PV', 'TV'])
    
    AV_audio, _ = librosa.load(AV_audio)
    MV_audio, _ = librosa.load(MV_audio)
    PV_audio, _ = librosa.load(PV_audio)
    TV_audio, _ = librosa.load(TV_audio)
    
    #AV_audio=AV_audio[:num_of_samples]
    #MV_audio=MV_audio[:num_of_samples]
    #PV_audio=PV_audio[:num_of_samples]
    #TV_audio=TV_audio[:num_of_samples]
    
    """
    new_row = {
    'AV': AV_audio,
    'MV': MV_audio,
    'PV': PV_audio,
    'TV': TV_audio}
    
    df = df.append(new_row, ignore_index=True)
    """
    
    length = min(AV_audio.shape[0],MV_audio.shape[0],PV_audio.shape[0],TV_audio.shape[0])

    av_split = np.split(AV_audio, np.arange(length // 5, length, length // 5))
    mv_split = np.split(MV_audio, np.arange(length // 5, length, length // 5))
    pv_split = np.split(PV_audio, np.arange(length // 5, length, length // 5))
    tv_split = np.split(TV_audio, np.arange(length // 5, length, length // 5))

    for av, mv, pv, tv in zip(av_split, mv_split,pv_split,tv_split):
        new_row = {
        'AV': av,
        'MV': mv,
        'PV': pv,
        'TV': tv}

        df = df.append(new_row, ignore_index=True)
    
    
    
    

    #fe.set_len_extr(df,sec=5,augment=False)
    # stat ae
    fe.stat_amplitude_envelope_extr(df, np.mean, "mean")
    fe.stat_amplitude_envelope_extr(df, np.median, "median")
    fe.stat_amplitude_envelope_extr(df, np.std, "std")
    # perc ae
    fe.stat_amplitude_envelope_perc_extr(df, percentile=75)
    # stat rms
    fe.stat_rms_extr(df, np.mean, "mean")
    fe.stat_rms_extr(df, np.median, "median")
    fe.stat_rms_extr(df, np.std, "std")
    # perc rms
    fe.stat_rms_perc_extr(df, percentile=75)
    # max mag freq
    fe.max_frequency_extr(df)
    fe.total_energy_extr(df)
    # ZCR
    fe.zero_crossing_rate_extr(df)
    # onset detection
    fe.onset_detection_extr(df)
    # skewness and kurtosis
    fe.skewness_extr(df)
    fe.kurtosis_extr(df)
    # Band Energy Ratio
    fe.band_energy_ratio_extr(df, np.mean, "mean", 2000)
    fe.band_energy_ratio_extr(df, np.std, "std", 2000)
    # Spectral centroid
    fe.spectral_centroid_extr(df, np.mean, "mean")
    # Bandwidth
    fe.spectral_bandwidth_extr(df,  np.mean, "mean")
    # Autocorrelation
    fe.autocorrelation_extr(df,  np.mean, "mean")
    # MFCC's
    
    fe.mfccs_extr(df)
    df = df.drop(columns=[ 'AV', 'MV', 'PV', 'TV'])
    

    X = df[selected_feats] #extract only these features
    
    X=scaler.transform(X)
    result = loaded_model.predict(X)
    res = max(set(result), key=list(result).count)
    return 'Murmur Present' if res==1 else 'Murmur Absent'
    #return str(result[0])


demo = gr.Interface(
    fn=predict,
    inputs = [gr.Audio(type="filepath"), gr.Audio(type="filepath"), gr.Audio(type="filepath"),gr.Audio(type="filepath")],
    outputs=["text"]
    
)

demo.launch(inbrowser=True)