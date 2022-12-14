import pandas as pd
import os
import librosa
import warnings

warnings.filterwarnings('ignore')


class DataLoaderML:
    def __init__(self):
        self.heart_audio_path = os.path.join(os.getcwd(),
                                             r"the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data")
        self.annotations_path = os.path.join(os.getcwd(), r"the-circor-digiscope-phonocardiogram-dataset-1.0.3")
        self.annotations = pd.read_csv(os.path.join(self.annotations_path, "training_data.csv"))[
            ['Patient ID', 'Murmur']]
        self.audios = pd.DataFrame(columns=['Patient_ID', 'AV', 'MV', 'PV', 'TV', 'MURMUR'])

    def __create_dataset(self, fulldataset):
        print(f"Please wait. Out of {self.annotations.shape[0]} patient audios,")
        for index, row in self.annotations.iterrows():
            patient_id = row['Patient ID']
            murmur = row['Murmur']
            file_AV_path = os.path.join(self.heart_audio_path, f"{patient_id}_AV.wav")
            file_MV_path = os.path.join(self.heart_audio_path, f"{patient_id}_MV.wav")
            file_PV_path = os.path.join(self.heart_audio_path, f"{patient_id}_PV.wav")
            file_TV_path = os.path.join(self.heart_audio_path, f"{patient_id}_TV.wav")

            try:
                AV_audio, _ = librosa.load(file_AV_path)
            except:
                AV_audio = None
            try:
                MV_audio, _ = librosa.load(file_MV_path)
            except:
                MV_audio = None

            try:
                PV_audio, _ = librosa.load(file_PV_path)
            except:
                PV_audio = None

            try:
                TV_audio, _ = librosa.load(file_TV_path)
            except:
                TV_audio = None

            new_row = {'Patient_ID': patient_id,
                       'AV': AV_audio,
                       'MV': MV_audio,
                       'PV': PV_audio,
                       'TV': TV_audio,
                       'MURMUR': murmur}

            self.audios = self.audios.append(new_row, ignore_index=True)

            if index % 10 == 0:
                print(f"{index} patient audios have been loaded")

            if not fulldataset and index == 4:
                break

    def get_audios(self, fullDataset=True):
        self.__create_dataset(fullDataset)
