import requests
from tqdm import tqdm
import zipfile
import os
import numpy as np
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
from data.process.dataset_config import SAMPLING_RATE, DS_TRAIN, DS_EVAL, OFFSET_SAMPLE, NUM_SAMPLES_PER_FRAME, MIN_LENGTH_WAVEFORM
from wfdb.processing import resample_multichan
from data.process.preprocessing import is_beat, butter_bandpass_filter, change_resolution_labels, change_resolution_labels_with_peaks




def main():
    # the repo designed to have files live in /rebar/data/ecg/
    # downloadextract_ECGfiles()
    preprocess_ECGdata_perfile(change_resolution=False)


def downloadextract_ECGfiles(zippath="data/ecg.zip", targetpath="data/ecg_segmentation", redownload=False):
    if os.path.exists(targetpath) and redownload == False:
        print("ECG files already exist")
        return

    link = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
    print("Downloading ECG files (440 MB) ...")
    download_file(link, zippath)

    print("Unzipping ECG files ...")
    with zipfile.ZipFile(zippath,"r") as zip_ref:
        zip_ref.extractall(targetpath)
    os.remove(zippath)
    print("Done extracting and downloading")

def preprocess_ECGdata_perfile(ecgpath="data/ecg_segmentation", processedecgpath="data/ecg_segmentation/processed/records_npy", reprocess=False, change_resolution = False):
    # if os.path.exists(processedecgpath) and reprocess == False:
    #     print("ECG data has already been processed")
    #     return
    
    print("Processing ECG per recordings ...")
    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]

    os.makedirs(processedecgpath, exist_ok=True)            # Create dir to save npy records

    for record_id in sorted(record_ids):
        # Import recording and annotations
        record_path = os.path.join(ecgpath,"mit-bih-arrhythmia-database-1.0.0", record_id)
        record = wfdb.rdrecord(record_path)
        waveform = record.__dict__['p_signal']
        annotation = wfdb.rdann(record_path, 'atr')
        fs = record.fs
        # Resample to 250Hz
        if fs != SAMPLING_RATE:
            waveform, resampled_ann = resample_multichan(waveform, annotation, fs, SAMPLING_RATE)
            # Get exactly the same length for all waveforms
            waveform = waveform[:MIN_LENGTH_WAVEFORM, :]
            labels = resampled_ann.symbol
            sample = resampled_ann.sample

        # Apply filters
        filtered_waveform = butter_bandpass_filter(waveform)
        
        # Normalize
        filtered_waveform = filtered_waveform - np.mean(filtered_waveform, axis=0)
        filtered_waveform = filtered_waveform / np.std(filtered_waveform, axis=0)
        
        # Create labels
        padded_labels = np.zeros(len(waveform))
        for i,l in enumerate(labels):
            if i==len(labels)-1:
                padded_labels[sample[i]:] = is_beat(l)
            else:
                padded_labels[sample[i] - OFFSET_SAMPLE // 2: sample[i] + OFFSET_SAMPLE // 2] = is_beat(l)

        # Denoise and re-normalize
        filtered_waveform = denoiseECG_single(filtered_waveform, hz = SAMPLING_RATE)
        
        # Save recording
        record_save_path = os.path.join(processedecgpath, record_id)
        os.makedirs(record_save_path, exist_ok=True)
        np.save(record_save_path + f'/{record_id}.npy', filtered_waveform)
        # Save labels
        np.save(record_save_path + f'/{record_id}_labels.npy', padded_labels)




def denoiseECG_single(data, hz=250):

    """
    Parameters:
        data (np.array): 2D numpy array (time x channels)
    
    Returns:
        np.array: Normalized and denoised ECG signal.
    """
    data_filtered = np.empty(data.shape)
    
    for c in range(data.shape[1]):
        data_filtered[:, c] = nk.ecg_clean(data[:, c], sampling_rate=hz)
    
    # Normalize
    feature_means = np.mean(data_filtered, axis=0)
    feature_std = np.std(data_filtered, axis=0)
    data_normalized = (data_filtered - feature_means) / feature_std
    
    return data_normalized


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename



if __name__ == "__main__":
    main()