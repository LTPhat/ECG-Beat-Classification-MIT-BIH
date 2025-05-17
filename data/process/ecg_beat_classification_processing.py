import requests
from tqdm import tqdm
import zipfile
import os
import numpy as np
import random
import wfdb
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from data.process.dataset_config import SAMPLING_RATE
from wfdb.processing import resample_multichan
from data.process.preprocessing import butter_bandpass_filter, denoiseECG_single
from pann_tomkins import Pan_tompkins, get_final_peaks
from imblearn.over_sampling import SMOTE
from collections import Counter


# Constants
SAMPLING_RATE = 250  # Target sampling rate in Hz
# Define train/test split following recommended MIT-BIH protocol
# DS1 (training): 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
# DS2 (testing): 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234
train_records = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207',
                  '208','209', '215', '220', '223', '230']
test_records =['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']


# AAMI Beat Classes mapping according to MIT-BIH annotations
# N: Normal beats (N, L, R, e, j)
# S: Supraventricular ectopic beats (A, a, J, S)
# V: Ventricular ectopic beats (V, E)
# F: Fusion beats (F)
# Q: Unknown beats (/, f, Q)
classes_dict = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular ectopic
    'V': 2, 'E': 2,                          # Ventricular ectopic
    'F': 3,                                  # Fusion
    '/': 4, 'f': 4, 'Q': 4                   # Unknown
}

#=========================================================
def main():
    # the repo designed to have files live in /rebar/data/ecg/
    # downloadextract_ECGfiles()
    preprocess_ECGdata()
    save_test_data_csv(test_records=test_records, ecgpath="data/ecg", csv_save_dir="data/ecg_classification/processed/test_csv")

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
def downloadextract_ECGfiles(zippath="data/ecg.zip", targetpath="data/ecgs", redownload=False):
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

def preprocess_ECGdata(ecgpath="data/ecg", processedecgpath="data/ecg_classification/processed", augment=False):
    """
    Preprocesses the ECG data for classification (train only)
    """
    
    # if os.path.exists(processedecgpath) and reprocess == False:
    #     print("ECG data has already been processed")
    #     return
    if not os.path.exists(processedecgpath):
        os.makedirs(processedecgpath, exist_ok=True)

    print("Processing ECG files ...")
    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]
    
    all_ecgs = []
    all_labels = []
    all_names = []
    # Loop through records to create ecgs and labels
    for record_id in record_ids:
        record_path = os.path.join(ecgpath,"mit-bih-arrhythmia-database-1.0.0", record_id)
        record = wfdb.rdrecord(record_path)
        waveform = record.__dict__['p_signal']
        annotation = wfdb.rdann(record_path, 'atr')
        sample = annotation.sample
        labels = annotation.symbol
        fs = record.fs
        # Resample to 250Hz
        if fs != SAMPLING_RATE:
            waveform, resampled_ann = resample_multichan(waveform, annotation, fs, SAMPLING_RATE)
            labels = resampled_ann.symbol
            sample = resampled_ann.sample
            num_seconds = 5
            num_samples = SAMPLING_RATE * num_seconds

        waveform_before_filter = waveform.copy()
        # -------------------Bandpass Filter (0.5 - 50 Hz)--------
        waveform_filtered = butter_bandpass_filter(waveform, lowcut=0.5, highcut=45, fs=SAMPLING_RATE, order=2)
        waveform_filtered = denoiseECG_single(waveform_filtered, SAMPLING_RATE)
        # -------------------Normalization------------------------
        feature_means = np.mean(waveform_filtered, axis=0)
        feature_std = np.std(waveform_filtered, axis=0)
        waveform_filtered = (waveform_filtered - feature_means) / feature_std
        
        # -------------------Plotting-----------------------------
        # num_seconds = 5
        # num_samples = SAMPLING_RATE * num_seconds

        # time_axis = np.arange(num_samples) / SAMPLING_RATE
        # plt.figure(figsize=(10, 6))

        # # Original waveform
        # plt.subplot(3, 1, 1)
        # plt.plot(time_axis, waveform_before_filter[num_samples: num_samples * 2, 0], color='gray')
        # plt.title("Waveform Before Filtering")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")
        # plt.grid(True)

        # # After filtering
        # plt.subplot(3, 1, 2)
        # plt.plot(time_axis, waveform_filtered[num_samples: num_samples * 2, 0], color='blue')
        # plt.title("Waveform After Bandpass Filter (0.5–50 Hz)")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")
        # plt.grid(True)

        # # After normalization
        # plt.subplot(3, 1, 3)
        # plt.plot(time_axis, waveform_normalized[:num_samples, 0], color='green')
        # plt.title("Waveform After Normalization")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Normalized Amplitude")
        # plt.grid(True)

        # plt.tight_layout()
        # plt.show()

        segment_length = 250  # Total segment length (1s if sampling rate is 250 Hz)
        half_len = segment_length // 2 

        segments = []
        segment_labels = []
        # Create beat -based labels around peaks
        for i, r in enumerate(sample):
            if labels[i] not in classes_dict:
                continue  # Skip unrecognized beats
            
            # Skip if segment would go out of bounds
            if r - half_len < 0 or r + half_len > len(waveform_filtered):
                continue

            # Extract segment around R-peak
            segment = waveform_filtered[r - half_len : r + half_len, 0]  # Shape: (250, 1) # One lead
            # plt.plot(segment)
            # plt.show()
            label = classes_dict[labels[i]]

            segments.append(segment)            
            segment_labels.append(label)
    
        all_ecgs.extend(segments)
        all_labels.extend(segment_labels)
        all_names.extend([record_id] * len(segment_labels))

    all_ecgs = np.array(all_ecgs)
    all_labels = np.array(all_labels)
    all_names = np.array(all_names)

    
    # Get boolean masks
    train_set = set(train_records)
    train_mask = np.isin(all_names, list(train_set))
    test_set = set(test_records)
    test_mask = np.isin(all_names, list(test_set))
    # Apply masks to split the data into train-val-test
    train_data = all_ecgs[train_mask]
    train_labels = all_labels[train_mask]
    train_names = all_names[train_mask]

    test_data = all_ecgs[test_mask]
    test_labels = all_labels[test_mask]
    test_names = all_names[test_mask]


    # Save raw data before segmentation
    np.save(os.path.join(processedecgpath, "train_data.npy"), train_data)
    np.save(os.path.join(processedecgpath, "train_labels.npy"), train_labels)
    np.save(os.path.join(processedecgpath, "train_names.npy"), train_names)

    if augment:
        train_data = np.load(os.path.join(processedecgpath, "train_data.npy"))
        train_labels = np.load(os.path.join(processedecgpath, "train_labels.npy"))
        train_data_aug, train_labels_aug = SMOTE_augmentation(train_data, train_labels, target_counts={1: 4000, 2: 4000, 3: 4000})
        np.save(os.path.join(processedecgpath, "train_data_aug.npy"), train_data_aug)
        np.save(os.path.join(processedecgpath, "train_labels_aug.npy"), train_labels_aug)
        print("Saved augmented data!")

    np.save(os.path.join(processedecgpath, "test_data.npy"), test_data)
    np.save(os.path.join(processedecgpath, "test_labels.npy"), test_labels)
    np.save(os.path.join(processedecgpath, "test_names.npy"), test_names)
    print("Done Preprocessing and Train Dataset Generation")


def preprocess_ECGdata_test(ecgpath="data/ecg", processedecgpath="data/ecg_classification/processed", reprocess=False):
    if not os.path.exists(processedecgpath):
        os.makedirs(processedecgpath, exist_ok=True)

    print("Processing ECG files ...")
    if not os.path.exists(processedecgpath):
        os.makedirs(processedecgpath, exist_ok=True)

    print("Processing ECG files ...")
    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]
    
    all_ecgs = []
    all_labels = []
    all_names = []
    # Loop through records to create ecgs and labels
    for record_id in record_ids:
        if record_id not in test_records:
            continue        # If not in training set, skip
        record_path = os.path.join(ecgpath,"mit-bih-arrhythmia-database-1.0.0", record_id)
        record = wfdb.rdrecord(record_path)
        waveform = record.__dict__['p_signal']
        annotation = wfdb.rdann(record_path, 'atr')
        labels = annotation.symbol
        sample = annotation.sample
        fs = record.fs
        # Resample to 250Hz
        if fs != SAMPLING_RATE:
            waveform, resampled_ann = resample_multichan(waveform, annotation, fs, SAMPLING_RATE)
            labels = resampled_ann.symbol
            sample = resampled_ann.sample
        # -------------------Bandpass Filter (0.5 - 50 Hz)--------
        waveform_filtered = butter_bandpass_filter(waveform, lowcut=0.5, highcut=45, fs=SAMPLING_RATE, order=2)
        waveform_filtered = denoiseECG_single(waveform_filtered, SAMPLING_RATE)
        # -------------------Normalization------------------------
        feature_means = np.mean(waveform_filtered, axis=0)
        feature_std = np.std(waveform_filtered, axis=0)
        waveform_filtered = (waveform_filtered - feature_means) / feature_std
        
        # Use Pann-Tompkins to get peaks
        pan_tompkins = Pan_tompkins(waveform_filtered[:, 0], SAMPLING_RATE)
        integrated_signal = pan_tompkins.fit()
        peaks = get_final_peaks(integrated_signal)
        # Segment beats based on detected peaks
        segment_length = 250
        half_len = segment_length // 2 - 20
        segments = []
        segment_labels = []
        print(len(peaks), len(labels))
        # Create beat -based labels around peaks
        for i, r in enumerate(peaks):
            if i >= len(labels):
                break
            if labels[i] not in classes_dict:
                continue  # Skip unrecognized beats
            
            # Skip if segment would go out of bounds
            if r - half_len < 0 or r + half_len > len(waveform_filtered):
                continue

            # Extract segment around R-peak
            segment = waveform_filtered[r - half_len : r + half_len, 0]  # Shape: (250, 1) # One lead
            label = classes_dict[labels[i]]
            plt.plot(segment)
            plt.show()
            segments.append(segment)            
            segment_labels.append(label)

        all_ecgs.extend(segments)
        all_labels.extend(segment_labels)   
        all_names.extend([record_id] * len(segment_labels))
    all_ecgs = np.array(all_ecgs)
    all_labels = np.array(all_labels)
    all_names = np.array(all_names)

    np.save(os.path.join(processedecgpath, "test_data.npy"), all_ecgs)
    np.save(os.path.join(processedecgpath, "test_labels.npy"),all_labels)
    np.save(os.path.join(processedecgpath, "test_names.npy"), all_names)

    print("Done Preprocessing Test Dataset Generation")

def save_test_data_csv(test_records, ecgpath="data/ecg", csv_save_dir="data/ecg_classification/processed/csv_test"):   
                       
    if not os.path.exists(csv_save_dir):
        os.makedirs(csv_save_dir, exist_ok=True)

    print("Converting test ECG files to CSV for demo...")
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]
    # Loop through records to load and save waveform
    for record_id in record_ids:
        record_path = os.path.join(ecgpath,"mit-bih-arrhythmia-database-1.0.0", record_id)
        if record_id not in test_records:
            continue
        try:
            # Load ECG waveform and annotations
            record = wfdb.rdrecord(record_path)
            waveform = record.p_signal  # shape: (n_samples, n_channels)
            
            # Convert to DataFrame
            df = pd.DataFrame(waveform, columns=['Lead1', 'Lead2'])

            # Save to CSV
            csv_filename = os.path.join(csv_save_dir, f"{record_id}.csv")
            df.to_csv(csv_filename, index=False)

            print(f"Saved waveform to {csv_filename}")

        except Exception as e:
            print(f"Failed to process {record_id}: {e}")



def denoiseECG(data, hz=250):
    data_filtered = np.empty(data.shape)
    for n in range(data_filtered.shape[0]):
        for c in range(data.shape[1]):
            newecg = nk.ecg_clean(data[n,c,:], sampling_rate=hz)
            data_filtered[n,c] = newecg
    data = data_filtered

    feature_means = np.mean(data, axis=(2))
    feature_std = np.std(data, axis=(2))
    data = (data - feature_means[:, :, np.newaxis]) / (feature_std)[:, :, np.newaxis]

    # data = np.transpose(data, (0,2,1))

    return data

def SMOTE_augmentation(train_data, train_labels, target_counts):
    current_counts = Counter(train_labels)
    sampling_strategy = {}
    for cls, target in target_counts.items():
        if current_counts[cls] < target:
            sampling_strategy[cls] = target

    # Determine the smallest class count
    min_class_size = min([current_counts[cls] for cls in sampling_strategy.keys()])
    k_neighbors = max(1, min(min_class_size - 1, 5))  
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k_neighbors)
    train_data_aug, train_labels_aug = sm.fit_resample(train_data, train_labels)
    return train_data_aug, train_labels_aug


if __name__ == "__main__":
    main()
# %%









