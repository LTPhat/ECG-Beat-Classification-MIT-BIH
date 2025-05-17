import requests
from tqdm import tqdm
import zipfile
import os
import numpy as np
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
from data.process.dataset_config import SAMPLING_RATE, DS_TRAIN, DS_EVAL, OFFSET_SAMPLE, NUM_SAMPLES_PER_FRAME
from wfdb.processing import resample_multichan
from data.process.preprocessing import is_beat, butter_bandpass_filter, change_resolution_labels, change_resolution_labels_with_peaks

def main():
    # the repo designed to have files live in /rebar/data/ecg/
    downloadextract_ECGfiles()
    # preprocess_ECGdata(change_resolution=False)


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

def preprocess_ECGdata(ecgpath="data/ecg_segmentation", processedecgpath="data/ecg_segmentation/processed", reprocess=False, change_resolution = False):
    # if os.path.exists(processedecgpath) and reprocess == False:
    #     print("ECG data has already been processed")
    #     return
    
    print("Processing ECG files ...")
    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "mit-bih-arrhythmia-database-1.0.0")) if '.dat' in file]
    
    all_ecgs = []
    all_labels = []
    all_names = []
    # Loop through records to create ecgs and labels
    for record_id in sorted(record_ids):
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


        # Apply filters
        filtered_waveform = butter_bandpass_filter(waveform, order = 2)
        # Denoise for one-recording---------------
        filtered_waveform = denoiseECG_single(filtered_waveform, SAMPLING_RATE)

        # --calculate First derivative (optional)
        first_derivative = np.diff(filtered_waveform, axis=0, prepend=filtered_waveform[0:1]) * SAMPLING_RATE
        # second_derivative = np.diff(first_derivative, axis=0, prepend=first_derivative[0:1]) * SAMPLING_RATE
        filtered_waveform = -first_derivative
        # # filtered_waveform = -second_derivative
        
        # ------------Testing -----------------
        # check_range = [[100, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]]
        # plt.plot(waveform[check_range[2][0]:check_range[2][1], 0])
        # plt.plot(filtered_waveform[1000:1500, 0], color = 'green')
        # filtered_waveform = butter_bandpass_filter(inv_first_derivative)
        # plt.plot(filtered_waveform[check_range[2][0]:check_range[2][1], 0], color = 'purple')
        # plt.show()
        # exit()


        # # # Normalize for one recording (optional - already have at denoiseECG function) 
        # filtered_waveform = filtered_waveform - np.mean(filtered_waveform, axis=0)
        # filtered_waveform = filtered_waveform / np.std(filtered_waveform, axis=0)
        
        # ---------Create labels-------------------
        padded_labels = np.zeros(len(waveform))
        for i,l in enumerate(labels):
            if i==len(labels)-1:
                padded_labels[sample[i]:] = is_beat(l)
            else:
                padded_labels[sample[i] - OFFSET_SAMPLE // 2: sample[i] + OFFSET_SAMPLE // 2] = is_beat(l)

        
        # Plot ECG (optional)
        # if record_id == '222':
        #     check_ranges = [[0, 50000], [50000, 100000], [100000, 150000], [150000, 200000], [200000, 250000], [250000, 300000]]
        #     for range in check_ranges:
        #         print("Plotting ECG for " + record_id, range)
        #         plot_ecg_predictions(filtered_waveform=filtered_waveform[range[0]:range[1],:], val_labels=padded_labels[range[0]:range[1]], val_predictions=padded_labels[range[0]:range[1]], record_id=record_id)
        
        all_labels.append(padded_labels)
        all_ecgs.append(filtered_waveform[:,:].T)
        all_names.append(record_id)

    all_names = np.array(all_names)
    signal_lens = [sig.shape[1] for sig in all_ecgs]
    all_ecgs = np.array([sig[:,:min(signal_lens)] for sig in all_ecgs])
    all_labels = np.array([label[:min(signal_lens)] for label in all_labels])
    
    train_data = np.array([all_ecgs[idx] for idx, record_id in enumerate(all_names) if int(record_id) in DS_TRAIN])
    train_labels = np.array([all_labels[idx] for idx, record_id in enumerate(all_names) if int(record_id) in DS_TRAIN])
    train_names = np.array([[record_id for idx, record_id in enumerate(all_names) if int(record_id) in DS_TRAIN]])


    val_data = np.array([all_ecgs[idx] for idx, record_id in enumerate(all_names) if int(record_id) in DS_EVAL])
    val_labels = np.array([all_labels[idx] for idx, record_id in enumerate(all_names) if int(record_id) in DS_EVAL])    
    val_names = np.array([[record_id for idx, record_id in enumerate(all_names) if int(record_id) in DS_EVAL]])

    # Normalize ecgs aand changes it to be batch,time, channel   -- Denoise and Normalize for the whole dataset
    train_data, val_data = denoiseECG(train_data), denoiseECG(val_data)
    print("Denoise and Normalize completed!")

    # # Save ecgs to file
    os.makedirs(processedecgpath, exist_ok=True)

    np.save(os.path.join(processedecgpath, "train_data.npy"), train_data)
    np.save(os.path.join(processedecgpath, "train_labels.npy"), train_labels)
    np.save(os.path.join(processedecgpath, "train_names.npy"), train_names)

    np.save(os.path.join(processedecgpath, "val_data.npy"), val_data)
    np.save(os.path.join(processedecgpath, "val_labels.npy"), val_labels)
    np.save(os.path.join(processedecgpath, "val_names.npy"), val_names)

    # # Process subseq dataset
    T = train_data.shape[1]                                                         # bc its been transposed
    subseq_size= 2500
    train_data = np.stack(np.split(train_data[:, :subseq_size * (T // subseq_size), :], (T // subseq_size), 1), axis=1)
    train_data = np.reshape(train_data, (-1, train_data.shape[2], train_data.shape[3]))
    
    train_labels = np.stack(np.split(train_labels[:, :subseq_size * (T // subseq_size)], (T // subseq_size), -1), axis=1).astype(int)
    train_labels = np.reshape(train_labels, (-1, train_labels.shape[2]))
    train_names = np.repeat(train_names, (T // subseq_size))


    # Split into 0.125s chunks
    if change_resolution:
        train_labels = change_resolution_labels_with_peaks(train_data, train_labels, NUM_SAMPLES_PER_FRAME)

    label_sq_size = train_labels.shape[1]

    np.save(os.path.join(processedecgpath, 'train_data_subseq.npy'), train_data)
    np.save(os.path.join(processedecgpath, 'train_labels_subseq.npy'), train_labels)
    np.save(os.path.join(processedecgpath, "train_names_subseq.npy"), train_names)

    T = val_data.shape[1]
    val_data = np.stack(np.split(val_data[:, :subseq_size* (T // subseq_size), :], (T // subseq_size), 1), axis=1)
    val_data = np.reshape(val_data, (-1, val_data.shape[2], val_data.shape[3]))
    val_labels = np.stack(np.split(val_labels[:, :subseq_size * (T // subseq_size)], (T // subseq_size), -1), axis=1).astype(int)
    val_labels = np.reshape(val_labels, (-1, val_labels.shape[2]))
    val_names = np.repeat(val_names, (T // subseq_size))


    if change_resolution:   
        val_labels = change_resolution_labels_with_peaks(val_data, val_labels, NUM_SAMPLES_PER_FRAME=NUM_SAMPLES_PER_FRAME)
    label_sq_size = val_labels.shape[1]

    np.save(os.path.join(processedecgpath, "val_data_subseq.npy"), val_data)
    np.save(os.path.join(processedecgpath, "val_labels_subseq.npy"), val_labels)
    np.save(os.path.join(processedecgpath, "val_names_subseq.npy"), val_names)


    print("========= SAVE DATASET DONE ===========")


def denoiseECG_single(data, hz=250):

    """
    Denoise ECG for one recording
    Parameters:
        data (np.array): 2D numpy array (time x channels)
    
    Returns:
        np.array: Normalized and denoised ECG signal.
    """
    data_filtered = np.empty(data.shape)
    
    for c in range(data.shape[1]):
        data_filtered[:, c] = nk.ecg_clean(data[:, c], sampling_rate=hz)
    
    # Normalize
    # feature_means = np.mean(data_filtered, axis=0)
    # feature_std = np.std(data_filtered, axis=0)
    # data_normalized = (data_filtered - feature_means) / feature_std
    data_normalized = data_filtered
    return data_normalized


def denoiseECG(data, hz=250):
    """
    Denoise and Normalize ECG (for the whole dataset)
    """
    data_filtered = np.empty(data.shape)
    for n in range(data_filtered.shape[0]):
        for c in range(data.shape[1]):
            newecg = nk.ecg_clean(data[n,c,:], sampling_rate=hz)
            data_filtered[n,c] = newecg
    data = data_filtered

    feature_means = np.mean(data, axis=(2))
    feature_std = np.std(data, axis=(2))
    data = (data - feature_means[:, :, np.newaxis]) / (feature_std)[:, :, np.newaxis]

    data = np.transpose(data, (0,2,1))

    return data


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