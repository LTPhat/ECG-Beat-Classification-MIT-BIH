from data.process.dataset_config import NUM_SAMPLES_PER_FRAME, OFFSET_SAMPLE
import collections
from collections import Counter
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt
import neurokit2 as nk

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


def change_resolution_labels_with_peaks(data, label, NUM_SAMPLES_PER_FRAME):
    new_labels = []
    
    for seq_idx, subseq in enumerate(label):
        calib_labels = []
        for i in range(0, len(subseq) - NUM_SAMPLES_PER_FRAME + 1, NUM_SAMPLES_PER_FRAME):
            most_common_label = int(Counter(subseq[i : i + NUM_SAMPLES_PER_FRAME]).most_common(1)[0][0])
            calib_labels.append(most_common_label)

        new_calib_labels = calib_labels[:]  # Copy to modify
        i = 0
        while i < len(calib_labels):
            if calib_labels[i] == 1:
                group_indices = [i]
                
                j = i + 1
                while j < len(calib_labels) and calib_labels[j] == 1:
                    group_indices.append(j)
                    j += 1
                
                if len(group_indices) > 1:
                    max_value = -np.inf
                    best_index = group_indices[0]
                    
                    for idx in group_indices:
                        start = idx * NUM_SAMPLES_PER_FRAME
                        end = min(start + NUM_SAMPLES_PER_FRAME, len(data[seq_idx]))
                        block_data = data[seq_idx][start:end]
                        
                        max_block_value = np.max(np.abs(block_data))
                        
                        # Compear current peak of current block with previous best block
                        if max_block_value > max_value:
                            max_value = max_block_value
                            best_index = idx
                    
                    # Set Only the Best Block to '1' other will be zero
                    for idx in group_indices:
                        new_calib_labels[idx] = 1 if idx == best_index else 0
                
                i = j  # Skip processed blocks
            else:
                i += 1
        new_labels.append(new_calib_labels)
    
    return np.array(new_labels)


def change_resolution_labels(tensor, NUM_SAMPLES_PER_FRAME):
    new_tensor = []
    for subseq in tensor:
        calib_labels = []
        for i in range(0, len(subseq) - NUM_SAMPLES_PER_FRAME + 1, NUM_SAMPLES_PER_FRAME):  
            most_common_label = int(Counter(subseq[i : i + NUM_SAMPLES_PER_FRAME]).most_common(1)[0][0])
            calib_labels.append(most_common_label)
        new_tensor.append(calib_labels)
    return np.array(new_tensor)


# def butter_bandpass(lowcut=5, highcut=50, fs=250, order=2):
#     return butter(order, [lowcut, highcut], fs=fs, btype='band')

# def butter_bandpass_filter(data, lowcut=5, highcut=50, fs=250, order=2):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y
def butter_bandpass(lowcut=0.5, highcut=50, fs=250, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut=0.5, highcut=50, fs=250, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # If multi-channel, apply along each channel
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return np.array([filtfilt(b, a, data[:, ch]) for ch in range(data.shape[1])]).T


def is_beat(annotation):
    beat_labels = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'E']
    return 1 if annotation in beat_labels else 0


def enforce_min_spacing(padded_labels, min_distance=100):
    """Ensure consecutive 1s in padded_labels are at least min_distance apart using middle points of 1s."""
    
    # Find all consecutive regions of 1s
    beat_regions = []
    in_beat = False
    
    for i in range(len(padded_labels)):
        if padded_labels[i] == 1 and not in_beat:
            start = i
            in_beat = True
        elif padded_labels[i] == 0 and in_beat:
            end = i - 1
            in_beat = False
            middle = (start + end) // 2  # Middle point of the 1s region
            beat_regions.append(middle)

    if in_beat:  # Handle case where the last segment extends to the end
        middle = (start + len(padded_labels) - 1) // 2
        beat_regions.append(middle)

    # Keep only regions that are min_distance apart
    filtered_indices = [beat_regions[0]]
    for i in range(1, len(beat_regions)):
        if beat_regions[i] - filtered_indices[-1] >= min_distance:
            filtered_indices.append(beat_regions[i])

    # Reset padded_labels and mark only valid beats with sample_offset
    new_labels = np.zeros_like(padded_labels)
    for index in filtered_indices:
        if index < OFFSET_SAMPLE:
            new_labels[:index + OFFSET_SAMPLE] = 1
        elif index >= len(padded_labels) - OFFSET_SAMPLE:
            new_labels[index - OFFSET_SAMPLE:] = 1
        else:
            new_labels[index - OFFSET_SAMPLE: index + OFFSET_SAMPLE] = 1

    return new_labels



def plot_waveform(waveform, filtered_waveform, fs=250, title="ECG Signal Before and After Filtering"):
    """
    Plot the original and filtered ECG waveforms.
    
    Parameters:
    """
    time = np.arange(waveform.shape[0]) / fs
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, waveform[:, 0], label="Original", alpha=0.7)
    plt.plot(time, filtered_waveform[:, 0], label="Filtered")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_ecg_predictions(filtered_waveform, val_predictions, val_labels, record_id, sample_rate=250, num_seconds=10):
    """
    Plots filtered ECG waveform, predictions, and labels, marking peaks in predicted beats.
 
    """
    num_samples = num_seconds * sample_rate
    time_axis = np.linspace(0, num_seconds, num_samples)

    # Get the first channel only
    ecg_signal = filtered_waveform[:num_samples, 0]
    predictions = val_predictions[:num_samples]
    labels = val_labels[:num_samples]

    labels_peaks = get_peaks(ecg_signal, labels)
    predictions_peaks = get_peaks(ecg_signal, predictions)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot ECG waveform
    axs[0].plot(time_axis, ecg_signal, color='black', label="Filtered ECG")
    axs[0].scatter(np.array(labels_peaks) / sample_rate, ecg_signal[labels_peaks], color='red', marker='o', label="Label Peaks")
    axs[0].scatter(np.array(predictions_peaks) / sample_rate, ecg_signal[predictions_peaks], color='green', marker='x', label="Predicted Peaks")
    axs[0].set_ylabel("ECG Signal")
    axs[0].legend()

    # Plot Ground Truth Labels
    axs[1].plot(time_axis, labels, color='blue', linestyle='dashed', label="True Labels")
    axs[1].scatter(np.array(labels_peaks) / sample_rate, np.ones(len(labels_peaks)), color='red', marker='o', label="Label Peaks")
    axs[1].set_ylabel("True Labels (0/1)")
    axs[1].legend()

    # Plot Model Predictions
    axs[2].plot(time_axis, predictions, color='red', linestyle='dotted', label="Predictions")
    axs[2].scatter(np.array(predictions_peaks) / sample_rate, np.ones(len(predictions_peaks)), color='green', marker='x', label="Predicted Peaks")
    axs[2].set_ylabel("Predictions (0/1)")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].legend()

    plt.suptitle(f"ECG Signal, Labels, and Predictions for Record {record_id}", fontsize=16)

    plt.tight_layout()
    plt.show()


def get_peaks(ecg_signal, preferences):
    """
    Get the position of peak or prediction or labels
    """
    peaks = []
    in_peak = False
    for i in range(len(preferences)):
        if preferences[i] == 1:
            if not in_peak:
                start = i  # Start of a detected beat
                in_peak = True
            # Find max within detected 1s
            if i == len(preferences) - 1 or preferences[i + 1] == 0:
                peak_idx = start + np.argmax(ecg_signal[start:i+1])     # Peak index --> With max absolute value
                peaks.append(peak_idx)
                in_peak = False

    return peaks