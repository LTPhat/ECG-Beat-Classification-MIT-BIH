import numpy as np
import pandas as pd
from scipy.signal import resample
from collections import Counter
import matplotlib.pyplot as plt
import torch
from data.process.pann_tomkins import Pan_tompkins, get_final_peaks
from networks.model import ResNetClassifier
from data.process.dataset_config import NUM_SAMPLES_PER_FRAME, OFFSET_SAMPLE
from data.process.preprocessing import butter_bandpass_filter, denoiseECG_single
import argparse


# --------- Parameters ---------
SAMPLING_RATE = 250
SEGMENT_LENGTH = 250
HALF_LEN = SEGMENT_LENGTH // 2 - 20

# --------- Class Map ---------
classes_dict = {
    0: 'Normal (N)',
    1: 'Supraventricular ectopic (S)',
    2: 'Ventricular ectopic (V)',
    3: 'Fusion (F)',
    4: 'Unknown (Q)'
}
# --------- Read ECG CSV ---------
def load_ecg_from_csv(csv_path, column_name=['Lead1', 'Lead2']):
    df = pd.read_csv(csv_path)
    ecg_signal = df[column_name].values.astype(np.float32)
    return ecg_signal

# --------- Main Inference Pipeline ---------
def classify_ecg_beats(csv_path, model, lead, column_name=['Lead1', 'Lead2'], saved_model_path='./logs/best_model.pth'):
    ecg_signal = load_ecg_from_csv(csv_path, column_name)
    # Resample the signal to 250 Hz
    original_rate = 360
    target_rate = 250

    # calculate new number of samples
    duration = len(ecg_signal) / original_rate
    num_target_samples = int(duration * target_rate)

    ecg_signal = resample(ecg_signal, num_target_samples)   # (438111, 2)
   
    # Filterin
    ecg_filtered = butter_bandpass_filter(ecg_signal, 0.5, 45, SAMPLING_RATE, order=2)
    ecg_filtered = denoiseECG_single(ecg_filtered, SAMPLING_RATE)

    # Normalization
    ecg_filtered = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)

    # Peak detection
    pan = Pan_tompkins(ecg_filtered[:, lead], SAMPLING_RATE)  # Apply to first lead
    integrated = pan.fit()
    peaks = get_final_peaks(integrated)

    # Beat segmentation
    segments = []
    for r in peaks:
        if r - HALF_LEN < 0 or r + HALF_LEN > len(ecg_filtered):
            continue
        segment = ecg_filtered[r - HALF_LEN : r + HALF_LEN, 0]  # Get only first lead
        segments.append(segment)

    if not segments:
        print("No valid beats detected.")
        return {}

    # Prepare segments for model input
    segments = np.array(segments)  
    segments = torch.tensor(segments, dtype=torch.float32)

    # Call model results
    model.load_state_dict(torch.load('./logs/best_model.pth'))
    model.eval()

    # Model expects input shape accordingly
    predictions = model(segments)
    predictions = torch.argmax(predictions, dim=1).cpu().numpy()

    # Convert predictions to class labels
    predictions = [classes_dict.get(pred, f"{pred} beat") for pred in predictions]

    # Count predictions
    counts = Counter(predictions)
    results = {classes_dict.get(k, f"Beat {k}"): v for k, v in counts.items()}
    
    return results

def main():

    argparser = argparse.ArgumentParser(description="ECG Beat Classification")
    argparser.add_argument('--csv_path', type=str, default='data/ecg_classification/processed/test_csv/105.csv', help='Path to the CSV file')
    argparser.add_argument('--model_path', type=str, default='./logs/best_model.pth', help='Path to the trained model')
    argparser.add_argument('--lead', type=int, default=0, help='Lead to use for classification (0 or 1)')
    args = argparser.parse_args()

    csv_path = args.csv_path
    model_path = args.model_path
    lead = args.lead

    # Load the model
    model = ResNetClassifier(input_size=NUM_SAMPLES_PER_FRAME, num_classes=5)
    # Path to the CSV file
    model.load_state_dict(torch.load(model_path))
    # Classify beats
    results = classify_ecg_beats(csv_path, model, lead=lead, saved_model_path=model_path)
    print("Classification Results:", results)

    return results

if __name__ == "__main__":
    main()
