import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from scipy.signal import resample, find_peaks
from api import classify_ecg_beats, load_ecg_from_csv
from networks.model import ResNetClassifier
from data.process.pann_tomkins import Pan_tompkins, get_final_peaks
from data.process.preprocessing import butter_bandpass_filter, denoiseECG_single
from data.process.dataset_config import SAMPLING_RATE

classes_dict = {
    0: 'Normal (N)',
    1: 'Supraventricular ectopic (S)',
    2: 'Ventricular ectopic (V)',
    3: 'Fusion (F)',
    4: 'Unknown (Q)'
}
st.set_page_config(
page_title="ECG Beat Classification App",
layout="wide",
initial_sidebar_state="auto",
menu_items={
    }
)
title_style = """
        <style>
        .title {
            text-align: center;
            font-size: 45px;
        }
        </style>
        """
st.markdown(
title_style,
unsafe_allow_html=True
)
title  = """
<h1 class = "title" >ECG Beat Classification App</h1>
</div>
"""
st.markdown(title,
            unsafe_allow_html=True)

def visualize_ecg_signal(file_name, ecg_signal, title="ECG Signal", xlabel="Samples", ylabel="Amplitude"):
    # ============== Visualization ==============
    # ===============Plot raw ECG signal first 10 seconds
    st.subheader("Raw ECG Signal (10 seconds)")
    n_samples = 10 * 250  # 30 seconds of data
    fig = plt.figure(figsize=(10, 4))
    plt.plot(ecg_signal[:n_samples, 0], label='Lead 1', color='blue')
    plt.title("Raw ECG Signal of sample file '{}'".format(file_name))
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    st.pyplot(fig)
    # ======================Visualize the filtered ECG signal
    st.subheader("Filtered ECG Signal (10 seconds)")
    ecg_filtered = butter_bandpass_filter(ecg_signal, 0.5, 45, SAMPLING_RATE, order=3)
    ecg_filtered = denoiseECG_single(ecg_filtered, SAMPLING_RATE)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(ecg_filtered[:n_samples, 0], label='Lead 1', color='blue')
    plt.title("Filtered ECG Signal of sample file '{}'".format(file_name))
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    st.pyplot(fig)
    # ======================Visualize R-peaks detection
    st.subheader("Peaks Detection (10 seconds)")
    pan = Pan_tompkins(ecg_filtered[:, 0], SAMPLING_RATE)  # Apply to first lead
    peaks_singal = pan.fit()
    peaks_indices, _ = find_peaks(peaks_singal, distance=150)
    offset = 12
    peaks_indices = [i - offset for i in peaks_indices]    
    fig = plt.figure(figsize=(10, 4))
    valid_peaks = [i for i in peaks_indices if 0 <= i < n_samples]  
    plt.plot(valid_peaks, ecg_filtered[valid_peaks, 0], 'ro', label='R-peaks')
    plt.plot(ecg_filtered[:n_samples, 0], label='ECG Signal', color='blue')
    plt.title("Integrated ECG Signal of sample file '{}'".format(file_name))
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    st.pyplot(fig)

    # ====================== Visualize beat segments=========================
    st.subheader("Beat Segments")
    segment_len = 250
    half_len = segment_len // 2
    n_samples = 10 * 250  # 10 seconds

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot ECG signal
    ax.plot(ecg_filtered[:n_samples, 0], label="Filtered ECG", color='blue')
    
    # Plot R-peaks
    valid_peaks = [i for i in peaks_indices if 0 <= i < n_samples]
    ax.plot(valid_peaks, ecg_filtered[valid_peaks, 0], 'ro', label='R-peaks')
    # Highlight beat segments
    for i, r in enumerate(valid_peaks):
        if r - half_len < 0 or r + half_len > len(ecg_filtered):
            continue
        ax.axvspan(r - half_len, r + half_len, color='green', alpha=0.4)
    ax.set_title("Beat Segments of sample file '{}'".format(file_name))
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

def add_explain_data():
#     data = {
#     "AAMI Class": ["N", "S", "V", "F", "Q"],
#     "Description": [
#         "Normal beats",
#         "Supraventricular ectopic beats",
#         "Ventricular ectopic beats",
#         "Fusion beats",
#         "Unknown/unclassifiable beats"
#     ],
#     "MIT-BIH Beat Types": [
#         "Normal, LBBB, RBBB, Atrial escape, Nodal escape",
#         "Atrial premature, Aberrated atrial premature, Nodal premature",
#         "PVC, Ventricular escape",
#         "Fusion of ventricular and normal beat",
#         "Paced, Fusion of paced and normal, Unclassifiable"
#     ],
#     "Symbols": [
#         "N, L, R, e, j",
#         "A, a, J",
#         "V, E",
#         "F",
#         "/, f, Q"
#     ]
# }

# # Create a DataFrame
#     df = pd.DataFrame(data)
#     st.dataframe(df)
# Create the HTML table
    html_table = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #999;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
    <table>
        <tr>
            <th>AAMI Class</th>
            <th>Description</th>
            <th>MIT-BIH Beat Types</th>
            <th>Symbols</th>
        </tr>
        <tr>
            <td>N</td>
            <td>Normal beats</td>
            <td>Normal, LBBB, RBBB, Atrial escape, Nodal escape</td>
            <td>N, L, R, e, j</td>
        </tr>
        <tr>
            <td>S</td>
            <td>Supraventricular ectopic beats</td>
            <td>Atrial premature, Aberrated atrial premature, Nodal premature</td>
            <td>A, a, J</td>
        </tr>
        <tr>
            <td>V</td>
            <td>Ventricular ectopic beats</td>
            <td>PVC, Ventricular escape</td>
            <td>V, E</td>
        </tr>
        <tr>
            <td>F</td>
            <td>Fusion beats</td>
            <td>Fusion of ventricular and normal beat</td>
            <td>F</td>
        </tr>
        <tr>
            <td>Q</td>
            <td>Unknown/unclassifiable beats</td>
            <td>Paced, Fusion of paced and normal, Unclassifiable</td>
            <td>/, f, Q</td>
        </tr>
    </table>
    """

    # Display the HTML table
    st.markdown(html_table, unsafe_allow_html=True)


def main():
    st.image("imgs/example_beat.png", use_container_width=True)
    add_explain_data()
    # st.markdown(
    #     """
    #     <style>
    #     .title {
    #         text-align: center;
    #         font-size: 20px;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
     
    # === Streamlit UI ===
    uploaded_file = st.file_uploader("Upload your ECG CSV data file", type=["csv"])

    if uploaded_file is not None:
        # Save uploaded file to disk temporarily
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File uploaded successfully.")

        ecg_signal = load_ecg_from_csv("temp.csv", column_name=['Lead1', 'Lead2'])
        original_rate = 360
        target_rate = 250

        # calculate new number of samples
        duration = len(ecg_signal) / original_rate
        num_target_samples = int(duration * target_rate)

        ecg_signal = resample(ecg_signal, num_target_samples)   
        
        visualize_ecg_signal(uploaded_file.name, ecg_signal)


        # Load your pretrained model
        model = ResNetClassifier()
        model.load_state_dict(torch.load('./logs/best_model.pth'))
        model.eval()
        # Perform classification
        with st.spinner("Classifying ECG beats..."):
            results = classify_ecg_beats("temp.csv", model, lead=0)
        if results:
            # === Show bar chart ===
            st.subheader("Beat Classification Results of file '{}'".format(uploaded_file.name))
            st.write("Detected {} beats.".format(sum(results.values())))
            beat_classes = list(results.keys())
            counts = list(results.values())

            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(beat_classes, counts, color="skyblue")
            ax.tick_params(axis='x', labelsize=8)
            ax.set_xlabel("Beat Type")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of ECG Beat Classes of file '{}'".format(uploaded_file.name))

            # Add count on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', 
                        ha='center', va='bottom', fontsize=10)

            st.pyplot(fig)

            # === Check for abnormal beats ===
            n_count = results.get("Beat Normal (N)", 0)
            abnormal_count = results.get("Beat Ventricular ectopic (V)", 0) + results.get("Beat Supraventricular ectopic (S)", 0)
            # st.write("Normal (N):", n_count)
            # st.write("Supraventricular ectopic(S):", results.get("S", 0))
            # st.write("Ventricular ectopic (V):", results.get("V", 0))
            # st.write("Fusion (F):", results.get("F", 0))
            # st.write("Unknown (Q):", results.get("Q", 0))

            if n_count > 0 and abnormal_count / n_count > 0.1:
                st.warning("⚠️ This ECG sample may indicate a heart-related problem due to high abnormal beat ratio (S or V).")
            else:
                st.success("✅ Low rate of abnormal beats detected. ECG sample is likely normal.")

        else:
            st.error("No valid beats were detected in the ECG file.")
    else:
        st.info("Please upload a CSV file containing ECG data.")
    # Clean up temporary file
    if os.path.exists("temp.csv"):
        os.remove("temp.csv")
if __name__ == "__main__":
    main()