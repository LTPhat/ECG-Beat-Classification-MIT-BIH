# ECG Beat Classification MIT-BIH
- Biomedical Engineering (EE-3037) Course Project.

## Run

**1/ Create env and setup project**
```sh
conda create -n [env_name] python=3.8.16
pip install -r requirements.txt
pip install -e .
```

**2/ Download dataset and data preprocessing**
```sh
python data/process/ecg_beat_classification_processing.py
```

**3/Train**
```python
python train.py
```

## Inference
- Command line:
```sh
python api.py --csv_path [csv_data_file]
```
- UI:
```sh
streamlit run app.py
```
