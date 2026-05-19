# Dysarthria Detection from Speech Audio Signals

This is my Masters dissertation project at Teesside 
University. The system detects dysarthria — a 
neurological speech disorder — from audio recordings 
using deep learning.

---

## What it does

Upload a WAV audio file and the app instantly 
predicts whether the speaker has dysarthria or not.

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Dys-CNN | 86.13% | 87.59% |
| Dys-CNNBiLSTM | 90.32% | 91.35% |
| Dys-CnnAttBiLSTM | 90.57% | 91.71% |
| Dys-Ensemble | 90.66% | 91.68% |

---

## How to Run

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

---

## Dataset
TORGO Dataset available at Kaggle:
https://www.kaggle.com/datasets/iamhungundji/dysarthria-detection

---

## Built With
Python, TensorFlow, Librosa, Streamlit

---

**Developed by**

**Harish Gudi**
MSc Computer Science | Teesside University | 2026
