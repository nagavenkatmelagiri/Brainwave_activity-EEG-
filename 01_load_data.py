# Step 1: Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load EEG datasets
emotion_df = pd.read_csv(r'C:\Users\NANNURI ARJUN REDDY\Downloads\brainwave_dataset1')
eye_df = pd.read_csv(r'C:\Users\NANNURI ARJUN REDDY\Downloads\eyedetection_dataset1')

print('Emotion Dataset:')
print(emotion_df.head())
print('\nEye State Dataset:')
print(eye_df.head())
