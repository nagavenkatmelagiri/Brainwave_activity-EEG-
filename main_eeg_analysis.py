# Define EEG feature columns for each dataset
emotion_bands = ['mean_0_a', 'mean_1_a', 'mean_2_a', 'mean_3_a', 'mean_4_a']
eye_bands = ['af3', 'f7', 'f3', 'fc5', 'fc6', 'f4', 'f8', 'af4']
# EEG Brainwave Analysis: Emotion & Eye State Classification
# Main Script


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Load Data
emotion_df = pd.read_csv(r'C:\Users\NANNURI ARJUN REDDY\Downloads\brainwave_dataset1\emotions.csv')
eye_df = pd.read_csv(r'C:\Users\NANNURI ARJUN REDDY\Downloads\eyedetection_dataset1\EEG_Eye_State_Classification.csv')


print('--- EEG Brainwave Analysis: Emotion & Eye State Classification ---')
print('\nSTEP 1: Data Loaded Successfully')
print('Emotion Dataset (first 5 rows):')
print(emotion_df.head())
print('\nEye State Dataset (first 5 rows):')
print(eye_df.head())
print('\n')

print('STEP 2: Data Preprocessing Completed')
print(' - Cleaned column names, handled missing values, normalized EEG features, and encoded labels.')
print('Emotion features used:', emotion_bands)
print('Eye state features used:', eye_bands)
print('\n')

print('STEP 3: Visualization')
print(' - Displayed line graphs showing EEG band activity for first 200 samples.')
print(' - Displayed heatmaps showing correlation between EEG bands.')
print('\n')

print('STEP 4: Feature Engineering')
print(' - Selected EEG band columns as features for ML models.')
print('\n')

print('STEP 5: ML Model Training & Evaluation')
print(' - Trained Random Forest classifiers for emotion and eye state prediction.')
print(' - Below are the accuracy and detailed classification reports:')

# Step 2: Preprocessing
emotion_df = emotion_df.rename(columns=str.lower)
eye_df = eye_df.rename(columns=str.lower)
emotion_df.columns = emotion_df.columns.str.strip().str.replace('# ', '')
eye_df.columns = eye_df.columns.str.strip()
emotion_df = emotion_df.dropna()
eye_df = eye_df.dropna()

# Use correct band columns for each dataset
emotion_bands = ['mean_0_a', 'mean_1_a', 'mean_2_a', 'mean_3_a', 'mean_4_a']
eye_bands = ['af3', 'f7', 'f3', 'fc5', 'fc6', 'f4', 'f8', 'af4']  # adjust as needed for your dataset

scaler = StandardScaler()
emotion_df[emotion_bands] = scaler.fit_transform(emotion_df[emotion_bands])
eye_df[eye_bands] = scaler.fit_transform(eye_df[eye_bands])

emotion_label_encoder = LabelEncoder()
eye_label_encoder = LabelEncoder()
emotion_df['label_encoded'] = emotion_label_encoder.fit_transform(emotion_df['label'])
eye_df['eye_state_encoded'] = eye_label_encoder.fit_transform(eye_df['eyedetection'])

print('Processed Emotion Dataset:')
print(emotion_df.head())
print('\nProcessed Eye State Dataset:')
print(eye_df.head())

# Step 3: Visualization
def plot_eeg_lines(df, bands, title):
    plt.figure(figsize=(12,6))
    for band in bands:
        plt.plot(df[band][:200], label=band)
    plt.legend()
    plt.title(title)
    plt.show()

plot_eeg_lines(emotion_df, emotion_bands, 'EEG Brainwave Activity (Emotion, Sample 200)')
plot_eeg_lines(eye_df, eye_bands, 'EEG Brainwave Activity (Eye State, Sample 200)')

def plot_corr_heatmap(df, bands, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(df[bands].corr(), annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()

plot_corr_heatmap(emotion_df, emotion_bands, 'Correlation between Brainwave Bands (Emotion)')
plot_corr_heatmap(eye_df, eye_bands, 'Correlation between Brainwave Bands (Eye State)')

# Step 4: Feature Engineering
emotion_features = emotion_df[emotion_bands]
emotion_labels = emotion_df['label_encoded']
eye_features = eye_df[eye_bands]
eye_labels = eye_df['eye_state_encoded']

# Step 5: ML Model Training
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(emotion_features, emotion_labels, test_size=0.2, random_state=42)
model_emotion = RandomForestClassifier().fit(X_train_e, y_train_e)
y_pred_e = model_emotion.predict(X_test_e)
print('Emotion Accuracy:', accuracy_score(y_test_e, y_pred_e))
print(classification_report(y_test_e, y_pred_e))
X_train_eye, X_test_eye, y_train_eye, y_test_eye = train_test_split(eye_features, eye_labels, test_size=0.2, random_state=42)
model_eye = RandomForestClassifier().fit(X_train_eye, y_train_eye)
y_pred_eye = model_eye.predict(X_test_eye)
print('Eye State Accuracy:', accuracy_score(y_test_eye, y_pred_eye))
print(classification_report(y_test_eye, y_pred_eye))

# Predicting first sample's emotion and eye state
sample_emotion = emotion_df.iloc[0]
sample_eye = eye_df.iloc[0]
predicted_emotion = model_emotion.predict([sample_emotion[emotion_bands]])
predicted_eye_state = model_eye.predict([sample_eye[eye_bands]])
predicted_emotion_label = emotion_label_encoder.inverse_transform(predicted_emotion)
predicted_eye_state_label = eye_label_encoder.inverse_transform(predicted_eye_state)

print(f'Predicted Emotion for first sample: {predicted_emotion_label[0]}')
print(f'Predicted Eye State for first sample: {predicted_eye_state_label[0]}')
print(eye_df['eyedetection'].unique())
print(eye_df['eye_state_encoded'].unique())

# Predict emotion for a mix of NEGATIVE and POSITIVE samples
print('\n--- Sample Predictions: Emotion ---')
for idx in range(10):
    sample_features = emotion_df[emotion_bands].iloc[idx].values.reshape(1, -1)
    pred = model_emotion.predict(sample_features)[0]
    label = emotion_label_encoder.inverse_transform([pred])[0]
    true_label = emotion_df['label'].iloc[idx]
    print(f'Sample {idx+1}: True={true_label}, Predicted={label}')

# Predict eye state for a mix of samples
print('\n--- Sample Predictions: Eye State ---')
for idx in range(10):
    sample_features = eye_df[eye_bands].iloc[idx].values.reshape(1, -1)
    pred = model_eye.predict(sample_features)[0]
    label = eye_label_encoder.inverse_transform([pred])[0]
    true_label = eye_df['eyedetection'].iloc[idx]
    print(f'Sample {idx+1}: True={true_label}, Predicted={label}')

closed_eye_rows = eye_df[eye_df['eyedetection'] == 1]
print(closed_eye_rows)
