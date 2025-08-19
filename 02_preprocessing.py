# Step 2: Data Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Check for missing values
print('Missing values in Emotion Dataset:')
print(emotion_df.isnull().sum())
print('\nMissing values in Eye State Dataset:')
print(eye_df.isnull().sum())

# Standardize column names
emotion_df = emotion_df.rename(columns=str.lower)
eye_df = eye_df.rename(columns=str.lower)

# Handle missing values
emotion_df = emotion_df.dropna()
eye_df = eye_df.dropna()

# Select EEG band columns
bands = ['delta','theta','alpha','beta','gamma']

# Normalize EEG features
scaler = StandardScaler()
emotion_df[bands] = scaler.fit_transform(emotion_df[bands])
eye_df[bands] = scaler.fit_transform(eye_df[bands])

# Encode labels
emotion_label_encoder = LabelEncoder()
eye_label_encoder = LabelEncoder()
emotion_df['label_encoded'] = emotion_label_encoder.fit_transform(emotion_df['label'])
eye_df['eye_state_encoded'] = eye_label_encoder.fit_transform(eye_df['eye_state'])

print('Processed Emotion Dataset:')
print(emotion_df.head())
print('\nProcessed Eye State Dataset:')
print(eye_df.head())
