# Step 4: Feature Engineering
bands = ['delta','theta','alpha','beta','gamma']

# Features and labels for emotion
emotion_features = emotion_df[bands]
emotion_labels = emotion_df['label_encoded']

# Features and labels for eye state
eye_features = eye_df[bands]
eye_labels = eye_df['eye_state_encoded']

# Optionally, extract more statistical features
# Example: mean, variance, skewness, kurtosis
# You can use scipy.stats for advanced features
