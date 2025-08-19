# Step 3: Visualization
import matplotlib.pyplot as plt
import seaborn as sns

bands = ['delta','theta','alpha','beta','gamma']

# Line plot for emotion dataset
def plot_eeg_lines(df, title):
    plt.figure(figsize=(12,6))
    for band in bands:
        plt.plot(df[band][:200], label=band)
    plt.legend()
    plt.title(title)
    plt.show()

plot_eeg_lines(emotion_df, 'EEG Brainwave Activity (Emotion, Sample 200)')
plot_eeg_lines(eye_df, 'EEG Brainwave Activity (Eye State, Sample 200)')

# Correlation heatmap
def plot_corr_heatmap(df, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(df[bands].corr(), annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()

plot_corr_heatmap(emotion_df, 'Correlation between Brainwave Bands (Emotion)')
plot_corr_heatmap(eye_df, 'Correlation between Brainwave Bands (Eye State)')
