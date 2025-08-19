# Step 5: ML Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Emotion classification
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(emotion_features, emotion_labels, test_size=0.2, random_state=42)
model_emotion = RandomForestClassifier().fit(X_train_e, y_train_e)
y_pred_e = model_emotion.predict(X_test_e)
print('Emotion Accuracy:', accuracy_score(y_test_e, y_pred_e))
print(classification_report(y_test_e, y_pred_e))

# Eye state classification
X_train_eye, X_test_eye, y_train_eye, y_test_eye = train_test_split(eye_features, eye_labels, test_size=0.2, random_state=42)
model_eye = RandomForestClassifier().fit(X_train_eye, y_train_eye)
y_pred_eye = model_eye.predict(X_test_eye)
print('Eye State Accuracy:', accuracy_score(y_test_eye, y_pred_eye))
print(classification_report(y_test_eye, y_pred_eye))
