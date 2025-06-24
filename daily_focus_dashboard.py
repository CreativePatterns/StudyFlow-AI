import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# --- Load data ---
study_data = pd.read_csv("study_focus.csv")  # Replace with your dataset path

# --- Feature engineering ---
def label_focus(val):
    return "Low" if val < 4.5 else "High"

# --- Loose balancing function ---
def balance_focus_classes_loose(df, target_col='focus_bin', ratio=1.5):
    counts = df[target_col].value_counts()
    min_count = counts.min()
    balanced_parts = []
    for cls in counts.index:
        n_samples = int(min_count * ratio) if cls == counts.idxmax() else min_count
        cls_samples = df[df[target_col] == cls]
        if len(cls_samples) > n_samples:
            cls_samples = cls_samples.sample(n=n_samples, random_state=42)
        balanced_parts.append(cls_samples)
    balanced_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

study_data["focus_bin"] = study_data["focus_level"].apply(label_focus)
study_data["mood_x_sleep"] = study_data["mood_score"] * study_data["sleep_hours"]
study_data["study_efficiency"] = study_data["study_hours"] / (study_data["sleep_hours"] + 1)


balanced_study_data = balance_focus_classes_loose(study_data, target_col='focus_bin', ratio=2)

# --- Feature selection ---
features = [
    "study_hours", "mood_score", "sleep_hours",
    "journaling", "social_interaction", "mood_x_sleep", "study_efficiency"
]
X = balanced_study_data[features]
y = balanced_study_data["focus_bin"]

# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train model ---
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, y)

# --- Streamlit App ---
st.title("🎯 Daily Focus Predictor")
st.markdown("Enter your data to forecast your focus state.")

sleep = st.slider("🛌 Sleep Hours", 0.0, 10.0, 6.0)
mood = st.slider("🙂 Mood Score", 1.0, 10.0, 6.0)
study = st.slider("📚 Study Hours", 0.0, 10.0, 4.0)
journaling = st.checkbox("📝 Did you journal today?", value=True)
social = st.slider("🗣️ Social Interaction (0-10)", 0.0, 10.0, 5.0)

interaction1 = mood * sleep
interaction2 = study / (sleep + 1)

input_df = pd.DataFrame([{
    "study_hours": study,
    "mood_score": mood,
    "sleep_hours": sleep,
    "journaling": int(journaling),
    "social_interaction": social,
    "mood_x_sleep": interaction1,
    "study_efficiency": interaction2
}])

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

st.subheader(f"🧠 Predicted Focus Level: {prediction}")

# --- Optional: Show confusion matrix ---
if st.checkbox("🔍 Show confusion matrix"):
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred, labels=["Low", "Medium", "High"])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

# --- Optional metrics ---
scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
st.write("Cross-validated accuracy scores:", scores)
st.write("Mean accuracy:", scores.mean())