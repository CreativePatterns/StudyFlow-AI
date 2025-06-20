import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

study_data = pd.read_csv("study_focus_2years.csv")
# --- Use the provided dataset named 'study_data' ---
data = study_data.copy()

# --- Feature engineering and labeling ---
def label_focus(val):
    if val < 4:
        return "Low"
    elif val < 7:
        return "Medium"
    else:
        return "High"

data["focus_bin"] = data["focus_level"].apply(label_focus)
data["study_sleep_interaction"] = data["study_hours"] * data["sleep_hours"]

features = [
    "sleep_hours", "mood_score", "study_hours",
    "journaling", "study_sleep_interaction"
]

X = data[features]
y = data["focus_bin"]

# --- Train model ---
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# --- Streamlit UI ---
st.title("🎯 Daily Focus Assistant")
st.markdown("Predict your focus level and get smart suggestions based on your inputs.")

sleep = st.slider("🛌 Sleep Hours", 0.0, 10.0, 6.0)
mood = st.slider("🙂 Mood Score", 1.0, 10.0, 6.0)
study = st.slider("📚 Study Hours", 0.0, 10.0, 4.0)
journaling = st.checkbox("📝 Did you journal today?", value=True)

interaction = sleep * study

# --- Prepare input ---
today = pd.DataFrame([{
    "sleep_hours": sleep,
    "mood_score": mood,
    "study_hours": study,
    "journaling": int(journaling),
    "study_sleep_interaction": interaction
}])

# --- Prediction ---
pred = clf.predict(today)[0]
st.subheader(f"🧠 Today’s Focus Forecast: {pred}")

# --- Burnout Detection ---
recent_days = data.tail(5)
low_streak = 0
for _, row in recent_days.iterrows():
    row_input = pd.DataFrame([{
        "sleep_hours": row["sleep_hours"],
        "mood_score": row["mood_score"],
        "study_hours": row["study_hours"],
        "journaling": row["journaling"],
        "study_sleep_interaction": row["sleep_hours"] * row["study_hours"]
    }])
    if clf.predict(row_input)[0] == "Low":
        low_streak += 1

if low_streak >= 3:
    st.warning("⚠️ Burnout Alert: 3+ recent low-focus days detected. Rest is advised.")
else:
    st.success("✅ No burnout warning. You’re staying consistent!")

# --- Suggestion based on mood ---
if mood < 5:
    st.info("🎨 Mood is low — try a creative or relaxing task today.")
else:
    st.info("💼 Mood looks good — great day for focused work.")
