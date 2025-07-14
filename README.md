# 📚 StudyFlow-AI: Your Focus, Decoded.

Welcome to **StudyFlow-AI**, a machine learning–powered companion that helps you understand how your daily habits — like sleep, mood, study time, and journaling — shape your focus.

It’s more than just a predictor. It’s a quiet observer, gently alerting you to potential burnout and nudging you toward balance. Built with **Random Forests**, **Streamlit**, and a lot of intention.

---

## What It Does

- 🧠 **Predicts Daily Focus**  
  Classifies your focus as **Low**, **Medium**, or **High** using your self-tracked inputs.

- 🚨 **Detects Burnout Risk**  
  Flags streaks of low focus and encourages you to pause before the crash.

- 🗺️ **Offers Personalized Suggestions**  
  Uses your mood score to recommend tasks aligned with your current emotional energy.

---

## Why This Project?

This wasn’t just about algorithms — it was about reclaiming mental clarity.  
The process began as a way to **understand personal energy rhythms** and transformed into a tool to support students and self-learners through the ebb and flow of motivation.

> _"When the calendar fails you, maybe the data can guide you."_

---

## Technologies Used

| Feature                     | Stack                         |
|----------------------------|-------------------------------|
| Model                      | `RandomForestClassifier` (sklearn) |
| Data Interaction           | `pandas`, `numpy`             |
| Visualization & UI         | `streamlit`, `matplotlib`     |
| Burnout Tracking           | Custom streak detection logic |
| Deployment   | Streamlit Cloud / GitHub Pages |

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/studyflow-ai.git
   cd studyflow-ai
