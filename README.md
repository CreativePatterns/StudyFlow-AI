# 🎯 StudyFlow AI – Daily Focus Predictor

This is a lightweight Streamlit-based web app that predicts your daily **focus level** based on lifestyle factors like mood, sleep, study habits, and journaling. It's built for students, self-learners, and anyone looking to track their productivity mindset and catch burnout before it sneaks up.

### 🌐 Live App:
Try it here: https://studyflow-ai.streamlit.app/

---

## 🔍 What It Does

- **Predicts Focus**  
  Using a trained logistic regression model, the app predicts whether your focus is **High** or **Low**.

- **Considers Key Lifestyle Signals**  
  Including mood score, sleep hours, journaling, and social interaction.

- **Provides Feedback**  
  Shows predictions instantly, with optional confusion matrix + accuracy metrics for nerds like us 😎

---

## 🧪 Tech Stack

- **Frontend/UI**: Streamlit  
- **Backend Model**: Scikit-learn's Logistic Regression  
- **Data Engineering**: Custom interaction terms + loose balancing for class distribution  
- **Deployment**: [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 🧠 Feature Engineering

We use thoughtful feature combinations to capture daily dynamics:
- `mood_x_sleep` = mood score × sleep hours
- `study_efficiency` = study hours / (sleep hours + 1)

We also **loosely balance** the dataset with a custom function that keeps as much data as possible without letting one class dominate.

---

## ⚙️ How It Works (Simplified Flow)

1. User inputs daily metrics into sliders/checks.
2. App calculates extra interaction terms.
3. Logistic regression model makes a prediction.
4. App shows focus level + optional confusion matrix & accuracy.
   
---

## 📊 Dataset

The app uses a local CSV file `study_focus.csv` for training.  
It’s a synthetic but realistic dataset based on:
- sleep hours
- mood score
- study time
- journaling (yes/no)
- social interaction levels
- and focus score labeled as High/Low

---

## 🛠️ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/studyflow-ai.git
   cd studyflow-ai
