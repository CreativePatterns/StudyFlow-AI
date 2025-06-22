# StudyFlow-AI Overview:

This project implements a machine learning model to predict daily focus levels based on various lifestyle and study-related features such as sleep hours, mood score, study hours, and journaling habits. The goal is to provide users with actionable insights into their study patterns and potential burnout risks.

# What the Model Does:

- Predicts Focus Level: Classifies daily focus into categories (Low, Medium, High) using a Random Forest classifier trained on historical data.

- Burnout Detection: Flags potential burnout by monitoring recent low-focus streaks.

- Personalized Suggestions: Offers task recommendations based on the user's current mood score.

# Technical Approach:

- Data Preparation: Feature engineering included creating interaction terms (e.g., study hours × sleep hours) and categorical labeling of focus levels.

- Model Training: The Random Forest classifier was trained on selected features after preprocessing.

- Dashboard Integration: Streamlit was used to build an interactive web app where users can input daily data and receive real-time focus predictions and guidance.

# Challenges & Solutions:

- Environment Setup: Installing and configuring Streamlit for local development required careful troubleshooting, including resolving PATH issues on Windows.

- Feature Selection: Balancing model complexity and interpretability involved iterative testing with different feature subsets to improve prediction accuracy.

- User Interaction: Designing a user-friendly interface while maintaining prediction reliability was addressed through thoughtful UI elements and input validation.

# Developer Notes:
This project was a mix of learning and patience. It needed steady effort and the flexibility to adjust when things got tough. Taking time to reflect helped turn challenges into progress and kept the work moving forward.
