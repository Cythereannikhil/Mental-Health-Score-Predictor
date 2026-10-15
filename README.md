# Mental Health Score Prediction

A machine learning project that predicts mental health scores using students' social media usage patterns, academic activities, and lifestyle habits.

## Overview

Social media has become an important part of students' daily lives. This project explores the relationship between social media usage, study routines, sleep, physical activity, and stress levels.

The application collects user information through a web interface and uses a trained machine learning model to generate a predicted mental health score.

It also stores user submissions and their prediction results in an SQLite database.

## Features

- Predict mental health scores using a trained machine learning model.
- Collect student information and daily lifestyle details.
- Analyse inputs such as social media usage, phone unlocks, study hours, sleep, physical activity, and stress level.
- Provide an interactive interface using Streamlit.
- Serve model predictions through a FastAPI backend.
- Store user details and prediction results in an SQLite database.
- Provide an admin dashboard to view and search saved records.
- Export saved records to CSV.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- FastAPI
- SQLite
- REST API
- Joblib

## Project Structure

```text
Mental-Health-Score/
│
├── app.py
├── main.py
├── Mental_Health_Model.pkl
├── requirements.txt
├── .gitignore
├── README.md
└── ML_Project.ipynb
