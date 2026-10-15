from pathlib import Path
import os
import hmac
import sqlite3
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. APPLICATION SETUP
# --------------------------------------------------

app = FastAPI(
    title="Mental Health Score Prediction API",
    description="Predict mental health scores and store user records.",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"
DATABASE_PATH = BASE_DIR / "mental_health.db"

# Set ADMIN_API_KEY as an environment variable before starting
# the backend. Do not use a public or easily guessed password.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

if not ADMIN_API_KEY:
    raise RuntimeError(
        "ADMIN_API_KEY environment variable is missing."
    )


# --------------------------------------------------
# 2. LOAD MACHINE LEARNING MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Key"]
)


# --------------------------------------------------
# 4. DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT NOT NULL,

                age INTEGER NOT NULL,

                gender TEXT NOT NULL,

                country TEXT NOT NULL,

                academic_level TEXT NOT NULL,

                most_used_platform TEXT NOT NULL,

                purpose_of_use TEXT NOT NULL,

                avg_daily_usage_hours REAL NOT NULL,

                daily_unlocks INTEGER NOT NULL,

                study_hours REAL NOT NULL,

                physical_activity_hours REAL NOT NULL,

                sleep_hours_per_night REAL NOT NULL,

                stress_level TEXT NOT NULL,

                predicted_score REAL NOT NULL,

                created_at TEXT NOT NULL

            )
        """)


# Create database and table when the application starts.
create_database()


# --------------------------------------------------
# 5. INPUT DATA VALIDATION
# --------------------------------------------------

class StudentData(BaseModel):

    name: str = Field(min_length=1, max_length=100)

    age: int = Field(ge=10, le=100)

    gender: Literal["Male", "Female"]

    country: str = Field(min_length=1, max_length=100)

    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]

    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]

    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]

    avg_daily_usage_hours: float = Field(ge=0, le=24)

    daily_unlocks: int = Field(ge=0)

    study_hours: float = Field(ge=0, le=24)

    physical_activity_hours: float = Field(ge=0, le=24)

    sleep_hours_per_night: float = Field(ge=0, le=24)

    stress_level: Literal[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]


# --------------------------------------------------
# 6. COUNTRY GROUPING
# --------------------------------------------------

TOP_COUNTRIES = {
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
}


# --------------------------------------------------
# 7. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Mental Health Score API is running"
    }


# --------------------------------------------------
# 8. PREDICTION AND DATABASE STORAGE
# --------------------------------------------------

@app.post("/predict")
def predict_mental_health(data: StudentData):

    try:

        # Group country as used during model training.
        country_group = (
            data.country
            if data.country in TOP_COUNTRIES
            else "Other"
        )

        # Prepare model input.
        input_data = pd.DataFrame([{

            "Study_Hours": data.study_hours,

            "Age": data.age,

            "Avg_Daily_Usage_Hours":
                data.avg_daily_usage_hours,

            "Daily_Unlocks": data.daily_unlocks,

            "Physical_Activity_Hours":
                data.physical_activity_hours,

            "Sleep_Hours_Per_Night":
                data.sleep_hours_per_night,

            "Stress_Level": data.stress_level,

            "Gender": data.gender,

            "Academic_Level": data.academic_level,

            "Most_Used_Platform":
                data.most_used_platform,

            "Purpose_Of_Use":
                data.purpose_of_use,

            "Grouped_country": country_group

        }])

        # Match the exact feature order used during training.
        if hasattr(model, "feature_names_in_"):
            input_data = input_data[
                list(model.feature_names_in_)
            ]

        # Generate prediction.
        prediction = float(model.predict(input_data)[0])

        if not pd.notna(prediction):
            raise ValueError("Model returned an invalid prediction.")

        score = round(prediction, 2)

        # Save user information and prediction.
        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        with get_connection() as connection:

            cursor = connection.execute("""

                INSERT INTO users (

                    name,
                    age,
                    gender,
                    country,
                    academic_level,
                    most_used_platform,
                    purpose_of_use,
                    avg_daily_usage_hours,
                    daily_unlocks,
                    study_hours,
                    physical_activity_hours,
                    sleep_hours_per_night,
                    stress_level,
                    predicted_score,
                    created_at

                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                data.name.strip(),
                data.age,
                data.gender,
                data.country.strip(),
                data.academic_level,
                data.most_used_platform,
                data.purpose_of_use,
                data.avg_daily_usage_hours,
                data.daily_unlocks,
                data.study_hours,
                data.physical_activity_hours,
                data.sleep_hours_per_night,
                data.stress_level,
                score,
                created_at

            ))

            user_id = cursor.lastrowid

        return {

            "message": "Prediction generated and saved successfully",

            "user_id": user_id,

            "name": data.name.strip(),

            "predicted_mental_health_score": score

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction or database operation failed: {str(error)}"
        )


# --------------------------------------------------
# 9. ADMIN AUTHENTICATION
# --------------------------------------------------

def verify_admin(x_admin_key: str | None):

    if not x_admin_key or not hmac.compare_digest(
        x_admin_key,
        ADMIN_API_KEY
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials."
        )


# --------------------------------------------------
# 10. GET ALL USER RECORDS
# --------------------------------------------------

@app.get("/users")
def get_all_users(
    x_admin_key: str | None = Header(default=None)
):

    verify_admin(x_admin_key)

    with get_connection() as connection:

        records = connection.execute("""
            SELECT *
            FROM users
            ORDER BY id DESC
        """).fetchall()

    return {
        "total_users": len(records),
        "users": [dict(record) for record in records]
    }


# --------------------------------------------------
# 11. GET A PARTICULAR USER
# --------------------------------------------------

@app.get("/users/{user_id}")
def get_user(
    user_id: int,
    x_admin_key: str | None = Header(default=None)
):

    verify_admin(x_admin_key)

    with get_connection() as connection:

        record = connection.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

    if record is None:

        raise HTTPException(
            status_code=404,
            detail="User record not found."
        )

    return dict(record)