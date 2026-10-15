import os

import pandas as pd
import requests
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Mental Health Score Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# 2. COMPACT UI STYLING
# --------------------------------------------------

st.markdown("""
<style>

.block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}

h1 {
    font-size: 28px !important;
}

h2 {
    font-size: 21px !important;
}

h3 {
    font-size: 17px !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. PAGE NAVIGATION
# --------------------------------------------------

page = st.radio(
    "Select Page",
    ["Prediction", "Admin Dashboard"],
    horizontal=True
)

st.divider()


# --------------------------------------------------
# 4. PREDICTION PAGE
# --------------------------------------------------

if page == "Prediction":

    st.title("Mental Health Score Predictor")

    st.write(
        "Enter your details and daily lifestyle information "
        "to generate a machine learning prediction."
    )

    st.caption(
        "The prediction is an estimate from the trained model, "
        "not a medical diagnosis."
    )

    with st.form("prediction_form"):

        # Personal information
        st.subheader("Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            name = st.text_input(
                "Full Name",
                placeholder="Enter your name"
            )

            age = st.number_input(
                "Age",
                min_value=10,
                max_value=100,
                value=21
            )

        with col2:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            country = st.text_input(
                "Country",
                value="India"
            )

        with col3:

            academic_level = st.selectbox(
                "Academic Level",
                [
                    "Undergraduate",
                    "Graduate",
                    "High School"
                ]
            )

        col1, col2 = st.columns(2)

        with col1:

            platform = st.selectbox(
                "Most Used Platform",
                [
                    "Instagram",
                    "YouTube",
                    "WhatsApp",
                    "Facebook",
                    "LinkedIn",
                    "Snapchat",
                    "Twitter",
                    "TikTok",
                    "LINE",
                    "KakaoTalk",
                    "VKontakte",
                    "WeChat"
                ]
            )

        with col2:

            purpose = st.selectbox(
                "Purpose of Use",
                [
                    "Networking",
                    "Education",
                    "Entertainment",
                    "News"
                ]
            )

        st.divider()

        # Daily usage
        st.subheader("Daily Social Media Usage")

        col1, col2 = st.columns(2)

        with col1:

            avg_daily_usage = st.number_input(
                "Average Daily Usage (hours)",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5,
                help="Average hours spent on social media daily."
            )

        with col2:

            daily_unlocks = st.number_input(
                "Daily Phone Unlocks",
                min_value=0,
                value=50,
                step=1
            )

        st.divider()

        # Lifestyle
        st.subheader("Lifestyle Information")

        col1, col2 = st.columns(2)

        with col1:

            study_hours = st.number_input(
                "Study Hours Per Day",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5
            )

            physical_activity = st.number_input(
                "Physical Activity (hours/day)",
                min_value=0.0,
                max_value=24.0,
                value=1.0,
                step=0.5
            )

        with col2:

            sleep_hours = st.number_input(
                "Sleep Hours Per Night",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )

            stress_level = st.selectbox(
                "Stress Level",
                [
                    "Low",
                    "Medium",
                    "High",
                    "Very High"
                ],
                index=1
            )

        submitted = st.form_submit_button(
            "Predict Mental Health Score",
            type="primary"
        )

    # Submit prediction request
    if submitted:

        if not name.strip():
            st.error("Please enter your full name.")

        elif not country.strip():
            st.error("Please enter your country.")

        else:

            payload = {

                "name": name.strip(),

                "age": int(age),

                "gender": gender,

                "country": country.strip(),

                "academic_level": academic_level,

                "most_used_platform": platform,

                "purpose_of_use": purpose,

                "avg_daily_usage_hours": float(
                    avg_daily_usage
                ),

                "daily_unlocks": int(daily_unlocks),

                "study_hours": float(study_hours),

                "physical_activity_hours": float(
                    physical_activity
                ),

                "sleep_hours_per_night": float(
                    sleep_hours
                ),

                "stress_level": stress_level

            }

            try:

                with st.spinner(
                    "Generating prediction and saving your record..."
                ):

                    response = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=30
                    )

                if response.ok:

                    result = response.json()

                    st.success(
                        "Prediction generated and record saved successfully."
                    )

                    st.subheader("Prediction Result")

                    st.metric(
                        "Predicted Mental Health Score",
                        result["predicted_mental_health_score"]
                    )

                    st.write(
                        f"Record ID: {result['user_id']}"
                    )

                    st.caption(
                        "Your information and prediction have been "
                        "saved to the application's database."
                    )

                else:

                    try:
                        error_detail = response.json().get(
                            "detail",
                            response.text
                        )
                    except ValueError:
                        error_detail = response.text

                    st.error(
                        f"Request failed: {error_detail}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "FastAPI is not running. Start the backend first."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. Please try again."
                )

            except requests.exceptions.RequestException as error:

                st.error(f"Request error: {error}")


# --------------------------------------------------
# 5. ADMIN DASHBOARD
# --------------------------------------------------

elif page == "Admin Dashboard":

    st.title("Admin Dashboard")

    st.write(
        "View saved user records and export the data."
    )

    st.caption(
        "Only authorized administrators should access this page."
    )

    # Keep the key in Streamlit session state, not in the URL.
    admin_key = st.text_input(
        "Admin API Key",
        type="password",
        help="Enter the key configured for the FastAPI backend."
    )

    if st.button("Load User Records", type="primary"):

        if not admin_key:

            st.error("Please enter the admin API key.")

        else:

            try:

                with st.spinner("Loading records..."):

                    response = requests.get(
                        f"{API_URL}/users",
                        headers={
                            "X-Admin-Key": admin_key
                        },
                        timeout=30
                    )

                if response.ok:

                    result = response.json()

                    records = result.get("users", [])

                    st.session_state["admin_key"] = admin_key
                    st.session_state["user_records"] = records

                elif response.status_code == 401:

                    st.error("Invalid admin API key.")

                else:

                    st.error(
                        f"API error: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "FastAPI is not running. Start the backend first."
                )

            except requests.exceptions.RequestException as error:

                st.error(f"Request error: {error}")

    # Display previously loaded records
    if "user_records" in st.session_state:

        records = st.session_state["user_records"]

        st.divider()

        st.subheader("Saved User Records")

        st.metric(
            "Total Saved Records",
            len(records)
        )

        if records:

            df = pd.DataFrame(records)

            # Search by user name
            search_name = st.text_input(
                "Search by Name",
                placeholder="Enter a name to filter records"
            )

            if search_name.strip():

                df = df[
                    df["name"].str.contains(
                        search_name.strip(),
                        case=False,
                        na=False
                    )
                ]

            # Display table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # CSV export
            csv_data = df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="Download Records as CSV",
                data=csv_data,
                file_name="mental_health_records.csv",
                mime="text/csv"
            )

        else:

            st.info(
                "No records found. Submit a prediction first."
            )

    st.caption(
        "Records may contain sensitive personal information. "
        "Keep the database and exported CSV files private."
    )