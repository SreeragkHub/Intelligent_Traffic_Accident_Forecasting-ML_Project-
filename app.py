"""
Intelligent Traffic Accident Forecasting - Streamlit Front End
Rebuilds the cleaning / encoding / modeling pipeline from the notebook and
wraps it in an interactive app: EDA, model comparison, and a live prediction form.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Accident Forecasting",
    page_icon="🚦",
    layout="wide",
)

CAT_COLS = [
    "Weather", "Road_Type", "Time_of_Day", "Road_Condition",
    "Vehicle_Type", "Road_Light_Condition", "Driver_Distraction", "Urban_or_Rural",
]

CAT_OPTIONS = {
    "Weather": ["Clear", "Rainy", "Foggy", "Stormy", "Snowy"],
    "Road_Type": ["City Road", "Highway", "Rural Road", "Mountain Road"],
    "Time_of_Day": ["Afternoon", "Morning", "Evening", "Night"],
    "Road_Condition": ["Dry", "Wet", "Icy", "Under Construction"],
    "Vehicle_Type": ["Bus", "Car", "Motorcycle", "Truck"],
    "Road_Light_Condition": ["Artificial Light", "Daylight", "No Light"],
    "Driver_Distraction": ["Eating/Drinking", "GPS", "Mobile Phone", "None", "Passenger"],
    "Urban_or_Rural": ["Rural", "Urban"],
}

MODEL_FACTORY = {
    "KNN": lambda: KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": lambda: GaussianNB(),
    "SVC": lambda: SVC(probability=True),
    "Decision Tree": lambda: DecisionTreeClassifier(criterion="entropy"),
    "Random Forest": lambda: RandomForestClassifier(random_state=42),
}


# --------------------------------------------------------------------------------------
# Data cleaning (mirrors the notebook's preprocessing steps)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    if "Speed_Limit" in df.columns:
        mode_speed = df["Speed_Limit"].mode()[0]
        df.loc[df["Speed_Limit"] > 180, "Speed_Limit"] = mode_speed

    if "Driver_Age" in df.columns:
        mode_age = df["Driver_Age"].mode()[0]
        df.loc[df["Driver_Age"] < 18, "Driver_Age"] = mode_age

    if "Vehicle_Age" in df.columns:
        mode_vage = df["Vehicle_Age"].mode()[0]
        df.loc[df["Vehicle_Age"] > 20, "Vehicle_Age"] = mode_vage

    if "Accident_Severity" in df.columns:
        df = df.drop(columns=["Accident_Severity"])

    if "Driver_Distraction" in df.columns:
        df["Driver_Distraction"] = df["Driver_Distraction"].fillna("None")

    fill_mode_cols = [
        "Weather", "Road_Type", "Time_of_Day", "Road_Condition", "Vehicle_Type",
        "Road_Light_Condition", "Urban_or_Rural", "Traffic_Density", "Speed_Limit",
        "Number_of_Vehicles", "Driver_Alcohol", "Driver_Age", "Driver_Experience",
        "Accident", "Seatbelt_Fastened", "Is_Weekend",
    ]
    for col in fill_mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


@st.cache_data(show_spinner=False)
def encode_data(df: pd.DataFrame):
    present_cat = [c for c in CAT_COLS if c in df.columns]
    dummies = pd.get_dummies(df[present_cat], drop_first=True, dtype=int)
    df_final = pd.concat([df.drop(columns=present_cat), dummies], axis=1)
    return df_final


@st.cache_resource(show_spinner=False)
def train_all_models(df_final: pd.DataFrame):
    x = df_final.drop(columns=["Accident"])
    y = df_final["Accident"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    results = {}
    for name, factory in MODEL_FACTORY.items():
        model = factory()
        model.fit(x_train_s, y_train)
        preds = model.predict(x_test_s)
        acc = accuracy_score(y_test, preds)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "confusion_matrix": confusion_matrix(y_test, preds),
            "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        }

    return {
        "results": results,
        "scaler": scaler,
        "feature_columns": list(x.columns),
        "x_test_s": x_test_s,
        "y_test": y_test,
    }


def build_input_row(inputs: dict, feature_columns: list) -> pd.DataFrame:
    row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    numeric_fields = [
        "Traffic_Density", "Speed_Limit", "Number_of_Vehicles", "Driver_Alcohol",
        "Driver_Age", "Driver_Experience", "Vehicle_Age", "Seatbelt_Fastened",
        "Weather_Temperature", "Is_Weekend",
    ]
    for field in numeric_fields:
        if field in row.columns:
            row.at[0, field] = inputs[field]

    for prefix in CAT_COLS:
        val = inputs.get(prefix)
        col = f"{prefix}_{val}"
        if col in row.columns:
            row.at[0, col] = 1
        # if col isn't found, it's the dropped baseline category -> leave zeros

    return row[feature_columns]


# --------------------------------------------------------------------------------------
# Sidebar - data source
# --------------------------------------------------------------------------------------
st.sidebar.title("🚦 Traffic Accident Forecasting")
st.sidebar.markdown("Upload the training CSV used in the notebook (e.g. `Traffic_Accident.csv`).")

uploaded_file = st.sidebar.file_uploader("Training data (CSV)", type=["csv"])

if uploaded_file is None:
    st.title("🚦 Intelligent Traffic Accident Forecasting")
    st.info(
        "👈 Upload your **Traffic_Accident.csv** file in the sidebar to get started. "
        "This app cleans the data, trains the same models compared in your notebook "
        "(KNN, Naive Bayes, SVC, Decision Tree, Random Forest), and lets you predict "
        "accident risk for a new scenario."
    )
    st.stop()

raw_df = pd.read_csv(uploaded_file)
clean_df = clean_data(raw_df)
final_df = encode_data(clean_df)

if "Accident" not in final_df.columns:
    st.error("The uploaded file doesn't contain an 'Accident' column, which is required as the prediction target.")
    st.stop()

bundle = train_all_models(final_df)
results = bundle["results"]
feature_columns = bundle["feature_columns"]
scaler = bundle["scaler"]

best_model_name = max(results, key=lambda k: results[k]["accuracy"])

st.sidebar.success(f"Data loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
st.sidebar.metric("Best model", best_model_name, f"{results[best_model_name]['accuracy']*100:.2f}% accuracy")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
st.title("🚦 Intelligent Traffic Accident Forecasting")

tab_overview, tab_models, tab_predict = st.tabs(
    ["📊 Data Overview", "🧠 Model Comparison", "🔮 Predict Accident Risk"]
)

# ---- Overview tab ----
with tab_overview:
    st.subheader("Dataset snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (after de-dup)", clean_df.shape[0])
    c2.metric("Raw columns", raw_df.shape[1])
    c3.metric("Accident rate", f"{clean_df['Accident'].mean()*100:.1f}%")

    st.dataframe(clean_df.head(20), use_container_width=True)

    st.subheader("Categorical distributions")
    dist_cols = [c for c in CAT_COLS if c in clean_df.columns]
    picked = st.selectbox("Choose a feature to explore", dist_cols)
    counts = clean_df[picked].value_counts().reset_index()
    counts.columns = [picked, "count"]
    fig = px.bar(counts, x=picked, y="count", color=picked, title=f"{picked} distribution")
    st.plotly_chart(fig, use_container_width=True)

# ---- Model comparison tab ----
with tab_models:
    st.subheader("Accuracy comparison across models")
    acc_df = pd.DataFrame(
        {"Model": list(results.keys()), "Accuracy (%)": [results[m]["accuracy"] * 100 for m in results]}
    ).sort_values("Accuracy (%)", ascending=False)

    fig2 = px.bar(
        acc_df, x="Model", y="Accuracy (%)", color="Model", text_auto=".2f",
        title="Model accuracy comparison (held-out 20% test split)",
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Inspect a specific model")
    chosen = st.selectbox("Model", list(results.keys()), index=list(results.keys()).index(best_model_name))
    r = results[chosen]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", f"{r['accuracy']*100:.2f}%")
        cm = r["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.write("Confusion matrix")
        st.dataframe(cm_df, use_container_width=True)
    with col_b:
        report_df = pd.DataFrame(r["report"]).transpose()
        st.write("Classification report")
        st.dataframe(report_df.round(3), use_container_width=True)

# ---- Prediction tab ----
with tab_predict:
    st.subheader("Enter scenario details")
    st.caption(f"Predictions use the **{best_model_name}** model (highest test accuracy).")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            weather = st.selectbox("Weather", CAT_OPTIONS["Weather"])
            road_type = st.selectbox("Road Type", CAT_OPTIONS["Road_Type"])
            time_of_day = st.selectbox("Time of Day", CAT_OPTIONS["Time_of_Day"])
            road_condition = st.selectbox("Road Condition", CAT_OPTIONS["Road_Condition"])

        with col2:
            vehicle_type = st.selectbox("Vehicle Type", CAT_OPTIONS["Vehicle_Type"])
            road_light = st.selectbox("Road Light Condition", CAT_OPTIONS["Road_Light_Condition"])
            distraction = st.selectbox("Driver Distraction", CAT_OPTIONS["Driver_Distraction"])
            urban_rural = st.selectbox("Area", CAT_OPTIONS["Urban_or_Rural"])

        with col3:
            traffic_density = st.selectbox("Traffic Density", [0, 1, 2], index=1)
            speed_limit = st.slider("Speed Limit (km/h)", 20, 180, 60)
            num_vehicles = st.slider("Number of Vehicles Involved", 1, 15, 2)
            driver_age = st.slider("Driver Age", 18, 80, 35)

        col4, col5, col6 = st.columns(3)
        with col4:
            driver_experience = st.slider("Driver Experience (years)", 0, 60, 10)
            vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
        with col5:
            weather_temp = st.slider("Weather Temperature (°C)", -10.0, 35.0, 20.0)
            driver_alcohol = st.radio("Driver Alcohol Involvement", ["No", "Yes"], horizontal=True)
        with col6:
            seatbelt = st.radio("Seatbelt Fastened", ["Yes", "No"], horizontal=True)
            is_weekend = st.radio("Is Weekend", ["No", "Yes"], horizontal=True)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        inputs = {
            "Weather": weather,
            "Road_Type": road_type,
            "Time_of_Day": time_of_day,
            "Road_Condition": road_condition,
            "Vehicle_Type": vehicle_type,
            "Road_Light_Condition": road_light,
            "Driver_Distraction": distraction,
            "Urban_or_Rural": urban_rural,
            "Traffic_Density": traffic_density,
            "Speed_Limit": speed_limit,
            "Number_of_Vehicles": num_vehicles,
            "Driver_Alcohol": 1 if driver_alcohol == "Yes" else 0,
            "Driver_Age": driver_age,
            "Driver_Experience": driver_experience,
            "Vehicle_Age": vehicle_age,
            "Seatbelt_Fastened": 1 if seatbelt == "Yes" else 0,
            "Weather_Temperature": weather_temp,
            "Is_Weekend": 1 if is_weekend == "Yes" else 0,
        }

        row = build_input_row(inputs, feature_columns)
        row_scaled = scaler.transform(row)

        model = results[best_model_name]["model"]
        pred = model.predict(row_scaled)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row_scaled)[0][1]

        st.divider()
        if pred == 1:
            st.error("⚠️ High risk: an accident is predicted for this scenario.")
        else:
            st.success("✅ Low risk: no accident is predicted for this scenario.")

        if proba is not None:
            st.metric("Estimated accident probability", f"{proba*100:.1f}%")
            st.progress(min(max(proba, 0.0), 1.0))
