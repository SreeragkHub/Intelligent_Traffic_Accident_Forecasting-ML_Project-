# 🚦 Intelligent Traffic Accident Forecasting

A machine learning web app that predicts the likelihood of a traffic accident based on road, weather, vehicle, and driver conditions — built with **scikit-learn** and **Streamlit**.

## Overview

This project started as a Jupyter notebook exploring which factors (weather, road type, driver behavior, vehicle condition, etc.) are most predictive of traffic accidents. It compares five classification models — **KNN, Naive Bayes, SVC, Decision Tree, and Random Forest** — and wraps the best-performing pipeline in an interactive Streamlit front end so anyone can explore the data and test predictions without touching code.

## Features

- **Data Overview** — upload a CSV and instantly see cleaned data, summary stats, and interactive distribution charts for each feature (weather, road type, time of day, etc.)
- **Model Comparison** — trains all five models on the uploaded data and compares accuracy, confusion matrices, and classification reports side by side
- **Predict Accident Risk** — a form to type in exact scenario values (speed limit, driver age, road conditions, etc.) and get a live prediction with probability score from the best model
- **Batch Predict (CSV)** — upload a CSV of multiple new scenarios and get accident-risk predictions for every row at once, downloadable as a CSV

## Tech Stack

- Python
- pandas / NumPy — data cleaning and feature engineering
- scikit-learn — model training (KNN, Naive Bayes, SVM, Decision Tree, Random Forest)
- Streamlit — interactive web front end
- Plotly — interactive charts

## Dataset

The app expects a CSV with the same structure used during training, including columns such as:
`Weather`, `Road_Type`, `Time_of_Day`, `Road_Condition`, `Vehicle_Type`, `Road_Light_Condition`, `Driver_Distraction`, `Urban_or_Rural`, `Traffic_Density`, `Speed_Limit`, `Number_of_Vehicles`, `Driver_Alcohol`, `Driver_Age`, `Driver_Experience`, `Vehicle_Age`, `Seatbelt_Fastened`, `Weather_Temperature`, `Is_Weekend`, and the target column `Accident`.

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/traffic-accident-app.git
cd traffic-accident-app
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload your dataset in the sidebar, and explore.

## Project Structure

```
traffic-accident-app/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
└── README.md               # Project documentation
```

## How It Works

1. **Cleaning** — removes duplicates, caps unrealistic outliers (e.g. speed limits over 180, driver ages under 18), and fills missing values with the column mode.
2. **Encoding** — one-hot encodes categorical features (weather, road type, etc.).
3. **Training** — splits data 80/20, scales features, and trains five classifiers.
4. **Evaluation** — compares accuracy and reports per model so the best performer is used for predictions.
5. **Prediction** — new scenarios (typed in or uploaded via CSV) are encoded the same way and scored by the best model.

## Future Improvements

- Persist a trained model (e.g. with `joblib`) instead of retraining on every upload
- Add SHAP/feature-importance explanations for individual predictions
- Hyperparameter tuning via grid search inside the app

## License

Add a license of your choice (e.g. MIT) if you plan to make this public.
