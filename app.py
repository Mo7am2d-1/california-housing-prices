"""Streamlit app: California housing price prediction.

Trains a random forest on the California housing dataset (included as
housing.csv) and predicts the median house value for district attributes
you set in the sidebar.

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


@st.cache_resource
def train_model():
    import os
    if not os.path.exists("housing.csv"):
        print("housing.csv not found, generating from scikit-learn ...")
        from sklearn.datasets import fetch_california_housing
        fetch_california_housing(as_frame=True).frame.to_csv("housing.csv", index=False)

    df = pd.read_csv("housing.csv")
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    stats = {"r2": r2_score(y_te, pred),
             "rmse": mean_squared_error(y_te, pred) ** 0.5}
    return model, df, stats


model, df, stats = train_model()

st.title("California housing price prediction")
st.write("Practice project on the California housing dataset "
         f"({len(df)} districts, 1990 census).")
st.write(f"Random forest on a held-out test split - R2 **{stats['r2']:.3f}**, "
         f"RMSE **{stats['rmse']:.2f}** (in $100,000s)")

st.sidebar.header("District attributes")
row = {
    "MedInc": st.sidebar.slider("Median income ($10k)", 0.5, 15.0, 3.0, 0.1),
    "HouseAge": st.sidebar.slider("Median house age (years)", 1, 52, 25),
    "AveRooms": st.sidebar.slider("Average rooms", 1.0, 15.0, 5.0, 0.1),
    "AveBedrms": st.sidebar.slider("Average bedrooms", 0.3, 5.0, 1.0, 0.05),
    "Population": st.sidebar.slider("Population", 3, 35000, 1500, 10),
    "AveOccup": st.sidebar.slider("Average occupancy", 0.7, 10.0, 3.0, 0.1),
    "Latitude": st.sidebar.slider("Latitude", 32.5, 42.0, 34.0, 0.01),
    "Longitude": st.sidebar.slider("Longitude", -124.5, -114.0, -118.0, 0.01),
}

pred = model.predict(pd.DataFrame([row]))[0]
st.subheader(f"Predicted median house value: ${pred * 100000:,.0f}")
st.caption("Educational model on 1990 census data - predictions reflect 1990 prices, not today's market.")
