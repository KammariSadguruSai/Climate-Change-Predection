import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.title("🌍 Climate Change Prediction App")

# GitHub raw link for the dataset
url = 'https://raw.githubusercontent.com/KammariSadguruSai/Climate-Change-Predection/main/clean_climate_change_dataset.csv'

# Load the dataset into a DataFrame
df = pd.read_csv(url)

# Display the dataset
st.write("### Dataset Preview")
st.dataframe(df.head())

# Create a download button
csv = df.to_csv(index=False)  # Convert dataframe to CSV
st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="clean_climate_change_dataset.csv",
    mime="text/csv"
)

# Upload dataset
uploaded_file = st.file_uploader("Upload your Climate Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    # 🔹 Handle missing and non-numeric values
    df.replace(["Unknown", "N/A", "NaN", ""], np.nan, inplace=True)  # Replace unknowns with NaN
    df.fillna(method="ffill", inplace=True)  # Forward fill missing values
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables to numeric

    # Select target column
    target_col = st.selectbox("🎯 Select Target Variable", df.columns)

    # Select features
    feature_cols = st.multiselect("📌 Select Features", df.columns.drop(target_col))

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # 🔹 Convert all data to numeric (ensure float values)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Show evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("## 📈 Model Evaluation")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Plot actual vs. predicted values
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs. Predicted")
        st.pyplot(fig)

        # Predict on new data
        st.write("## 🔮 Make a Prediction")
        input_data = {}
        for feature in feature_cols:
            input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.write(f"### 🎯 Predicted {target_col}: **{prediction[0]:.2f}**")
