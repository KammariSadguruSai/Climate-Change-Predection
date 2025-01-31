# Climate Change: Analyzing Trends and Developing Sustainable Solutions

## Project Overview
This project analyzes climate change trends using machine learning and statistical techniques. The objective is to identify key factors influencing climate change and develop predictive models for future climate patterns.

## Authors
- **Kammari Sadguru Sai** - 23955A6715  
- **Eerla Venkatesh** - 23955A6718  
- **Department of Data Science**

## Table of Contents
- [Requirements](#requirements)
- [Setup and Libraries](#setup-and-libraries)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Implementation](#model-implementation)
- [Model Evaluation](#model-evaluation)
- [Deployment Using Streamlit](#deployment-using-streamlit)
- [References](#references)
- [Code and Model Access](#code-and-model-access)

## Requirements
- Computer with internet access
- Python environment with the following libraries installed:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Streamlit
  - Scikit-learn
- Dataset: [Climate Change Dataset (2020-2024)](https://www.kaggle.com/datasets/atifmasih/climate-change-dataset2020-2024)

## Setup and Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
```

## Data Preprocessing
```python
df = pd.read_csv("climate_data.csv")
df.replace(["Unknown", "N/A", "NaN", ""], np.nan, inplace=True)  # Replace unknowns with NaN
df.fillna(method="ffill", inplace=True)  # Forward fill missing values
df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables to numeric
```

## Exploratory Data Analysis (EDA)
```python
sns.pairplot(df)
plt.show()
```

## Model Implementation
### Algorithms and Models Used
- **Random Forest Regressor**
  - Handles missing data effectively.
  - Reduces overfitting by averaging multiple decision trees.
  - Works well with non-linear relationships.

```python
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## Model Evaluation
```python
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
```

## Deployment Using Streamlit
- Upload a climate dataset in CSV format.
- Handle missing and non-numeric values automatically.
- Allow users to select features and target variables for prediction.
- Train a Random Forest model and evaluate it in real-time.
- Visualize actual vs. predicted values using scatter plots.
- Accept new user input and make predictions based on the trained model.

```python
input_data = {}
for feature in feature_cols:
    input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write(f"Predicted {target_col}: {prediction[0]:.2f}")
```

## References
- [Python Documentation](https://docs.python.org/3/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Code and Model Access
- **Code Repository:** [GitHub - Climate Change Analysis](https://github.com/KammariSadguruSai/Climate-Change-Predection/blob/main/climatechange_analysis.ipynb)
- **Model Deployment:** [Streamlit App - Climate Change Prediction](https://climatechangepredection.streamlit.app/)

