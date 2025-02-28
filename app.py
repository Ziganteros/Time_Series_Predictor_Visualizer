import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from commons import parameters, parameters_second_model


def load_data(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    return df

def forecast(df, model, params):
    
    p = {}
    for i in params.splitlines(): 
        key = i.split("=")[0]
        try: value = int(i.split("=")[1])
        except:
            try: value = float(i.split("=")[1])
            except: value = i.split("=")[1]
        p[key] = value
    print(p)

    if model == "Linear Regression": model = LinearRegression()
    elif model == "SVR": model = SVR(kernel=p['kernel'], C=p['C'], epsilon=p['epsilon'])
    elif model == "Random Forest": model = RandomForestRegressor(n_estimators=p['n_estimators'], criterion=p['criterion'])
    elif model == "Ridge": model = Ridge(alpha=p['alpha'])
    elif model == "Lasso": model = Lasso(alpha=p['alpha'])
    elif model == "Decision Tree": model = DecisionTreeRegressor(max_depth=p['max_depth'])
    else: return None, None

    df = df.head(p['numero di previsioni']+p['lookback'])
    forecast = []

    for i in range(len(df)-p['lookback']):
        df_temp = df[i:i+p['lookback']]
    
        # Reshape data for model training
        X = (np.array(range(p['lookback']))+i).reshape(-1, 1)  # X is the time step index (e.g. 0, 1, 2, ...)
        y = df_temp.values  # y is the target variable (the actual time series)

        model.fit(X, y)
        future_steps = np.array(len(df_temp) + 1).reshape(-1, 1)
        forecast.append(model.predict(future_steps)[0])
    print(f'forecast: {len(forecast)}')


    return p['numero di previsioni'], [float(x) for x in forecast]

st.title("Forecasting Tool")

# Sidebar for file upload and model selection
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Time Series (CSV)", type=["csv"])
model_choice = st.sidebar.selectbox("Select Model", [
    "Linear Regression", "SVR", "Random Forest", "Ridge", "Lasso", "Decision Tree"])
params = st.sidebar.text_area("Model Parameters", parameters[model_choice])
model_choice2 = st.sidebar.selectbox("Select a second model (optional)", [
    "---", "Linear Regression", "SVR", "Random Forest", "Ridge", "Lasso", "Decision Tree"])
params2 = st.sidebar.text_area("Second Model Parameters", parameters[model_choice2])


if st.sidebar.button("Run Forecast"):
    if uploaded_file:
        df = load_data(uploaded_file)
        window, forecast_values = forecast(df.iloc[:, 0], model_choice, params)
        window2, forecast_values2 = forecast(df.iloc[:, 0], model_choice2, params2)
        df = df.head(window)

        df['Forecast'] = forecast_values
        df['Forecast2'] = forecast_values2
        print(df)

        # Plot results
        fig, ax = plt.subplots()
        # Plot dei valori reali
        df["Close"].plot(ax=ax, label="Actual Data", color="blue")
        # Plot della previsione
        df["Forecast"].plot(ax=ax, label="Forecast", linestyle="dashed", color="red")
        if forecast_values2:
            # Plot della previsione
            df["Forecast2"].plot(ax=ax, label="Forecast (second model)", linestyle="dashed", color="black")

        ax.legend()
        st.pyplot(fig)
    else:
        st.sidebar.error("Please upload a CSV file.")
