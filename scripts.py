from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet

# Graficas de analisis exploratorio
def analysis_graphs(df, value, title):
    print("Description")
    print(df.describe())
    
    print("Frequency distribution")
    plt.figure(figsize=(3, 3))
    plt.title(f"Frequency Distribution - {title}")
    _, bins, _ = plt.hist(df[value], bins=30, density=False)
    mu, sigma = np.mean(df[value]), np.std(df[value])
    x = np.linspace(min(bins), max(bins), 1000) 
    bin_width = bins[1] - bins[0]
    scale = len(df) * bin_width
    y = norm.pdf(x, mu, sigma) * scale
    plt.plot(x, y, label=f'Ideal Gaussian', color='r')
    plt.xlabel(value)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    print("Time Series")
    df_median = df[value].median()
    plt.figure(figsize=(10, 5))
    plt.plot(df[value], '-o')
    plt.axhline(df_median, linestyle= '--', color='r', label=f"Mediana {df_median}")
    plt.title(f"Time Series - {title}")
    plt.xlabel("DATE")
    plt.ylabel(value)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("Seasonal Decompose")
    result = seasonal_decompose(df[value], model='additive', period=7)
    result.plot()
    
    
# Graficas de Promedio Movil
def moving_average(df, value, title, recomended):
    print("ACF plot")
    plot_acf(df, lags=50)
    plt.show()
    
    print("Finding Best Period")
    acf_values = acf(df, nlags=50)
    acf_values = acf_values[1:]
    max_idx = np.argmax(acf_values)
    print(f"Best Lag is {max_idx} with weight of {acf_values.max()}")  # Output: [0.523, 2]

    print("Moving Average Graph")
    best_ma = df.rolling(window=max_idx).mean()
    recomended_ma = df.rolling(window=recomended).mean()
    plt.figure(figsize=(10, 5))
    plt.title(f"Moving Average - {title}")
    plt.plot(df, color='lightblue')
    plt.plot(best_ma, label=f"Best Period (ACF) - {max_idx}")
    plt.plot(recomended_ma, label=f"Recommedned Period - {recomended}")
    plt.ylabel(value)
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Model SARIMA
def sarima_model(df, date_col, value_col, order=(1,1,1), seasonal_order=(1,1,1,12), title=""):
    # Asegurar que el índice tenga nombre
    if df.index.name is None:
        df.reset_index(inplace=True)

    # Verificar si la columna de fecha ya es índice
    if df.index.name != date_col:
        if date_col in df.columns:
            df.set_index(date_col, inplace=True)
        else:
            raise KeyError(f"'{date_col}' no está en las columnas ni es índice.")
    
    model = SARIMAX(df[value_col], order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    df['forecast'] = results.predict(start=0, end=len(df)-1, dynamic=False)
    
    plt.figure(figsize=(10,5))
    plt.plot(df[value_col], label='Original')
    plt.plot(df['forecast'], color='red', label='SARIMA Forecast')
    plt.title(f"SARIMA Model - {title}")
    plt.legend()
    plt.show()
    print(f"Model Summary:\n{results.summary()}")

# Función para aplicar Prophet
def apply_prophet(df, date_col, value_col, title=""):
    # Renombrar columnas como lo espera Prophet
    df_prophet = df.reset_index()[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    
    model = Prophet()
    model.fit(df_prophet)
    
    # Crear dataframe futuro
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    
    # Graficar
    fig1 = model.plot(forecast)
    plt.title(f"Prophet Forecast - {title}")
    plt.show()
    
    # Métricas
    mse = mean_squared_error(df_prophet['y'], forecast['yhat'])
    mae = mean_absolute_error(df_prophet['y'], forecast['yhat'])
    print(f"{title} - Prophet MSE: {mse:.2f}, MAE: {mae:.2f}")

# Función para comparar modelos SARIMA y Prophet
def compare_models(df, date_col, value_col, sarima_order, sarima_seasonal_order, title):
    df_copy = df.copy()

    # Si la columna de fecha no existe pero es el índice, resetear índice
    if date_col not in df_copy.columns and df_copy.index.name == date_col:
        df_copy.reset_index(inplace=True)

    # --- SARIMA ---
    df_sarima = df_copy.set_index(date_col)
    sarima_model = SARIMAX(df_sarima[value_col], order=sarima_order, seasonal_order=sarima_seasonal_order)
    sarima_results = sarima_model.fit(disp=False)
    sarima_forecast = sarima_results.predict(start=0, end=len(df_sarima)-1, dynamic=False)

    sarima_mse = mean_squared_error(df_sarima[value_col], sarima_forecast)
    sarima_mae = mean_absolute_error(df_sarima[value_col], sarima_forecast)

    # --- Prophet ---
    df_prophet = df_copy[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=0)
    forecast = prophet_model.predict(future)

    prophet_mse = mean_squared_error(df_prophet['y'], forecast['yhat'])
    prophet_mae = mean_absolute_error(df_prophet['y'], forecast['yhat'])

    best_model = "SARIMA" if sarima_mae < prophet_mae else "Prophet"

    return {
        "Dataset": title,
        "SARIMA_MSE": round(sarima_mse, 2),
        "SARIMA_MAE": round(sarima_mae, 2),
        "Prophet_MSE": round(prophet_mse, 2),
        "Prophet_MAE": round(prophet_mae, 2),
        "Best_Model": best_model
    }
