from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Error Metrics
def print_error_metrics(ypred, yval, title):
    mae = mean_absolute_error(yval, ypred)
    mse = mean_squared_error(yval, ypred)
    rmse = np.sqrt(mse)
    r2 = r2_score(yval, ypred)

    print(f"Error Metrics - {title}")
    print(f"MAE  (Mean Absolute Error):      {mae:.4f}")
    print(f"MSE  (Mean Squared Error):       {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.4f}")
    print(f"R²   (R-squared):                {r2:.4f}")


# Graficas de analisis exploratorio
def exploratory_analysis(df, value, title):
    print("Description")
    print(df.describe())
    
    print("Time Series")
    df_median = df[value].median()
    plt.figure(figsize=(13, 5))
    plt.plot(df[value], '-')
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
    
    from statsmodels.tsa.stattools import adfuller
    print('Resultados del Test de Dickey Fuller')
    dfTest = adfuller(df, autolag='AIC')
    salidaDf = pd.Series(dfTest[0:4], index=['Estadístico de prueba','p-value','# de retardos usados','# de observaciones usadas'])
    for key,value in dfTest[4].items():
            salidaDf['Critical Value (%s)'%key] = value
    print(salidaDf)
    
    if(salidaDf["p-value"]<0.05):
        print("Se rechaza Hipotesis Nula => data es estacionaria en media")
    else:
        print("NO se rechaza Hipotesis Nula => data NO es estacionaria en media")

    from statsmodels.graphics.tsaplots import plot_acf, acf

    print("ACF plot")
    plot_acf(df, lags=30)
    plt.show()
    
# Graficas de Promedio Movil
def moving_average(df, test, train, value, title, rec):
    print("Finding Best Periodsss")
    acf_values = acf(df, nlags=30)
    acf_values = acf_values[1:]
    max_idx = np.argmax(acf_values)
    print(f"Best Lag is {max_idx} with weight of {acf_values.max()}")  # Output: [0.523, 2]
    plt.figure(figsize=(13,5))
    plt.plot(train, '--', color="lightblue", label="Train Data")
    plt.plot(test, '--', color="pink", label="Validation Data")
    best_ma =  df.rolling(window=max_idx).mean()
    rec_ma = df.rolling(window=rec).mean()
    
    plt.plot(rec_ma, color='green', label=f"MA, window: {rec}")
    plt.plot(best_ma, color='blue', label=f"MA, window: {max_idx}")
    plt.title(f"Moving Average - {title}")
    plt.ylabel(value)
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Simple Exponential Smoothing
import statsmodels
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def ses_graph(train, test, value, title):
    pred_len = len(test)
    ses_model = SimpleExpSmoothing(train[value])
    ses_model = ses_model.fit()
    y_pred = ses_model.forecast(pred_len)
    plt.figure(figsize=(13,5))
    plt.title(f"Simple Exp. Smoothing - {title}")
    plt.plot(train, '--', color='pink', label='Training')
    plt.plot(ses_model.fittedvalues, label='T. Predictions', color='red')
    plt.plot(test, '--', color='lightgreen', label='Validation')
    plt.plot(y_pred, color='g', label='V. Predictions')
    plt.ylabel(value)
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print_error_metrics(y_pred, test, "SES - "+title)

# Holt Winters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def lineal_hw(train, test, value, title):
    len_ = len(test)
    es_model = ExponentialSmoothing(train[value], trend='add')
    es_model = es_model.fit()
    y_pred = es_model.forecast(len_)
    plt.figure(figsize=(13,5))
    plt.title(f"Linear Tendency Holt-Winters - {title}")
    plt.plot(train, '--', color='lightblue', label='Training')
    plt.plot(es_model.fittedvalues, label='T. Predictions')
    plt.plot(test, '--', color='lightgreen', label='Validation')
    plt.plot(y_pred, color='g', label='V. Predictions')
    plt.ylabel(value)
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print_error_metrics(y_pred, test, "Lineal HW - "+title)
    
    
def seasonal_hw(train, test, value, title, sp):
    len_ = len(test)
    es_model = ExponentialSmoothing(train[value], seasonal='mul', seasonal_periods=sp)
    es_model = es_model.fit()
    y_pred = es_model.forecast(len_)

    plt.figure(figsize=(13,5))
    plt.title(f"Sesonal Holt-Winters - {title}")
    plt.plot(train, '--', color='lightblue', label='Training')
    plt.plot(es_model.fittedvalues, label='T. Predictions')
    plt.plot(test, '--', color='lightgreen', label='Validation')
    plt.plot(y_pred, color='g', label='V. Predictions')
    plt.ylabel(value)
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print_error_metrics(y_pred, test, "Seasonal HW - "+title)

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

# Función para comparar modelos SARIMA y Prophet usando RMSE y MAE
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
    sarima_rmse = sarima_mse ** 0.5
    sarima_mae = mean_absolute_error(df_sarima[value_col], sarima_forecast)

    # --- Prophet ---
    df_prophet = df_copy[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=0)
    forecast = prophet_model.predict(future)

    prophet_mse = mean_squared_error(df_prophet['y'], forecast['yhat'])
    prophet_rmse = prophet_mse ** 0.5
    prophet_mae = mean_absolute_error(df_prophet['y'], forecast['yhat'])

    best_model = "SARIMA" if sarima_mae < prophet_mae else "Prophet"

    return {
        "Dataset": title,
        "SARIMA_RMSE": round(sarima_rmse, 2),
        "SARIMA_MAE": round(sarima_mae, 2),
        "Prophet_RMSE": round(prophet_rmse, 2),
        "Prophet_MAE": round(prophet_mae, 2),
        "Best_Model": best_model
    }
