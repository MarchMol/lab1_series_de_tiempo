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
    es_model = ExponentialSmoothing(train[value], trend='mul')
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
    # Dividir datos 70% train / 30% test
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    # Entrenar solo con train
    model = SARIMAX(train[value_col], order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    # Predecir sobre el conjunto test
    forecast = results.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    # Calcular métricas sobre test
    rmse = np.sqrt(mean_squared_error(test[value_col], forecast))
    mae = mean_absolute_error(test[value_col], forecast)
    # Graficar resultados
    plt.figure(figsize=(15, 6))
    plt.plot(train.index, train[value_col], label='Train', color='blue')
    plt.plot(test.index, test[value_col], label='Test', color='green')
    plt.plot(forecast.index, forecast, label=f"SARIMA{order}x{seasonal_order} (RMSE: {rmse:.2f}, MAE: {mae:.2f})", color='red', linestyle='--')
    plt.axvline(test.index[0], color='black', linestyle=':', linewidth=2, label='División Train/Test')

    plt.title(f"SARIMA - {title}")
    plt.xlabel("Fecha")
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Modelo SARIMA {order}x{seasonal_order} - {title}")
    print(f"{title} - SARIMA RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Función para aplicar Prophet
def apply_prophet(df, date_col, value_col, title=""):
    # Renombrar columnas como lo espera Prophet
    df_prophet = df.reset_index()[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    
    # Detectar frecuencia automáticamente
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
    freq = pd.infer_freq(df_prophet["ds"])
    if freq is None:
        # Si no detecta frecuencia, asumimos diaria
        freq = 'D'

    # División train/test
    split_idx = int(len(df_prophet) * 0.7)
    train = df_prophet.iloc[:split_idx]
    test = df_prophet.iloc[split_idx:]

    # Entrenar Prophet 
    model = Prophet()
    model.fit(train)

    # Forecast hasta el final de la serie (train + test)
    future = model.make_future_dataframe(periods=len(test), freq=freq)
    forecast = model.predict(future)
    # Gráfica completa de Prophet
    fig1 = model.plot(forecast)
    plt.title(f"Prophet Forecast - {title}")
    plt.show()

    # Gráfica de componentes sobre lo aprendido con el train
    model.plot_components(forecast)
    plt.suptitle(f"Componentes de Prophet (entrenamiento) - {title}", fontsize=16)
    plt.tight_layout()
    plt.show()
    # Calcular métricas sobre test
    forecast_test = forecast.iloc[split_idx:]
    rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
    mae = mean_absolute_error(test['y'], forecast_test['yhat'])
    # Gráfica
    plt.figure(figsize=(15, 6))
    plt.plot(train['ds'], train['y'], label='Train', color='blue')
    plt.plot(test['ds'], test['y'], label='Test', color='green')
    plt.plot(forecast_test['ds'], forecast_test['yhat'], label=f"Prophet (RMSE: {rmse:.2f}, MAE: {mae:.2f})", color='red', linestyle='--')
    plt.fill_between(forecast_test['ds'], forecast_test['yhat_lower'], forecast_test['yhat_upper'], color='red', alpha=0.2, label='Intervalo de Confianza')
    plt.axvline(test['ds'].iloc[0], color='black', linestyle=':', linewidth=2, label='División Train/Test')
    plt.title(f"Prophet - {title}")
    plt.xlabel("Fecha")
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"{title} - Prophet RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Función para comparar modelos usando RMSE y MAE
def plot_model_comparison(metrics_dict, title="Comparación de Modelos"):
     # Preparar los datos y calcular el promedio RMSE+MAE para ordenar
    sorted_models = sorted(
        metrics_dict.items(),
        key=lambda x: (x[1]['RMSE'] + x[1]['MAE']) / 2,
        reverse=True
    )
    model_names = [model for model, _ in sorted_models]
    rmse_values = [metrics['RMSE'] for _, metrics in sorted_models]
    mae_values = [metrics['MAE'] for _, metrics in sorted_models]

    # Crear la figura
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9A", "#D0BBFF", "#FFE38E"]
    # Posiciones de las métricas en el eje X
    metric_labels = ["RMSE", "MAE"]
    x = range(len(metric_labels))
    width = 0.15  # ancho de las barras

    # Dibujar barras para cada modelo
    for idx, model in enumerate(model_names):
        values = [rmse_values[idx], mae_values[idx]]
        ax.bar(
            [pos + width * idx for pos in x],  # posición de las barras
            values,
            width,
            label=model,
            color=colors[idx % len(colors)]
        )

    # Configurar etiquetas y título
    ax.set_xlabel('Métricas')
    ax.set_ylabel('Valor del Error')
    ax.set_title(title)
    ax.set_xticks([pos + width*(len(model_names)-1)/2 for pos in x])  # centrar ticks
    ax.set_xticklabels(metric_labels)
    ax.legend(title="Modelos")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
