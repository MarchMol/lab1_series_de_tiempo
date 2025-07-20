import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
female_births = pd.read_csv(
    './data/daily-total-female-births.csv', 
    parse_dates=['Date'], 
    index_col='Date',
    date_parser=dateparse
)

division = int(len(female_births)*0.7)
train_female_births = female_births[:division]
test_female_births = female_births[division:]


print(female_births.shape)
print(train_female_births.shape)
print(test_female_births.shape)

from statsmodels.tsa.seasonal import seasonal_decompose
def exploratory_analysis(df, value, title):
    print("Description")
    print(df.describe())

    print("Time Series")
    df_median = df[value].median()
    plt.figure(figsize=(13, 5))
    plt.plot(df[value],)
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
    
from scripts import analysis_graphs

date = "Date"
value = "Births"
title = "Daily Total Female Births"

exploratory_analysis(
    female_births,
    value,
    title
)

from statsmodels.tsa.stattools import adfuller
print('Resultados del Test de Dickey Fuller')
dfTest = adfuller(test_female_births, autolag='AIC')
salidaDf = pd.Series(dfTest[0:4], index=['Estadístico de prueba','p-value','# de retardos usados','# de observaciones usadas'])
for key,value in dfTest[4].items():
        salidaDf['Critical Value (%s)'%key] = value
print(salidaDf)

from statsmodels.graphics.tsaplots import plot_acf, acf

print("ACF plot")
plot_acf(train_female_births, lags=50)
plt.show()
    
print("Finding Best Period")
acf_values = acf(train_female_births, nlags=50)
acf_values = acf_values[1:]
max_idx = np.argmax(acf_values)
print(f"Best Lag is {max_idx} with weight of {acf_values.max()}")  # Output: [0.523, 2]

best_lag = max_idx



plt.plot(train_female_births)
best_ma = train_female_births.rolling(window=7).mean()
plt.plot(best_ma)
plt.plot(test_female_births)





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