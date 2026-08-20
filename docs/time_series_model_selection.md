# Time-Series Model Selection

The telemetry is time ordered, but the primary business problem is early failure warning rather than pure sensor forecasting.

Sensor forecasting is therefore treated as a supporting degradation-analysis task. The repository begins with two intentionally simple forecasting candidates:

1. Naive persistence forecast
2. Exponential smoothing with an additive trend

They are compared with rolling-origin backtesting using MAE and RMSE. A more complex model such as ARIMA should only be introduced if the exploratory analysis demonstrates a defensible need for it.

The selected forecasting baseline should be used to answer questions such as:

- Is a sensor moving toward an OEM threshold?
- How many operating hours remain before a projected threshold crossing?
- Does forecasting add useful information beyond historical rolling features?

The forecasting component must not use future observations when constructing the prediction inputs.
