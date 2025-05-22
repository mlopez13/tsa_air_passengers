# Air Passengers

This is a data analysis project for the classic Box & Jenkins dataset of international airline passengers. Source: Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series Analysis, Forecasting and Control. Third Edition. Holden-Day. Series G.

* Notebook 1 (`n1_eda.ipynb`). Exploratory Data Analysis of the dataset.
* Notebook 2 (`n2_naive_model.ipynb`). Train-val-test split, decomposition of data into trend and seasonal parts, and forecast with a naive model.
* Notebook 3 (`n3_exp_smoothing.ipynb`). Exponential smoothing models.
* Notebook 4 (`n4_arima_model.ipynb`). Exploration of ARIMA models: AR, MA, ARMA, ARIMA, SARIMA.
* Notebook 5 (`n5_lstm.ipynb`). Long short-term memory models.
* Notebook 6 (`n6_all_models.ipynb`). Comparison of all models.
* Utilities (`utils.py`). Some helper functions, such as `get_data`, or `train_val_test_split`.
* Models (`models.py`). Definition of model classes.
* Modules (`modules.py`). Definition of Deep Learning modules.

WIP : explore different values for dropout to improve the LSTM model.
