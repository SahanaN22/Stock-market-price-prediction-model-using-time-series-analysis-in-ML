import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output
st.set_page_config(layout="wide")
import plotly.graph_objects as go
from datetime import datetime

st.markdown(
    """
        <style>
                .stAppHeader {
                    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                    visibility: visible;  /* Ensure the header is visible */
                }

               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

# --- Helper Functions ---

def fetch_stock_data():
    """Fetches historical stock data using yfinance."""
    try:
        # Use st.session_state to cache the data
        # st.session_state.stock_data = ''
        # if 'stock_data' not in st.session_state:
        data = yf.download(st.session_state.stock_selected, start='2012-01-01', end=datetime.now())
        st.session_state.stock_data = data.droplevel(1, axis=1)
        print(st.session_state.stock_data)
            # st.session_state.ticker = ticker
            # st.session_state.period = period
        # else:
        #     data = st.session_state.stock_data
        # return data
    except Exception as e:
        st.error(f"Error fetching data for {st.session_state.stock_selected}: {e}")
        return None

def split_data(data, train_size_percent=0.8):
    """Splits data into training and testing sets."""
    train_size = int(len(data) * train_size_percent)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def prepare_lstm_data(train_data, test_data, n_steps, scaler=None):
    """Prepares data for LSTM model."""
    train_close = train_data['Close'].values.reshape(-1, 1)
    test_close = test_data['Close'].values.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train_close)
    else:
        scaled_train = scaler.transform(train_close)
    scaled_test = scaler.transform(test_close)

    X_train, y_train = [], []
    for i in range(n_steps, len(scaled_train)):
        X_train.append(scaled_train[i - n_steps:i, 0])
        y_train.append(scaled_train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test, y_test = [], []
    for i in range(n_steps, len(scaled_test)):
        X_test.append(scaled_test[i - n_steps:i, 0])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test, scaler

def build_lstm_model(n_steps, n_features=1):
    """Builds the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run_lstm(train_data, test_data, n_steps=60, epochs=1, batch_size=32):
    """Runs the LSTM model."""
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(train_data.copy(), test_data.copy(), n_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_lstm_model(n_steps)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    predicted_scaled = model.predict(X_test, verbose=0)
    predicted_prices = scaler.inverse_transform(predicted_scaled)
    predictions_index = test_data.index[n_steps:]
    predictions_df = pd.DataFrame({'ds': predictions_index, 'yhat': predicted_prices.flatten()})
    return predictions_df, scaler, model

def run_arima(train_data, test_data, order=(0, 1, 0)):
    """Runs the ARIMA model."""
    # try:
    #     model = ARIMA(train_data['Close'], order=order)
    #     model_fit = model.fit()
    #     predictions = model_fit.forecast(steps=len(test_data))
    #     predictions_df = pd.DataFrame({'ds': test_data.index, 'yhat': predictions})
    #     return predictions_df, model_fit
    # except Exception as e:
    #     st.error(f"Error running ARIMA: {e}")
    #     return None, None
    try:
        history = [x for x in train_data['Close'].values]
        predictions = []
        for t in range(len(test_data)):
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_data['Close'].values[t]
            history.append(obs)
            # train_data = pd.concat([train_data, test_data.iloc[[t]]]) #update train data
        predictions_df = pd.DataFrame({'ds': test_data.index, 'yhat': predictions})
        return predictions_df, model_fit
    except Exception as e:
        print(f"Error running ARIMA: {e}")
        return None, None
    


def run_holt_winters(train_data, test_data, seasonal='add', seasonal_periods=7):
    """Runs the Holt-Winters model."""
    try:
        model = ExponentialSmoothing(train_data['Close'], seasonal=seasonal, seasonal_periods=seasonal_periods)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test_data))
        predictions_df = pd.DataFrame({'ds': test_data.index, 'yhat': predictions})
        return predictions_df, model_fit
    except Exception as e:
        st.error(f"Error running Holt-Winters: {e}")
        return None, None

def run_prophet(train_data, test_data):
    """Runs the Prophet model."""
    try:
        train_prophet = train_data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        test_prophet = test_data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        model = Prophet(daily_seasonality=True)
        model.fit(train_prophet)
        forecast = model.predict(test_prophet[['ds']])
        predictions_df = forecast[['ds', 'yhat']].rename(columns={'yhat': 'yhat'})
        return predictions_df, model
    except Exception as e:
        st.error(f"Error running Prophet: {e}")
        return None, None

def evaluate_predictions(test_data, predictions_df, model_name):
    """Evaluates the predictions."""
    try:
        merged_df = pd.merge(test_data.reset_index(), predictions_df, left_on='Date', right_on='ds', how='inner')
        if merged_df.empty:
            st.error(f"No matching dates between test data and predictions for {model_name}.")
            return {}
        y_true = merged_df['Close']
        y_pred = merged_df['yhat']
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'MAE': mae, 'R2': r2}
    except Exception as e:
        st.error(f"Error evaluating predictions for {model_name}: {e}")
        return {}

def plot_predictions(train_data, test_data, predictions_df, model_name, ticker):
    """Plots the predictions along with training and test data."""
    fig = plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data['Close'], label='Train Data')
    plt.plot(test_data.index, test_data['Close'], label='Test Data')
    plt.plot(predictions_df['ds'], predictions_df['yhat'], label=f'{model_name} Predictions', color='purple')
    plt.title(f"{ticker} Close Price Prediction using {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    st.pyplot(fig)  # Use st.pyplot() to display Matplotlib figures

def plot_plotly_predictions(train_data, test_data, predictions_df, model_name, ticker):
    """Plots predictions using Plotly for better interactivity."""
    fig = px.line()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], name='Train Data'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], name='Test Data'))
    fig.add_trace(go.Scatter(x=predictions_df['ds'], y=predictions_df['yhat'], name=f'{model_name} Predictions'))
    fig.update_layout(title=f"{ticker} Close Price Prediction using {model_name}",
                      xaxis_title="Date",
                      yaxis_title="Close Price")
    st.plotly_chart(fig)  # Use st.plotly_chart

# --- Main Application ---

# def main():
st.title("Stock Price Prediction App")

# --- Layout with Two Columns ---
left_column, mid_column, right_column = st.columns([0.5, 0.02, 0.5])

# --- Left Column: EDA and Data Description ---
with left_column:
    st.header("Exploratory Data Analysis")
    # --- 1. Data Selection ---
    stock_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
    # fetch_stock_data()
    selected_stock = st.selectbox("Select a Stock", stock_list, index=None, key='stock_selected', on_change=fetch_stock_data)  # AAPL is default
    # fetch_stock_data()
    # print(stock_data)
    # stock_data = stock_data.droplevel(1, axis=1)
    # st.session_state.stock_data = ''
    
    # if st.session_state.stock_data is not None:
    #     st.session_state.stock_data = st.session_state.stock_data.droplevel(1, axis=1)

    if 'stock_data' not in st.session_state:
        st.stop()  # Stop if data fetching fails

    st.subheader("Data Description")
    st.write(st.session_state.stock_data.describe())

    st.subheader("Stock Data")
    st.dataframe(st.session_state.stock_data)

    st.subheader("Close Price Chart")
    fig = px.line(st.session_state.stock_data, x=st.session_state.stock_data.index, y='Close', title=f'{selected_stock} Close Price Over Time')
    st.plotly_chart(fig)

    st.subheader("Seasonal Decomposition (Close Price)")
    try:
        decomposition = seasonal_decompose(st.session_state.stock_data['Close'], model='additive', period=365)
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        decomposition.observed.plot(ax=axes[0], label='Observed')
        decomposition.trend.plot(ax=axes[1], label='Trend')
        decomposition.seasonal.plot(ax=axes[2], label='Seasonal')
        decomposition.resid.plot(ax=axes[3], label='Residual')
        for ax in axes:
            ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Seasonal decomposition failed: {e}.  Ensure your time series has enough data points.")

# --- Right Column: Model Training and Forecasting ---
with right_column:
    st.header("Model Training and Forecasting")
    # --- 3. Model Selection and Training ---
    model_list = ['LSTM', 'ARIMA', 'Holt-Winters', 'Prophet']
    selected_model = st.selectbox("Select a Model", model_list)
    train_size_percent = st.slider("Train Size (%)", 0.1, 0.99, 0.8)
    train_data, test_data = split_data(st.session_state.stock_data, train_size_percent)

    if st.button("Train Model"):  # Train model on button click
        with st.spinner(f"Training {selected_model} model..."):
            if selected_model == 'LSTM':
                lstm_predictions, lstm_scaler, lstm_model = run_lstm(train_data, test_data)
                st.session_state.lstm_predictions = lstm_predictions #store predictions
                if st.session_state.lstm_predictions is not None:
                    plot_plotly_predictions(train_data, test_data, st.session_state.lstm_predictions, 'LSTM', selected_stock)
                    lstm_metrics = evaluate_predictions(test_data, st.session_state.lstm_predictions, 'LSTM')
                    st.session_state.lstm_metrics = lstm_metrics
                    st.subheader("LSTM Model Evaluation")
                    st.write(pd.DataFrame([st.session_state.lstm_metrics]))
                    st.session_state.trained_model = lstm_model  # save model
                    st.session_state.scaler = lstm_scaler
                    st.session_state.n_steps = 60 # Store n_steps
                    
                else:
                    st.error("LSTM model training failed.")

            elif selected_model == 'ARIMA':
                arima_predictions, arima_model = run_arima(train_data, test_data)
                st.session_state.arima_predictions = arima_predictions
                if st.session_state.arima_predictions is not None:
                    plot_plotly_predictions(train_data, test_data, st.session_state.arima_predictions, 'ARIMA', selected_stock)
                    arima_metrics = evaluate_predictions(test_data, st.session_state.arima_predictions, 'ARIMA')
                    st.subheader("ARIMA Model Evaluation")
                    st.session_state.arima_metrics_df = pd.DataFrame([arima_metrics])
                    st.write(st.session_state.arima_metrics_df)
                    st.session_state.trained_model = arima_model  # save model
                    
                else:
                    st.error("ARIMA model training failed.")

            elif selected_model == 'Holt-Winters':
                hw_predictions, hw_model = run_holt_winters(train_data, test_data)
                st.session_state.hw_predictions = hw_predictions
                if st.session_state.hw_predictions is not None:
                    plot_plotly_predictions(train_data, test_data, st.session_state.hw_predictions, 'Holt-Winters', selected_stock)
                    hw_metrics = evaluate_predictions(test_data, st.session_state.hw_predictions, 'Holt-Winters')
                    st.session_state.hw_metrics = hw_metrics
                    st.subheader("Holt-Winters Model Evaluation")
                    st.write(pd.DataFrame([st.session_state.hw_metrics]))
                    st.session_state.trained_model = hw_model  # save model
                    
                else:
                    st.error("Holt-Winters model training failed.")

            elif selected_model == 'Prophet':
                prophet_predictions, prophet_model = run_prophet(train_data, test_data)
                st.session_state.prophet_predictions = prophet_predictions
                if st.session_state.prophet_predictions is not None:
                    plot_plotly_predictions(train_data, test_data, st.session_state.prophet_predictions, 'Prophet', selected_stock)
                    prophet_metrics = evaluate_predictions(test_data, st.session_state.prophet_predictions, 'Prophet')
                    st.session_state.prophet_metrics = prophet_metrics
                    st.subheader("Prophet Model Evaluation")
                    st.write(pd.DataFrame([st.session_state.prophet_metrics ]))
                    st.session_state.trained_model = prophet_model  # save model
                    
                else:
                    st.error("Prophet model training failed.")
        st.success(f"{selected_model} model trained successfully!")

    # --- 4. Future Predictions ---
    st.header("Future Predictions")
    future_steps = st.number_input("Number of Future Days to Forecast", min_value=1, max_value=365, value=30)

    if st.button("Make Future Predictions"):
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model first before making future predictions.")
        else:
            with st.spinner(f"Forecasting {future_steps} days into the future..."):
                try:
                    if selected_model == 'LSTM':
                        model = st.session_state.trained_model
                        scaler = st.session_state.scaler
                        n_steps = st.session_state.n_steps
                        last_n_days = st.session_state.stock_data['Close'].values[-n_steps:].reshape(-1, 1)
                        scaled_last_n_days = scaler.transform(last_n_days)
                        X_future = []
                        X_future.append(scaled_last_n_days)
                        X_future = np.array(X_future).reshape((1, n_steps, 1))

                        future_predictions = []
                        for _ in range(future_steps):
                            future_predictions_scaled = model.predict(X_future, verbose=0)
                            future_predictions.append(future_predictions_scaled[0, 0])
                            X_future = np.append(X_future[:, 1:, :], future_predictions_scaled.reshape(1, 1, 1), axis=1)

                        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
                        print(future_predictions)
                        future_dates = pd.to_datetime([st.session_state.stock_data.index[-1] + pd.Timedelta(days=i) for i in range(1, future_steps + 1)])
                        future_predictions_df = pd.DataFrame({'ds': future_dates, 'yhat': future_predictions.flatten()})
                        st.subheader(f"Future {selected_model} Predictions")
                        st.dataframe(future_predictions_df)
                        plot_plotly_predictions(st.session_state.stock_data.iloc[-60:], st.session_state.stock_data.iloc[-future_steps:], future_predictions_df, f'{selected_model} Future', selected_stock)
                        st.session_state.future_predictions_df = future_predictions_df

                    elif selected_model == 'ARIMA':
                        model_fit = st.session_state.trained_model
                        future_predictions = model_fit.forecast(steps=future_steps)
                        future_dates = pd.to_datetime([st.session_state.stock_data.index[-1] + pd.Timedelta(days=i) for i in range(1, future_steps + 1)])
                        future_predictions_df = pd.DataFrame({'ds': future_dates, 'yhat': future_predictions})
                        st.subheader(f"Future {selected_model} Predictions")
                        st.dataframe(future_predictions_df)
                        # plot_plotly_predictions(stock_data.iloc[-60:], pd.DataFrame(), future_predictions_df, f'{selected_model} Future', selected_stock)
                        plot_plotly_predictions(st.session_state.stock_data.iloc[-60:], st.session_state.stock_data.iloc[-future_steps:], future_predictions_df, f'{selected_model} Future', selected_stock)
                        st.session_state.future_predictions_df = future_predictions_df

                    elif selected_model == 'Holt-Winters':
                        model_fit = st.session_state.trained_model
                        future_predictions = model_fit.forecast(steps=future_steps)
                        future_dates = pd.to_datetime([st.session_state.stock_data.index[-1] + pd.Timedelta(days=i) for i in range(1, future_steps + 1)])
                        future_predictions_df = pd.DataFrame({'ds': future_dates, 'yhat': future_predictions})
                        st.subheader(f"Future {selected_model} Predictions")
                        st.dataframe(future_predictions_df)
                        plot_plotly_predictions(st.session_state.stock_data.iloc[-60:], st.session_state.stock_data.iloc[-future_steps:], future_predictions_df, f'{selected_model} Future', selected_stock)
                        st.session_state.future_predictions_df = future_predictions_df

                    elif selected_model == 'Prophet':
                        model = st.session_state.trained_model
                        future_dates = pd.DataFrame({'ds': pd.to_datetime([st.session_state.stock_data.index[-1] + pd.Timedelta(days=i) for i in range(1, future_steps + 1)])})
                        future_forecast = model.predict(future_dates)
                        future_predictions_df = future_forecast[['ds', 'yhat']].rename(columns={'yhat': 'yhat'})
                        st.subheader(f"Future {selected_model} Predictions")
                        st.dataframe(future_predictions_df)
                        plot_plotly_predictions(st.session_state.stock_data.iloc[-60:], st.session_state.stock_data.iloc[-future_steps:], future_predictions_df, f'{selected_model} Future', selected_stock)
                        st.session_state.future_predictions_df = future_predictions_df

                except Exception as e:
                    st.error(f"Error making future predictions: {e}")

# if __name__ == "__main__":
#     main()
