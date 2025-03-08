import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
from PIL import Image

# Streamlit Page Config
st.set_page_config(page_title='Yahoo Finance Analysis', page_icon='images/logo1.png', layout='wide', initial_sidebar_state='auto')

# Logo and Title
logo = Image.open('images/logo1.png')
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=120)  # Adjust width if needed
with col2:
    st.title('Yahoo Finance Analysis')

# Sidebar
st.sidebar.title('⚙️ Settings')
ticker = st.sidebar.text_input('📈 Enter Stock Ticker', 'AAPL')
start_date = st.sidebar.date_input('📅 Start Date', pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input('📅 End Date', pd.to_datetime('2024-12-31'))
metric = st.sidebar.selectbox('📊 Select Metric', ['Open', 'Close', 'High', 'Low', 'Volume'])
view_option = st.sidebar.radio('📉 Select Display Options', ['Historical Data', 'Forecast', 'Both'])


#Fetching Data
@st.cache_data
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        df.reset_index(inplace=True)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
        return None

#Load Data
if ticker and start_date and end_date:
    df = fetch_data(ticker, start_date, end_date)

    if df is not None:
        st.subheader(f'📝 {ticker} Stock Data Overview')
        st.write(df.head())

        # Missing Values
        st.write('🧹 **Missing Values:**')
        st.write(df.isnull().sum())

        # Data Summary
        st.write('📊 **Data Summary:**')
        st.write(df.describe())

        # Feature Engineering
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

        # Historical Data Visualization
        if view_option in ['Historical Data', 'Both']:
            st.subheader(f"📈 {metric} Price Over Time")
            if 'Date' in df.columns and metric in df.columns:
                fig = px.line(df, x='Date', y=metric, title=f'{ticker} {metric} Over Time', line_shape='spline')
                st.plotly_chart(fig)
        #Candlestic Chart
        st.subheader(f"🕯️ Candlestick Chart")
        candlestick_fig = go.Figure(data=[go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
        )])
        candlestick_fig.update_layout(title=f'{ticker} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(candlestick_fig)

        #Moving Averages
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        st.subheader('📈 Moving Averages')
        ma_fig = go.Figure()    
        ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
        ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_30'], mode='lines', name='MA 30'))
        ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_50'], mode='lines', name='MA 50'))
        st.plotly_chart(ma_fig)

        # Seasonal Decomposition
        st.subheader('🌊 Seasonal Decomposition')
        try:
            decomposition = seasonal_decompose(df['Close'].dropna(), period=30, model='additive')
            fig_trend = px.line(x=df['Date'], y=decomposition.trend, title='Trend Component')
            fig_seasonal = px.line(x=df['Date'], y=decomposition.seasonal, title='Seasonal Seasonal Component')
            fig_residual = px.line(x=df['Date'], y=decomposition.resid, title='Residual Component')

            st.plotly_chart(fig_trend)
            st.plotly_chart(fig_seasonal)
            st.plotly_chart(fig_residual)

        except Exception as e:
            st.warning(f"⚠️ An error occurred: {e}")

        # Forecasting with SARIMA
        if view_option in ['Forecast', 'Both']:
            st.subheader(f'🔮 Forecasting {metric} Price with SARIMA')

            try:
                df.set_index('Date', inplace=True)

                train_size = int(len(df) * 0.8)
                train, test = df['Close'][:train_size], df['Close'][train_size:]

                model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                results = model.fit()

                forecast = results.get_forecast(steps=len(test))

                mse = mean_squared_error(test, forecast)
                mae = mean_absolute_error(test, forecast)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric('📉 Mean Absolute Error (MAE)', f'{mae:.2f}')
                with col2:
                    st.metric('📉 Mean Squared Error (MSE)', f'{mse:.2f}')

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Actual'))
                fig_forecast.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Forecast', line=dict(color='red')))
                st.plotly_chart(fig_forecast)

                forecast_steps = 365
                future_forecast = results.predict(start=len(df), end=len(df) + forecast_steps - 1, dynamic=True)
                future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='B')[1:]
                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})

                future_fig = px.line(forecast_df, x='Date', y='Forecast', title='1-Year Future Forecast')
                st.plotly_chart(future_fig)
            except Exception as e:
                st.error(f"⚠️ Forecasting failed: {e}")

        df.reset_index(inplace=True)

# Footer
st.markdown('---')
st.write("📊 Built with ❤️ using Streamlit, Yahoo Finance API, and Python")
st.markdown('[🔗 My LinkedIn](https://www.linkedin.com/in/muhammad-ahmad-faizan/) | [🔗 My GitHub](https://github.com/Muhammad-Ahmad-Faizan)')
st.write("© 2025 | All Rights Reserved")