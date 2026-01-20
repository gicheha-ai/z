import os
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.utils
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# API Configuration - Alpha Vantage
API_KEYS = ['QE0TAOPZZN1VT8LH', 'QSCWPKVUYLOD506J']
current_key_index = 0
BASE_URL = "https://www.alphavantage.co/query"

# Global variables for storing predictions
prediction_data = {
    'current_price': 0,
    'prediction': 'NEUTRAL',
    'confidence': 0,
    'action': 'HOLD',
    'indicators': {},
    'timestamp': None,
    'chart_data': None
}

def switch_api_key():
    """Switch to next API key if rate limited"""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    print(f"Switched to API key index: {current_key_index}")

def fetch_alpha_vantage_data(symbol='EURUSD', interval='1min', outputsize='compact'):
    """Fetch forex data from Alpha Vantage using direct API calls"""
    for attempt in range(len(API_KEYS)):
        try:
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': symbol[:3],
                'to_symbol': symbol[3:],
                'interval': interval,
                'outputsize': outputsize,
                'apikey': API_KEYS[current_key_index],
                'datatype': 'json'
            }
            
            response = requests.get(BASE_URL, params=params, timeout=10)
            data = response.json()
            
            if "Time Series FX (" + interval + ")" not in data:
                if "Note" in data:
                    print(f"Rate limited: {data['Note']}")
                    switch_api_key()
                    continue
                raise Exception(f"API Error: {data}")
            
            time_series = data["Time Series FX (" + interval + ")"]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close']
            
            # Convert to numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            
            return df, data.get("Meta Data", {})
            
        except Exception as e:
            print(f"API attempt {attempt + 1} failed: {str(e)}")
            switch_api_key()
            time.sleep(2)
    
    # Fallback to mock data if API fails
    print("Using mock data as fallback")
    return create_mock_data(), {}

def create_mock_data():
    """Create mock data for testing when API fails"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    base_price = 1.0850
    prices = []
    
    for i in range(len(dates)):
        # Simulate some price movement
        change = np.random.normal(0, 0.0005)
        base_price += change
        prices.append(base_price)
    
    df = pd.DataFrame(index=dates[-100:])
    df['open'] = [p - abs(np.random.normal(0, 0.0002)) for p in prices]
    df['high'] = [p + abs(np.random.normal(0, 0.0003)) for p in prices]
    df['low'] = [p - abs(np.random.normal(0, 0.0003)) for p in prices]
    df['close'] = prices
    
    return df[-100:]

def calculate_all_indicators(df):
    """Calculate multiple technical indicators"""
    try:
        # Ensure numeric data
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Price-based indicators
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['EMA_12'] = ta.ema(df['close'], length=12)
        df['EMA_26'] = ta.ema(df['close'], length=26)
        
        # RSI
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df['BB_upper'] = bb['BBU_20_2.0']
            df['BB_middle'] = bb['BBM_20_2.0']
            df['BB_lower'] = bb['BBL_20_2.0']
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']
        
        # ATR (Volatility)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Williams %R
        df['Williams_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # ADX
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_result is not None:
            df['ADX'] = adx_result['ADX_14']
        
        # Calculate returns
        df['Returns'] = df['close'].pct_change()
        
        # Moving average crossovers
        df['MA_Crossover'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        df['Price_vs_SMA'] = np.where(df['close'] > df['SMA_20'], 1, -1)
        
        # Clean NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return df

def prepare_features(df, lookback=30):
    """Prepare features for machine learning"""
    features = [
        'RSI', 'MACD', 'MACD_hist', 'Stoch_K', 'Stoch_D',
        'ATR', 'ADX', 'Williams_R', 'Returns',
        'MA_Crossover', 'Price_vs_SMA'
    ]
    
    # Only use features that exist in dataframe
    available_features = [f for f in features if f in df.columns]
    
    X = pd.DataFrame()
    for feature in available_features:
        X[feature] = df[feature]
        # Add lagged features
        for lag in range(1, 3):
            X[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    
    # Add price momentum features
    if 'close' in df.columns:
        for period in [5, 10]:
            X[f'momentum_{period}'] = df['close'].pct_change(period)
    
    # Clean data
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return X.iloc[-lookback:]

def create_labels(df, horizon=3):
    """Create labels for prediction (1 for price increase, 0 for decrease)"""
    future_returns = df['close'].shift(-horizon) / df['close'] - 1
    labels = (future_returns > 0).astype(int)
    return labels

def train_model(X, y):
    """Train Random Forest model"""
    try:
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Use data for training
        split_idx = max(10, int(len(X_scaled) * 0.7))
        model.fit(X_scaled[:split_idx], y[:split_idx])
        
        return model, scaler
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None, None

def make_prediction(df):
    """Make prediction using trained model"""
    try:
        # Prepare features
        X = prepare_features(df)
        
        if len(X) < 20:
            return "NEUTRAL", 0.5
        
        # Create labels
        y = create_labels(df)
        
        if len(y) < len(X):
            y = y.iloc[:len(X)]
        
        # Train model
        model, scaler = train_model(X, y)
        
        if model is None:
            return "NEUTRAL", 0.5
        
        # Prepare latest features for prediction
        X_latest = X.iloc[-1:].copy()
        X_latest_scaled = scaler.transform(X_latest)
        
        # Make prediction
        prediction = model.predict(X_latest_scaled)[0]
        probabilities = model.predict_proba(X_latest_scaled)[0]
        confidence = max(probabilities)
        
        # Determine direction
        direction = "BULLISH" if prediction == 1 else "BEARISH"
        
        return direction, confidence
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return "NEUTRAL", 0.5

def generate_trading_signal(prediction, confidence, current_price, df):
    """Generate trading signal based on prediction"""
    try:
        # Get indicator status
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        macd_hist = df['MACD_hist'].iloc[-1] if 'MACD_hist' in df.columns else 0
        
        # Decision logic
        if prediction == "BULLISH" and confidence > 0.6:
            if rsi < 70 and macd_hist > -0.0001:
                return "BUY", confidence
                
        elif prediction == "BEARISH" and confidence > 0.6:
            if rsi > 30 and macd_hist < 0.0001:
                return "SELL", confidence
        
        # Check for extreme RSI
        if rsi > 75:
            return "SELL", 0.65
        elif rsi < 25:
            return "BUY", 0.65
            
        return "HOLD", confidence
        
    except Exception as e:
        print(f"Error generating signal: {str(e)}")
        return "HOLD", 0.5

def create_chart_data(df):
    """Create chart data for visualization"""
    try:
        # Use last 50 points for chart
        chart_df = df.iloc[-50:] if len(df) > 50 else df
        
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['open'],
            high=chart_df['high'],
            low=chart_df['low'],
            close=chart_df['close'],
            name='EUR/USD'
        ))
        
        # Add moving averages if available
        if 'SMA_20' in chart_df.columns:
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df['SMA_20'],
                line=dict(color='orange', width=1),
                name='SMA 20'
            ))
        
        # Update layout
        fig.update_layout(
            title='EUR/USD Price Chart (Last 50 periods)',
            yaxis_title='Price',
            xaxis_title='Time',
            template='plotly_dark',
            height=400,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Error creating chart: {str(e)}")
        return None

def update_predictions():
    """Update predictions periodically"""
    global prediction_data
    
    while True:
        try:
            print(f"Updating predictions at {datetime.now()}")
            
            # Fetch data
            df, meta_data = fetch_alpha_vantage_data()
            
            if df.empty:
                print("No data fetched, using mock data")
                df = create_mock_data()
            
            # Calculate indicators
            df = calculate_all_indicators(df)
            
            # Get current price
            current_price = df['close'].iloc[-1] if len(df) > 0 else 1.0850
            
            # Make prediction
            prediction, confidence = make_prediction(df)
            
            # Generate trading signal
            action, signal_confidence = generate_trading_signal(
                prediction, confidence, current_price, df
            )
            
            # Calculate indicator summary
            indicators_summary = {}
            indicator_mapping = {
                'RSI': 'RSI',
                'MACD': 'MACD',
                'Stoch_K': 'Stochastic',
                'ATR': 'ATR',
                'ADX': 'ADX',
                'Williams_R': 'Williams_R'
            }
            
            for col, name in indicator_mapping.items():
                if col in df.columns:
                    indicators_summary[name] = round(df[col].iloc[-1], 4)
            
            # Create chart
            chart_data = create_chart_data(df)
            
            # Update global prediction data
            prediction_data = {
                'current_price': round(float(current_price), 5),
                'prediction': prediction,
                'confidence': round(float(signal_confidence) * 100, 2),
                'action': action,
                'indicators': indicators_summary,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chart_data': chart_data
            }
            
            print(f"Updated: Price={prediction_data['current_price']}, "
                  f"Prediction={prediction_data['prediction']}, "
                  f"Action={prediction_data['action']}")
            
            # Wait for next update
            time.sleep(60)
            
        except Exception as e:
            print(f"Error in update_predictions: {str(e)}")
            time.sleep(30)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get current predictions"""
    return jsonify(prediction_data)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'api_keys_available': len(API_KEYS)
    })

def start_background_thread():
    """Start background thread for predictions"""
    thread = threading.Thread(target=update_predictions, daemon=True)
    thread.start()

if __name__ == '__main__':
    # Start background thread
    start_background_thread()
    
    # Initialize with some data
    print("Starting Forex Prediction System...")
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)