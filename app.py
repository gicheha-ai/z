import os
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from alpha_vantage.forex import Forex
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.utils
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# API Configuration
API_KEYS = ['QE0TAOPZZN1VT8LH', 'QSCWPKVUYLOD506J']
current_key_index = 0

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

def get_alpha_vantage_client():
    """Get Alpha Vantage client with current API key"""
    global current_key_index
    return Forex(API_KEYS[current_key_index])

def switch_api_key():
    """Switch to next API key if rate limited"""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    print(f"Switched to API key index: {current_key_index}")

def fetch_forex_data(symbol='EUR/USD', interval='1min', outputsize='full'):
    """Fetch forex data from Alpha Vantage"""
    for attempt in range(len(API_KEYS)):
        try:
            fx = get_alpha_vantage_client()
            data, meta_data = fx.get_currency_exchange_intraday(
                symbol.split('/')[0],
                symbol.split('/')[1],
                interval=interval,
                outputsize=outputsize
            )
            
            if data.empty:
                raise Exception("Empty data returned")
                
            # Convert to proper format
            data = data.sort_index()
            data.columns = ['open', 'high', 'low', 'close']
            return data, meta_data
            
        except Exception as e:
            print(f"API attempt {attempt + 1} failed: {str(e)}")
            switch_api_key()
            time.sleep(1)
    
    raise Exception("All API keys exhausted")

def calculate_all_indicators(df):
    """Calculate multiple technical indicators"""
    try:
        # Ensure numeric data
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Price-based indicators
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        df['EMA_12'] = ta.ema(df['close'], length=12)
        df['EMA_26'] = ta.ema(df['close'], length=26)
        
        # RSI
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        if bb is not None:
            df['BB_upper'] = bb['BBU_20_2.0']
            df['BB_middle'] = bb['BBM_20_2.0']
            df['BB_lower'] = bb['BBL_20_2.0']
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']
        
        # ATR (Volatility)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Volume indicators (using close price changes as proxy)
        df['Volume'] = df['close'].diff().abs()
        df['OBV'] = ta.obv(df['close'], df['Volume'])
        
        # Additional indicators
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        df['Williams_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # Ichimoku Cloud (simplified)
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        if ichimoku is not None:
            df['Ichimoku_Conversion'] = ichimoku['ITS_9']
            df['Ichimoku_Base'] = ichimoku['IKS_26']
        
        # Price patterns
        df['Doji'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name="doji")
        
        # Calculate returns
        df['Returns'] = df['close'].pct_change()
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Support and Resistance levels (simplified)
        df['Resistance'] = df['high'].rolling(20).max()
        df['Support'] = df['low'].rolling(20).min()
        
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
        'BB_width', 'ATR', 'ADX', 'Williams_R', 'Returns',
        'MA_Crossover', 'Price_vs_SMA', 'Log_Returns'
    ]
    
    # Only use features that exist in dataframe
    available_features = [f for f in features if f in df.columns]
    
    X = pd.DataFrame()
    for feature in available_features:
        X[feature] = df[feature]
        # Add lagged features
        for lag in range(1, 4):
            X[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        # Add rolling statistics
        X[f'{feature}_mean_5'] = df[feature].rolling(5).mean()
        X[f'{feature}_std_5'] = df[feature].rolling(5).std()
    
    # Add price momentum features
    if 'close' in df.columns:
        for period in [5, 10, 20]:
            X[f'momentum_{period}'] = df['close'].pct_change(period)
    
    # Add volatility features
    if 'Log_Returns' in df.columns:
        X['volatility_10'] = df['Log_Returns'].rolling(10).std()
    
    # Clean data
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return X.iloc[-lookback:]

def create_labels(df, horizon=5):
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
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Use last 80% for training
        split_idx = int(len(X_scaled) * 0.8)
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
        
        if len(X) < 30:
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
        
        # Adjust confidence based on recent accuracy
        recent_predictions = model.predict(scaler.transform(X[-20:]))
        recent_accuracy = np.mean(recent_predictions == y[-20:])
        adjusted_confidence = (confidence + recent_accuracy) / 2
        
        return direction, adjusted_confidence
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return "NEUTRAL", 0.5

def generate_trading_signal(prediction, confidence, current_price, df):
    """Generate trading signal based on prediction"""
    try:
        # Get recent price action
        recent_trend = df['close'].iloc[-5:].mean() - df['close'].iloc[-10:-5].mean()
        
        # Get indicator status
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        macd_hist = df['MACD_hist'].iloc[-1] if 'MACD_hist' in df.columns else 0
        
        # Decision logic
        if prediction == "BULLISH" and confidence > 0.65:
            if rsi < 70 and macd_hist > 0 and recent_trend > 0:
                return "BUY", confidence
            elif confidence > 0.75:
                return "BUY", confidence
                
        elif prediction == "BEARISH" and confidence > 0.65:
            if rsi > 30 and macd_hist < 0 and recent_trend < 0:
                return "SELL", confidence
            elif confidence > 0.75:
                return "SELL", confidence
        
        # Check for strong signals
        if confidence > 0.8:
            return "BUY" if prediction == "BULLISH" else "SELL", confidence
            
        # Check for extreme RSI
        if rsi > 80:
            return "SELL", 0.7
        elif rsi < 20:
            return "BUY", 0.7
            
        return "HOLD", confidence
        
    except Exception as e:
        print(f"Error generating signal: {str(e)}")
        return "HOLD", 0.5

def create_chart_data(df):
    """Create chart data for visualization"""
    try:
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df.index[-100:],
            open=df['open'].iloc[-100:],
            high=df['high'].iloc[-100:],
            low=df['low'].iloc[-100:],
            close=df['close'].iloc[-100:],
            name='EUR/USD'
        ))
        
        # Add moving averages
        if 'SMA_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index[-100:],
                y=df['SMA_20'].iloc[-100:],
                line=dict(color='orange', width=1),
                name='SMA 20'
            ))
        
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index[-100:],
                y=df['SMA_50'].iloc[-100:],
                line=dict(color='blue', width=1),
                name='SMA 50'
            ))
        
        # Update layout
        fig.update_layout(
            title='EUR/USD Price Chart',
            yaxis_title='Price',
            xaxis_title='Time',
            template='plotly_dark',
            height=500,
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
            df, meta_data = fetch_forex_data()
            
            if df.empty:
                print("No data fetched")
                time.sleep(60)
                continue
            
            # Calculate indicators
            df = calculate_all_indicators(df)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Make prediction
            prediction, confidence = make_prediction(df)
            
            # Generate trading signal
            action, signal_confidence = generate_trading_signal(
                prediction, confidence, current_price, df
            )
            
            # Calculate indicator summary
            indicators_summary = {}
            if 'RSI' in df.columns:
                indicators_summary['RSI'] = round(df['RSI'].iloc[-1], 2)
            if 'MACD' in df.columns:
                indicators_summary['MACD'] = round(df['MACD'].iloc[-1], 4)
            if 'Stoch_K' in df.columns:
                indicators_summary['Stochastic'] = round(df['Stoch_K'].iloc[-1], 2)
            
            # Create chart
            chart_data = create_chart_data(df)
            
            # Update global prediction data
            prediction_data = {
                'current_price': round(current_price, 5),
                'prediction': prediction,
                'confidence': round(signal_confidence * 100, 2),
                'action': action,
                'indicators': indicators_summary,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chart_data': chart_data
            }
            
            print(f"Updated: {prediction_data}")
            
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
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

def start_background_thread():
    """Start background thread for predictions"""
    thread = threading.Thread(target=update_predictions, daemon=True)
    thread.start()

if __name__ == '__main__':
    # Start background thread
    start_background_thread()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)