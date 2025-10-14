# Auto-converted from Colab notebook.
# A small stdout-capturing wrapper is prepended so we can capture the final printed output
# without changing the core algorithm. The original algorithm code follows unchanged.
import sys, io, os, datetime, traceback
_capture_buffer = io.StringIO()
_orig_stdout = sys.stdout
sys.stdout = _capture_buffer
# --- begin notebook code ---

## predictor v2

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
import shap
from datetime import date


warnings.filterwarnings("ignore")

CONFIDENCE_REQUIRED_TO_SUGGEST = 0.7
threshold = 0.0

# ----------------- 1. Fetch EUR/USD Hourly -----------------
def fetch_eurusd_1h(timezone: str = 'Asia/Kolkata') -> pd.DataFrame:
    eurusd = yf.download(
        'EURUSD=X',
        period='55d',
        interval='1h',
        progress=False,
        auto_adjust=False
    )
    eurusd.dropna(inplace=True)
    if eurusd.empty:
        raise ValueError("No data fetched for EURUSD for given dates")

    if eurusd.index.tz is None:
        eurusd.index = eurusd.index.tz_localize('UTC')
    else:
        eurusd.index = eurusd.index.tz_convert('UTC')

    eurusd = eurusd[eurusd.index.minute == 0]
    eurusd.index = eurusd.index.tz_convert(timezone)

    eurusd.rename(columns={
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)

    eurusd = eurusd.reset_index().rename(columns={'index': 'Date'})
    return eurusd

# Technical Indicators
def add_indicators(df):

    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
    df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    df['roc'] = ta.momentum.ROCIndicator(df['Close']).roc()
    df['ema_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    ema_13 = ta.trend.EMAIndicator(df['Close'], window=13).ema_indicator()
    df['bull_power'] = df['High'] - ema_13
    df['bear_power'] = df['Low'] - ema_13
    df['bull_bear_diff'] = df['bull_power'] - df['bear_power']
    df['bull_bear_ratio'] = df['bull_power'] / (df['bear_power'] + 1e-6)

    return df

# Candle Anatomy & Patterns
def add_candle_features(df):
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['is_doji'] = (df['body'] < (df['High'] - df['Low']) * 0.1).astype(int)
    df['is_bullish_engulf'] = ((df['Open'].shift(1) > df['Close'].shift(1)) &
                               (df['Close'] > df['Open']) &
                               (df['Open'] < df['Close'].shift(1)) &
                               (df['Close'] > df['Open'].shift(1))).astype(int)
    return df

# Support & Resistance
def detect_levels(df, window=20):
    support = df['Low'].rolling(window).min()
    resistance = df['High'].rolling(window).max()
    return support, resistance

def add_sr_distances(df, support, resistance):
    df['dist_support'] = df['Close'] - support
    df['dist_resistance'] = resistance - df['Close']
    return df

# Rolling Stats
def add_rolling_stats(df):
    df['rolling_mean'] = df['Close'].rolling(5).mean()
    df['rolling_std'] = df['Close'].rolling(5).std()
    return df

# Smarter Target Labeling
def add_target(df):
    df['next_close'] = df['Close'].shift(-1)
    df['pct_change'] = (df['next_close'] - df['Close']) / df['Close']
    df['target'] = np.where(df['pct_change'] > threshold, 1,
                            np.where(df['pct_change'] < -threshold, 0, np.nan))
    return df

def add_next_candle_differences(df):
    df['prev_rsi'] = df['rsi'].shift(1)
    df['rsi_change'] = df['rsi'] - df['prev_rsi']

    df['prev_macd'] = df['macd'].shift(1)
    df['macd_change'] = df['macd'] - df['prev_macd']

    df['prev_macd_signal'] = df['macd_signal'].shift(1)
    df['macd_signal_change'] = df['macd_signal'] - df['prev_macd_signal']

    df['prev_adx'] = df['adx'].shift(1)
    df['adx_change'] = df['adx'] - df['prev_adx']

    df['prev_ema_20'] = df['ema_20'].shift(1)
    df['ema_20_change'] = df['ema_20'] - df['prev_ema_20']

    df['prev_ema_50'] = df['ema_50'].shift(1)
    df['ema_50_change'] = df['ema_50'] - df['prev_ema_50']

    df['prev_bull_power'] = df['bull_power'].shift(1)
    df['bull_power_change'] = df['bull_power'] - df['prev_bull_power']

    df['prev_bear_power'] = df['bear_power'].shift(1)
    df['bear_power_change'] = df['bear_power'] - df['prev_bear_power']

    return df

# Feature Selection via SHAP
def select_features(X, y, model):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    importance = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(importance)[-20:]
    return X.iloc[:, top_indices]

# Bayesian Optimization
def optimize_model(X, y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
    search = BayesSearchCV(
        pipe,
        {
            'clf__n_estimators': (100, 500),
            'clf__max_depth': (3, 10),
            'clf__learning_rate': (0.01, 0.3, 'log-uniform')
        },
        n_iter=20,
        cv=3,
        scoring='accuracy',
        random_state=42
    )
    search.fit(X, y)
    return search.best_estimator_


# Train with Sliding Window
def train_model(df, window_size=200):

    df = add_indicators(df)

    df = add_candle_features(df)

    support, resistance = detect_levels(df)

    df = add_sr_distances(df, support, resistance)

    df = add_rolling_stats(df)

    df = add_next_candle_differences(df)

    df = add_target(df)
    df.iloc[-1] = df.iloc[-1].fillna(0)
    df.dropna(inplace=True)

    features = [col for col in df.columns if col not in ['target', 'next_close', 'pct_change', 'Date']]
    X = df[features]
    y = df['target']

    X_window = X[-window_size:]
    y_window = y[-window_size:]

    model = optimize_model(X_window, y_window)
    X_selected = select_features(X_window, y_window, model.named_steps['clf'])
    model.fit(X_selected, y_window)

    return model, X_selected.columns, model.named_steps['scaler']

# Meta-Prediction with Confidence
def predict_latest(df, model, selected_features, scaler):
    latest = df.tail(1)[selected_features]
    latest_scaled = scaler.transform(latest)
    proba = model.predict_proba(latest_scaled)[0]
    pred = np.argmax(proba)
    confidence = proba[pred]
    direction = "Up (Green)" if pred == 1 else "Down (Red)"

    print("\n1)\033[4m Input\033[0m :--\n")
    print(df.iloc[-1])

    print("\n2)\033[4m Output\033[0m :--")
    if confidence > CONFIDENCE_REQUIRED_TO_SUGGEST:
        print(f"\n***** High-Confidence || Prediction: {direction} || Confidence Level: {confidence:.2f} *****")
    else:
        print(f"\n***** Low Confidence || No Action Recommended || Prediction: {direction} || Confidence Level: {confidence:.2f} *****")

if __name__ == "__main__":
    print("##########")
    print("Application starting...")
    print("##########\n")
    print(f"--->> \033[4mFetching real-time EUR/USD data || Timeframe: 1hr\033[0m...")
    df = fetch_eurusd_1h()
    print(f"--->> Data Fetched || Length: {len(df)} || Sample:\n")
    df.rename(columns={'Datetime': 'Date'}, inplace=True)
    # print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head())
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ["Date","Open", "High", "Low", "Close", "Volume"]
    df.drop("Volume",  axis=1, inplace=True)

    print(df)



    print("--->> \033[4mTraining model\033[0m...")
    model, selected_features, scaler = train_model(df)

    print("\n1) \033[4mModel shown below\033[0m :--")
    print(model, "\n")
    print("2) \033[4mIndependent Scalers for training\033[0m :--")
    print(selected_features, "\n")
    print(f"3)\033[4m Feature Scaling used for training\033[0m :-- {scaler}\n")
    print("--->> \033[4mModel trained successfully\033[0m!\n")
    print("--->> \033[4mPredicting next candle\033[0m...")
    predict_latest(df, model, selected_features, scaler)


# --- end notebook code ---
# restore stdout and capture the last printed line as the prediction result
sys.stdout = _orig_stdout
try:
    all_output = _capture_buffer.getvalue().strip().splitlines()
except Exception as e:
    all_output = ["[error capturing output] " + str(e)]

if len(all_output) > 0:
    # choose the last non-empty printed line
    last_lines = [ln for ln in all_output if ln.strip() != ""]
    last_line = last_lines[-1] if last_lines else all_output[-1]
else:
    last_line = "No output captured from the notebook."

# ---- 🕒 Use IST time (India Standard Time) for accurate logs ----
import datetime, pytz, os, sys, traceback, requests

ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.datetime.now(ist).strftime("%Y-%m-%d %I:%M %p IST")

message = f"📊 EUR/USD Prediction Update\n\nTime: {now_ist}\nResult: {last_line}"

# ---- 🔐 Telegram setup ----
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
#CHAT_ID = os.getenv("CHAT_ID", "")  # single secret with comma-separated IDs
CHAT_ID = 753303744

if TELEGRAM_TOKEN and CHAT_ID:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        chat_ids = [cid.strip() for cid in CHAT_ID.split(",") if cid.strip()]
        sent_count = 0

        for chat_id in chat_ids:
            params = {"chat_id": chat_id, "text": message}
            r = requests.get(url, params=params, timeout=15)
            if r.ok:
                sent_count += 1
                print(f"✅ [{now_ist}] Message sent to {chat_id}")
            else:
                print(f"⚠️ [{now_ist}] Failed to send to {chat_id}: {r.text}")

        print(f"📨 [{now_ist}] Done. Sent to {sent_count} chats total.")
    except Exception as e:
        print(f"❌ [{now_ist}] Failed to send Telegram message:", e, file=sys.stderr)
        traceback.print_exc()
else:
    print(f"⚠️ [{now_ist}] TELEGRAM_TOKEN or CHAT_ID not set. Message not sent.")
    print("Captured output:")
    print(last_line)

print("\n--- Full captured notebook stdout ---")
print(_capture_buffer.getvalue())
