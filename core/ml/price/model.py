import pandas as pd
import numpy as np
import ta
import lightgbm
import xgboost as xgb
import yfinance as yf

from abc import ABC, abstractmethod

from core.data.instrument.instrument import Instrument

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datetime import datetime, timedelta

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
}

MARKET_INDEX_TICKER = "SPY"
VIX_TICKER = "^VIX"

_market_series_cache: dict[str, pd.DataFrame] = {}


def _fetch_reference_series(ticker_symbol: str) -> pd.DataFrame:
    if ticker_symbol not in _market_series_cache:
        df = yf.Ticker(ticker_symbol).history(period='max', interval='1d')
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        _market_series_cache[ticker_symbol] = df
    return _market_series_cache[ticker_symbol].copy()

class PriceDirPredictor(ABC):
    def __init__(self, instrument: Instrument, horizon_days = 5):
        self._instrument = instrument
        self.horizon = horizon_days
        self._model = self._init_model()

    @property
    def instrument(self):
        return self._instrument

    @property
    def features(self) -> list:
        return [
            'Return_1D',
            'Return_5D',
            'Return_10D',
            'SMA_Ratio',
            'RSI_14',
            'MACD',
            'ATR_14',
            'Volume_Change',
            'Volume_Price_Force',
            'Daily_Range_Normalized',
        ]
    
    @abstractmethod
    def _init_model(self):
        pass

    def _fetch_data(self, end_date: datetime = None) -> pd.DataFrame:
        df = self._instrument.market_data_history

        if df is None or df.empty:
            raise ValueError(f"No price history found for symbol: {self._instrument.symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if end_date is not None:
            df = df.loc[:end_date]
            
        return df.copy()
    
    def _compute_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        close_col = 'Close' 
        high_col = 'High'
        low_col = 'Low'
        vol_col = 'Volume'

        data['prev_close'] = data[close_col].shift(1)

        data['Return_1D'] = data[close_col].pct_change(1)
        data['Return_5D'] = data[close_col].pct_change(5)
        data['Return_10D'] = data[close_col].pct_change(10)

        data['SMA_5'] = ta.trend.sma_indicator(data[close_col], window=5)
        data['SMA_10'] = ta.trend.sma_indicator(data[close_col], window=10)
        data['SMA_20'] = ta.trend.sma_indicator(data[close_col], window=20)
        data['SMA_50'] = ta.trend.sma_indicator(data[close_col], window=50)
        data['SMA_Ratio'] = data['SMA_10'] / data['SMA_50']

        data['RSI_14'] = ta.momentum.rsi(data[close_col], window=14)
        data['MACD'] = ta.trend.macd_diff(data[close_col])
        data['ATR_14'] = ta.volatility.average_true_range(data[high_col], data[low_col], data[close_col], window=14)
        data['Volume_Change'] = data[vol_col].pct_change(1)

        data['Volume_Price_Force'] = data['Volume_Change'] * data['Return_1D']
        data['Daily_Range_Normalized'] = (data[high_col] - data[low_col]) / data['ATR_14']

        data = data.replace([np.inf, -np.inf], np.nan)

        return data

    def _prepare_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self._compute_raw_features(df)

        future_close = data['Close'].shift(-self.horizon)
        data['Target'] = (future_close > data['Close']).astype(int)

        return data.dropna()
    
    
    def train(self, end_date: datetime = None) -> dict:
        raw_df = self._fetch_data(end_date=end_date)
        processed_df = self._prepare_training_features(raw_df)

        if end_date is not None:
            valid_cutoff = end_date - timedelta(days=self.horizon)
            processed_df = processed_df.loc[:valid_cutoff]

        X = processed_df[self.features]
        y = processed_df['Target']

        self._model.fit(X, y)

        y_pred = self._model.predict(X)
        acc = accuracy_score(y, y_pred)

        return {
            'symbol': self._instrument.symbol,
            'train_end_date': end_date,
            'train_samples': len(X),
            'train_accuracy': acc
        }

    def predict_at_date(self, as_of_date: datetime) -> dict:
        raw_df = self._fetch_data(end_date=as_of_date)
        features_df = self._compute_raw_features(raw_df).dropna()

        if features_df.empty:
            return {
                'date': as_of_date,
                'signal': 0,
                'confidence_up': 0.5,
                'confidence_down': 0.5,
                'atr': 0.0,
                'close': 0.0,
                'sma_5': 0.0,
                'sma_20': 0.0,
                'sma_50': 0.0,
                'prev_close': 0.0
            }

        latest_row = features_df.iloc[-1]
        latest_X = features_df[self.features].iloc[[-1]]
        
        pred = self._model.predict(latest_X)[0]
        prob = self._model.predict_proba(latest_X)[0]

        return {
            'date': as_of_date,
            'signal': int(pred),
            'confidence_up': float(prob[1]),
            'confidence_down': float(prob[0]),
            'atr': float(latest_row['ATR_14']),
            'close': float(latest_row['Close']),
            'sma_5': float(latest_row['SMA_5']),
            'sma_20': float(latest_row['SMA_20']),
            'sma_50': float(latest_row['SMA_50']),
            'prev_close': float(latest_row['prev_close'])
        }
    
N_ESTIMATORS = 200
MAX_DEPTH = 5

class PriceDirPredictorRF(PriceDirPredictor):

    def _init_model(self):
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            max_depth=MAX_DEPTH,          
            random_state=42, 
            class_weight="balanced"
        )
    

class PriceDirPredictorLGBM(PriceDirPredictor):

    def _init_model(self):
        return lightgbm.LGBMClassifier(
            n_estimators=150,
            max_depth=4,  
            num_leaves=15, 
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )

    
class PriceDirPredictorXGB(PriceDirPredictor):
    def _init_model(self):
        return xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss'
        )