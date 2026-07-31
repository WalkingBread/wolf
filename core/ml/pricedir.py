import pandas as pd
import numpy as np
import ta

from abc import ABC, abstractmethod

from core.data.instrument.instrument import Instrument

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class PriceDirPredictor(ABC):
    def __init__(self, instrument: Instrument, horizon_days = 5):
        self._instrument = instrument
        self.horizon = horizon_days

        self._model = self._init_model()

    @abstractmethod
    def _init_model(self):
        pass

    def _fetch_data(self) -> pd.DataFrame:
        df = self._instrument.get_all_historical_market_data()

        if df is None or df.empty:
            raise ValueError(f"No price history found for symbol: {self.ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df.copy()
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        close_col = 'Close' 
        high_col = 'High'
        low_col = 'Low'
        vol_col = 'Volume'

        data['Return_1D'] = data[close_col].pct_change(1)
        data['Return_5D'] = data[close_col].pct_change(5)
        data['Return_10D'] = data[close_col].pct_change(10)

        data['SMA_10'] = ta.trend.sma_indicator(data[close_col], window=10)
        data['SMA_50'] = ta.trend.sma_indicator(data[close_col], window=50)
        data['SMA_Ratio'] = data['SMA_10'] / data['SMA_50']

        data['RSI_14'] = ta.momentum.rsi(data[close_col], window=14)
        data['MACD'] = ta.trend.macd_diff(data[close_col])
        data['ATR_14'] = ta.volatility.average_true_range(data[high_col], data[low_col], data[close_col], window=14)
        data['Volume_Change'] = data[vol_col].pct_change(1)

        data['Volume_Price_Force'] = data['Volume_Change'] * data['Return_1D']
        data['Daily_Range_Normalized'] = (data[high_col] - data[low_col]) / data['ATR_14']

        future_close = data[close_col].shift(-self.horizon)
        data['Target'] = (future_close > data[close_col]).astype(int)

        data = data.replace([np.inf, -np.inf], np.nan)

        return data.dropna()
    
N_ESTIMATORS = 200
MAX_DEPTH = 5

class PriceDirPredictorRF(PriceDirPredictor):
    def __init__(self, instrument: Instrument, horizon_days = 5):
        super().__init__(instrument, horizon_days)

    def _init_model(self):
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            max_depth=MAX_DEPTH,          
            random_state=42, 
            class_weight="balanced"
        )
    
    def train_and_evaluate(self, train_ratio: float = 0.8) -> dict:
        raw_df = self._fetch_data()
        processed_df = self._create_features(raw_df)

        base_features = [
            'Return_1D', 'Return_5D', 'Return_10D', 'SMA_Ratio', 
            'RSI_14', 'MACD', 'ATR_14', 'Volume_Change',
            'Volume_Price_Force', 'Daily_Range_Normalized'
        ]

        X = processed_df[base_features]
        y = processed_df['Target']

        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self._model.fit(X_train, y_train)

        test_preds = self._model.predict(X_test)
        acc = accuracy_score(y_test, test_preds)

        latest_features = X.iloc[[-1]]
        latest_pred = self._model.predict(latest_features)[0]
        latest_prob = self._model.predict_proba(latest_features)[0]

        return {
            "symbol": self._instrument.symbol,
            "accuracy": acc,
            "forecast_direction": "UP" if latest_pred == 1 else "DOWN",
            "confidence_up": float(latest_prob[1]),
            "confidence_down": float(latest_prob[0]),
            "feature_importance": dict(zip(base_features, np.round(self._model.feature_importances_, 4)))
        }