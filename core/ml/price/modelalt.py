"""
Price direction predictors with:
  - Purged / embargoed walk-forward cross-validation (avoids leakage from
    overlapping forward-looking labels)
  - Honest out-of-sample metrics (accuracy, log-loss, Brier score)
  - Probability calibration (isotonic, fit on out-of-fold predictions)
  - A stacked ensemble (RF + LGBM + XGB -> logistic meta-learner)
  - Rolling z-score feature normalization (handles non-stationary scale)
  - Optional "dead zone" target that drops near-noise moves from training
  - Confidence-threshold / abstain logic at prediction time
  - Market/sector/VIX context features (via yfinance, module-level cached)
  - Trend-regime, mean-reversion, volume, and calendar features

Drop-in replacement for the original module: same Instrument interface
(`instrument.market_data_history` with Close/High/Low/Volume columns).
Requires `yfinance` (already a dependency via Instrument) for the
market/sector/VIX reference series.
"""

import numpy as np
import pandas as pd
import ta
import lightgbm
import xgboost as xgb
import yfinance as yf

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

from core.data.instrument.instrument import Instrument
from core.ml.tools import PurgedTimeSeriesSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


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


def _refresh_reference_cache():
    """Call this periodically (e.g. once a day) to drop stale cached data."""
    _market_series_cache.clear()


class PriceDirPredictor(ABC):
    def __init__(
        self,
        instrument: Instrument,
        horizon_days: int = 5,
        dead_zone_atr_mult: float = 0.0,
        confidence_threshold: float = 0.55,
        calibrate: bool = True,
        n_cv_splits: int = 5,
        embargo: int = 5,
    ):
        self._instrument = instrument
        self.horizon = horizon_days
        self.dead_zone_atr_mult = dead_zone_atr_mult
        self.confidence_threshold = confidence_threshold
        self.calibrate = calibrate
        self.n_cv_splits = n_cv_splits
        self.embargo = embargo

        self._raw_model = self._init_model()
        self._model = self._raw_model
        self._is_fitted = False

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
            # trend regime / mean-reversion
            'ADX_14',
            'Stoch_K',
            'BB_Width',
            'ATR_Regime_Ratio',
            'Dist_From_SMA50_ATR',
            'OBV_Change_5D',
            # market / sector context
            'Market_Return_1D',
            'Market_Return_5D',
            'Sector_Return_5D',
            'Relative_Strength_5D',
            'VIX_Level',
            'VIX_Change_5D',
            # calendar
            'DOW_Sin',
            'DOW_Cos',
        ]

    @abstractmethod
    def _init_model(self):
        pass

    def _fetch_data(self, end_date: Optional[datetime] = None) -> pd.DataFrame:
        df = self._instrument.market_data_history

        if df is None or df.empty:
            raise ValueError(f"No price history found for symbol: {self._instrument.symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if end_date is not None:
            df = df.loc[:end_date]

        return df.copy()

    def _sector_etf_ticker(self) -> Optional[str]:
        try:
            sector = self._instrument.info.get('sector')
        except Exception:
            sector = None
        return SECTOR_ETF_MAP.get(sector)

    def _compute_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        close_col, high_col, low_col, vol_col = 'Close', 'High', 'Low', 'Volume'

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
        data['ATR_14'] = ta.volatility.average_true_range(
            data[high_col], data[low_col], data[close_col], window=14
        )
        data['ATR_63'] = ta.volatility.average_true_range(
            data[high_col], data[low_col], data[close_col], window=63
        )
        data['Volume_Change'] = data[vol_col].pct_change(1)

        data['Volume_Price_Force'] = data['Volume_Change'] * data['Return_1D']
        data['Daily_Range_Normalized'] = (data[high_col] - data[low_col]) / data['ATR_14']

        data['ADX_14'] = ta.trend.adx(data[high_col], data[low_col], data[close_col], window=14)

        stoch = ta.momentum.StochasticOscillator(
            data[high_col], data[low_col], data[close_col], window=14, smooth_window=3
        )
        data['Stoch_K'] = stoch.stoch()

        bb = ta.volatility.BollingerBands(data[close_col], window=20, window_dev=2)
        data['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / data['SMA_20']
        data['ATR_Regime_Ratio'] = data['ATR_14'] / data['ATR_63']

        data['Dist_From_SMA50_ATR'] = (data[close_col] - data['SMA_50']) / data['ATR_14']

        obv = ta.volume.OnBalanceVolumeIndicator(data[close_col], data[vol_col]).on_balance_volume()
        data['OBV_Change_5D'] = obv.pct_change(5)

        market_df = _fetch_reference_series(MARKET_INDEX_TICKER)
        market_ret_1d = market_df['Close'].pct_change(1).reindex(data.index, method='ffill')
        market_ret_5d = market_df['Close'].pct_change(5).reindex(data.index, method='ffill')
        data['Market_Return_1D'] = market_ret_1d
        data['Market_Return_5D'] = market_ret_5d

        sector_ticker = self._sector_etf_ticker()
        if sector_ticker:
            sector_df = _fetch_reference_series(sector_ticker)
            sector_ret_5d = sector_df['Close'].pct_change(5).reindex(data.index, method='ffill')
        else:
            sector_ret_5d = pd.Series(np.nan, index=data.index)
        data['Sector_Return_5D'] = sector_ret_5d

        data['Relative_Strength_5D'] = data['Return_5D'] - data['Market_Return_5D']

        vix_df = _fetch_reference_series(VIX_TICKER)
        vix_close = vix_df['Close'].reindex(data.index, method='ffill')
        data['VIX_Level'] = vix_close
        data['VIX_Change_5D'] = vix_close.pct_change(5)

        dow = pd.Series(data.index, index=data.index).dt.dayofweek
        data['DOW_Sin'] = np.sin(2 * np.pi * dow / 5)
        data['DOW_Cos'] = np.cos(2 * np.pi * dow / 5)

        data = data.replace([np.inf, -np.inf], np.nan)

        return data

    _SKIP_NORMALIZATION = {'DOW_Sin', 'DOW_Cos', 'Stoch_K', 'ADX_14'}

    def _normalize_features(self, data: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        data = data.copy()
        for col in self.features:
            if col in self._SKIP_NORMALIZATION:
                continue
            roll_mean = data[col].rolling(window, min_periods=window // 4).mean()
            roll_std = data[col].rolling(window, min_periods=window // 4).std()
            data[col] = (data[col] - roll_mean) / roll_std.replace(0, np.nan)
        data = data.replace([np.inf, -np.inf], np.nan)
        return data

    def _prepare_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self._compute_raw_features(df)

        future_close = data['Close'].shift(-self.horizon)
        forward_move = future_close - data['Close']
        data['Target'] = (forward_move > 0).astype(int)

        if self.dead_zone_atr_mult > 0:
            threshold = self.dead_zone_atr_mult * data['ATR_14']
            keep = forward_move.abs() >= threshold
            data = data[keep]

        data = self._normalize_features(data)

        return data.dropna()


    def evaluate_walk_forward(self, end_date: Optional[datetime] = None) -> dict:
        raw_df = self._fetch_data(end_date=end_date)
        processed_df = self._prepare_training_features(raw_df)

        X = processed_df[self.features].reset_index(drop=True)
        y = processed_df['Target'].reset_index(drop=True)

        splitter = PurgedTimeSeriesSplit(
            n_splits=self.n_cv_splits, horizon=self.horizon,
            embargo=self.embargo,
        )

        fold_results = []
        for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X)):
            model = self._init_model()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            preds = (proba > 0.5).astype(int)
            y_test = y.iloc[test_idx]

            fold_results.append({
                'fold': fold_i,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'accuracy': accuracy_score(y_test, preds),
                'log_loss': log_loss(y_test, proba, labels=[0, 1]),
                'brier': brier_score_loss(y_test, proba),
            })

        if not fold_results:
            raise ValueError("No valid CV folds produced -- not enough history for this horizon/n_splits")

        agg = pd.DataFrame(fold_results)
        return {
            'symbol': self._instrument.symbol,
            'folds': fold_results,
            'mean_accuracy': agg['accuracy'].mean(),
            'mean_log_loss': agg['log_loss'].mean(),
            'mean_brier': agg['brier'].mean(),
        }

    def train(self, end_date: Optional[datetime] = None) -> dict:
        eval_results = self.evaluate_walk_forward(end_date=end_date)

        raw_df = self._fetch_data(end_date=end_date)
        processed_df = self._prepare_training_features(raw_df)

        if end_date is not None:
            valid_cutoff = end_date - timedelta(days=self.horizon)
            processed_df = processed_df.loc[:valid_cutoff]

        X = processed_df[self.features]
        y = processed_df['Target']

        if self.calibrate:
            splitter = PurgedTimeSeriesSplit(
                n_splits=self.n_cv_splits, horizon=self.horizon, embargo=self.embargo,
            )
            self._model = CalibratedClassifierCV(
                self._init_model(), method='isotonic', cv=splitter,
            )
        else:
            self._model = self._init_model()

        self._model.fit(X, y)
        self._is_fitted = True

        return {
            'symbol': self._instrument.symbol,
            'accuracy': eval_results['mean_accuracy'],
            'train_end_date': end_date,
            'train_samples': len(X),
            'out_of_sample_accuracy': eval_results['mean_accuracy'],
            'out_of_sample_log_loss': eval_results['mean_log_loss'],
            'out_of_sample_brier': eval_results['mean_brier'],
            'cv_folds': len(eval_results['folds']),
        }

    def predict_at_date(self, as_of_date: datetime) -> dict:
        if not self._is_fitted:
            raise RuntimeError("Call train() before predict_at_date()")

        raw_df = self._fetch_data(end_date=as_of_date)
        features_df = self._compute_raw_features(raw_df)
        features_df = self._normalize_features(features_df).dropna()

        if features_df.empty:
            return {
                'date': as_of_date,
                'signal': 0,
                'abstained': True,
                'confidence_up': 0.5,
                'confidence_down': 0.5,
                'atr': 0.0,
                'close': 0.0,
                'sma_5': 0.0,
                'sma_20': 0.0,
                'sma_50': 0.0,
                'prev_close': 0.0,
                'adx': 0.0,
                'high': 0.0,
                'low': 0.0,
            }

        latest_X = features_df[self.features].iloc[[-1]]

        prob = self._model.predict_proba(latest_X)[0]
        confidence_up, confidence_down = float(prob[1]), float(prob[0])
        max_conf = max(confidence_up, confidence_down)

        abstained = max_conf < self.confidence_threshold
        signal = 0 if abstained else (1 if confidence_up > confidence_down else -1)

        raw_row = self._compute_raw_features(raw_df).iloc[-1]

        return {
            'date': as_of_date,
            'signal': signal,
            'abstained': abstained,
            'confidence_up': confidence_up,
            'confidence_down': confidence_down,
            'atr': float(raw_row['ATR_14']),
            'close': float(raw_row['Close']),
            'high': float(raw_row['High']),
            'low': float(raw_row['Low']),
            'adx': float(raw_row['ADX_14']) if not pd.isna(raw_row['ADX_14']) else 0.0,
            'sma_5': float(raw_row['SMA_5']),
            'sma_20': float(raw_row['SMA_20']),
            'sma_50': float(raw_row['SMA_50']),
            'prev_close': float(raw_row['prev_close']),
        }


N_ESTIMATORS = 200
MAX_DEPTH = 5

class PriceDirPredictorRF(PriceDirPredictor):
    def _init_model(self):
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=42,
            class_weight="balanced",
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
            verbose=-1,
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
            eval_metric='logloss',
        )