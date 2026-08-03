from core.ml.price.model import PriceDirPredictor

from core.ml.price.dto import StrategyResult, EngineState

from abc import ABC, abstractmethod
from datetime import datetime

class ModelStrategy(ABC):
    def __init__(self, model: PriceDirPredictor):
        self._model = model
        self._days_since_retrain = model.horizon

        self._position_tracker = {
            'entry_price': None,
            'stop_loss_price': None
        }

    @property
    def instrument(self):
        return self._model.instrument

    @property
    def instrument_symbol(self):
        return self._model.instrument.symbol

    @property
    def should_retrain(self) -> bool:
        return self._days_since_retrain >= self._model.horizon

    def _retrain(self, date: datetime):
        train_stats = self._model.train(end_date=date)
        self._days_since_retrain = 0

        return train_stats

    def apply(self, engine_state: EngineState) -> StrategyResult:
        if self.should_retrain:
            _ = self._retrain(engine_state.date)

        forecast = self._model.predict_at_date(as_of_date=engine_state.date)

        self._days_since_retrain += 1

        return self._strategy(engine_state, forecast)


    @abstractmethod
    def _strategy(self, engine_state: EngineState, forecast: dict):
        pass


RISK_PER_TRADE_PCT = 0.02
MAX_PORTFOLIO_CAP_PCT = 0.40 
STOP_LOSS_ATR_MULT = 2.0

class ATRStopLossStrategy(ModelStrategy):

    def _strategy(self, engine_state: EngineState, forecast: dict):
        signal = forecast['signal']
        confidence = forecast['confidence_up']
        atr = forecast['atr']
        close_price = forecast['close']

        asset = engine_state.portfolio.get_asset(self.instrument_symbol)
        if asset is not None and self._position_tracker['stop_loss_price'] is not None:
            stop_price = self._position_tracker['stop_loss_price']

            if close_price <= stop_price:
                self._position_tracker['entry_price'] = None
                self._position_tracker['stop_loss_price'] = None

                return StrategyResult('SELL', asset.volume * close_price)

        if signal == 1 and confidence > 0.55:
            if atr > 0 and close_price > 0:
                total_portfolio_value = engine_state.portfolio.get_value_at_date(engine_state.date) + engine_state.cash
                risk_budget = total_portfolio_value * RISK_PER_TRADE_PCT
                atr_stop_distance = STOP_LOSS_ATR_MULT * atr
                
                target_shares = risk_budget / atr_stop_distance
                calculated_amount = target_shares * close_price
    
                max_cash_allowed = engine_state.cash * MAX_PORTFOLIO_CAP_PCT
                final_allocation = min(calculated_amount, max_cash_allowed)
    
                if final_allocation > 50:
                    self._position_tracker['entry_price'] = close_price
                    self._position_tracker['stop_loss_price'] = close_price - atr_stop_distance
                    
                    return StrategyResult('BUY', final_allocation)
    
        elif signal == 0 and asset is not None:
            self._position_tracker['entry_price'] = None
            self._position_tracker['stop_loss_price'] = None

            return StrategyResult('SELL', asset.volume * close_price)

        return None