from core.ml.price.model import PriceDirPredictor

from core.ml.price.dto import StrategyResult, EngineState

from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass

class ModelStrategy(ABC):
    def __init__(self, model: PriceDirPredictor):
        self._model = model
        self._days_since_retrain = model.horizon

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
HIGH_CONF_PORTFOLIO_CAP = 0.80
STOP_LOSS_ATR_MULT = 2.5

RISK_PER_TRADE_PCT = 0.02
MAX_PORTFOLIO_CAP_PCT = 0.40
HIGH_CONF_PORTFOLIO_CAP = 0.80
HIGH_CONF_THRESHOLD = 0.75          # confidence_up above this unlocks the bigger cap
ENTRY_CONFIDENCE_THRESHOLD = 0.55
EXIT_ON_REVERSAL_CONFIDENCE = 0.55  # confidence_down needed to exit on a signal flip
 
STOP_LOSS_ATR_MULT = 3.0            # single source of truth for stop distance --
TRAILING_STOP_ATR_MULT = 3.0        # used consistently for sizing AND placement
TAKE_PROFIT_ATR_MULT = 4.0
 
MIN_TRADE_VALUE = 50.0

@dataclass
class TrackedPosition:
    entry_date: datetime
    entry_price: float
    highest_price: float
    stop_loss_price: float
    take_profit_price: float

    def get_days_held(self, date: datetime):
        return date - self.entry_date


class ATRStopLossStrategy(ModelStrategy):

    def __init__(self, model):
        super().__init__(model)

        self._tracked_position = None


    def _strategy(self, engine_state: EngineState, forecast: dict):
        signal = forecast['signal']
        confidence_up = forecast.get('confidence_up', 0.0)
        atr = forecast['atr']
        close_price = forecast['close']

        asset = engine_state.portfolio.get_asset(self.instrument_symbol)

        if asset is not None:
            highest_price = max(self._tracked_position.highest_price, close_price)
            self._tracked_position.highest_price = highest_price

            if atr > 0:
                trailing_stop = highest_price - (TRAILING_STOP_ATR_MULT * atr)
                if trailing_stop > self._tracked_position.stop_loss_price:
                    self._tracked_position.stop_loss_price = trailing_stop

            stop_price = self._tracked_position.stop_loss_price
            take_profit = self._tracked_position.take_profit_price

            if close_price <= stop_price or close_price >= take_profit:
                self._tracked_position = None
                return StrategyResult('SELL', asset.volume * close_price)

        elif asset is None and signal == 1 and confidence_up >= 0.62:
            if atr > 0 and close_price > 0:
                total_portfolio_value = engine_state.total_value

                risk_budget = total_portfolio_value * RISK_PER_TRADE_PCT
                atr_stop_distance = STOP_LOSS_ATR_MULT * atr
                
                target_shares = risk_budget / atr_stop_distance
                calculated_amount = target_shares * close_price
    
                max_cash_allowed = engine_state.cash * MAX_PORTFOLIO_CAP_PCT
                final_allocation = min(calculated_amount, max_cash_allowed)
    
                if final_allocation > 50:
                    self._tracked_position = TrackedPosition(
                        engine_state.date,
                        close_price,
                        close_price,
                        close_price - (STOP_LOSS_ATR_MULT * atr),
                        close_price + (TAKE_PROFIT_ATR_MULT * atr)
                    )
                    
                    return StrategyResult('BUY', final_allocation)

        return None