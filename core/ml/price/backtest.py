from core.portfolio.asset import Asset, Portfolio
from core.data.instrument.instrument import Instrument
from core.ml.price.dto import StrategyResult, EngineState
from core.ml.price.strategy import ModelStrategy

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

import pandas as pd

@dataclass
class Transaction:
    symbol: str
    type: str
    volume: float
    price: float
    total_amount: float
    fee: float

class BacktestEngine:
    def __init__(self, portfolio: Portfolio, start_date: datetime, 
                 strategy: ModelStrategy, initial_cash: float = 1000.0):
        self._portfolio = portfolio
        self._start_date = start_date
        self._strategy = strategy
        self._initial_cash = initial_cash
        self._cash = initial_cash

        self.date = start_date
        self.transaction_log: list[Transaction] = []

    @property
    def cash(self):
        return self._cash
    
    @property
    def assets_values(self):
        return self._portfolio.get_value_at_date(self.date)
    
    @property
    def total_value(self):
        return self.assets_values + self.cash
    
    def next_day(self):
        engine_state = EngineState(self.date, self._portfolio, self._cash)

        if self._strategy is not None:
            result: Optional[StrategyResult] = self._strategy.apply(engine_state)
            
            if result is not None:
                if result.type == 'BUY':
                    self.buy(self._strategy.instrument, result.amount)
                elif result.type == 'SELL':
                    self.sell(self._strategy.instrument.symbol, result.amount)

        if self.date < datetime.now():
            self.date += timedelta(days=1)
            return False

        return True


    def buy(self, instrument: Instrument, amount: float, fee: float = 0.0):
        if self._cash < amount:
            raise ValueError(f'Insufficient cash to acquire the instrument for amount {amount}.')

        market_data = instrument.get_market_data_at_closest_trading_day(self.date)
        price = self._portfolio.convert_to_native_currency(market_data['close'], instrument.currency)

        volume = (amount - fee) / price if price != 0 else 0

        asset = Asset(instrument, volume, price, self.date)

        self._portfolio.add(asset)

        self._cash -= amount

        self.transaction_log.append(
            Transaction(instrument.symbol, 'BUY', volume, price, amount, fee)
        )

    def sell(self, symbol: str, amount: float, fee: float = 0.0):
        asset: Asset = self._portfolio.get_asset(symbol)

        if not asset:
            raise ValueError(f"Cannot sell {symbol}: position does not exist in portfolio.")


        market_data = asset.instrument.get_market_data_at_closest_trading_day(self.date)
        price = self._portfolio.convert_to_native_currency(market_data['close'], asset.currency)

        volume_to_sell = amount / price if price != 0 else 0
        if volume_to_sell > asset.volume:
            volume_to_sell = asset.volume

        self._portfolio.reduce_asset_exposure(symbol, volume_to_sell)

        amount = volume_to_sell * price
        self._cash += amount - fee

        self.transaction_log.append(
            Transaction(symbol, 'SELL', volume_to_sell, price, amount, fee)
        )

    def show_trasaction_log(self):
        if not self.transaction_log:
            return "No transactions recorded."

        return '\n'.join(
            f"Transaction: {t.type} {t.volume:.4f} x {t.symbol} @ ${t.price:.2f} "
            f"| Total: ${t.total_amount:.2f} (Fee: ${t.fee:.2f})"
            for t in self.transaction_log
        )
    
    def show_transaction_log_df(self) -> pd.DataFrame:
        if not self.transaction_log:
            print("No transactions recorded.")
            return pd.DataFrame()
        
        data = [
            {
                "Type": t.type,
                "Symbol": t.symbol,
                "Volume": t.volume,
                "Price ($)": f"{t.price:,.2f}",
                "Total Amount ($)": f"{t.total_amount:,.2f}",
                "Fee ($)": f"{t.fee:,.2f}"
            }
            for t in self.transaction_log
        ]
        
        df = pd.DataFrame(data)
        return df.style.set_properties(**{'text-align': 'center'}).hide(axis="index")