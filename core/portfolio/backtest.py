from core.portfolio.asset import Asset, Portfolio
from core.data.instrument.instrument import Instrument

from datetime import datetime, timedelta

from typing import Callable

class BacktestEngine:
    def __init__(self, portfolio: Portfolio, start_date: datetime,
                 initial_cash: float = 1000.0):
        self._portfolio = portfolio
        self._start_date = start_date
        self._initial_cash = initial_cash
        self._cash = initial_cash

        self.date = start_date
        self.transaction_log = []

    @property
    def cash(self):
        return self._cash
    
    @property
    def assets_values(self):
        return self._portfolio.get_value_at_date(self.date)
    
    @property
    def total_value(self):
        return self.assets_values + self.cash
    
    def next_day(self, strategy: Callable = None):
        if strategy:
            strategy()

        if self.date < datetime.now():
            self.date += timedelta(days=1)


    def buy(self, instrument: Instrument, amount: float, fee: float = 0.0):
        if self._cash < amount:
            raise ValueError(f'Insufficient cash to acquire the instrument for amount {amount}.')

        market_data = instrument.get_market_data_at_closest_trading_day(self.date)
        price = market_data['close']

        volume = (amount - fee) / price

        asset = Asset(instrument, volume, price, self.date)

        self._portfolio.add(asset)

        self._cash -= amount

    def sell(self, symbol: str, amount: float, fee: float = 0.0):
        asset: Asset = self._portfolio.get_asset(symbol)
        if not asset:
            raise ValueError(f"Cannot sell {symbol}: position does not exist in portfolio.")

        market_data = asset.instrument.get_market_data_at_closest_trading_day(self.date)
        price = market_data['close']

        volume_to_sell = amount / price
        if volume_to_sell > asset.volume:
            volume_to_sell = asset.volume

        self._portfolio.reduce_asset_exposure(symbol, volume_to_sell)

        self._cash += volume_to_sell * price - fee