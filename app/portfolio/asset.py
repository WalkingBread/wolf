from datetime import datetime, timedelta

from app.data.instrument.instrument import Instrument

class Asset:
    def __init__(self, instrument: Instrument, volume: float, 
                 buy_price: float, purchase_date: datetime):
        self._instrument = instrument
        self.volume = volume
        self.buy_price = buy_price
        self.purchase_date = purchase_date

    @property
    def name(self) -> str:
        return self._instrument.full_name
    
    @property
    def symbol(self) -> str:
        return self._instrument.symbol
    
    @property
    def initial_value(self) -> float:
        return self.buy_price * self.volume
    
    def get_value_change(self, date: datetime = None) -> float:
        price_at_date = self._instrument.current_price
        if date:
            market_data = self._instrument.get_market_data_at_closest_trading_day(date)
            price_at_date = market_data['close']
        
        return (self.buy_price - price_at_date) * self.volume
    

class Portfolio:
    def __init__(self):
        self._assets = []

    def add(self, asset: Asset):
        self._assets.append(asset)

    