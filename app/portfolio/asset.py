from datetime import datetime

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
    
    @property
    def value(self) -> float:
        return self._instrument.current_price * self.volume
    
    def get_value_at_date(self, date: datetime) -> float:
        market_data = self._instrument.get_market_data_at_closest_trading_day(date)
        return market_data['close']
    
    def get_value_change(self, date: datetime = None) -> float:
        price_at_date = self._instrument.current_price
        if date:
            price_at_date = self.get_value_at_date(date)
        
        return (price_at_date - self.buy_price) * self.volume
    
    def get_percent_change(self, date: datetime = None) -> float:
        return (self.get_value_change(date) / self.initial_value) if self.initial_value != 0 else 0
    

class Portfolio:
    def __init__(self, name: str):
        self.name = name
        self._assets: list[Asset] = []

    @property
    def initial_value(self) -> float:
        return sum([a.initial_value for a in self._assets])

    @property
    def value(self) -> float:
        return sum([a.value for a in self._assets])

    def add(self, *assets: Asset) -> None:
        self._assets.extend(assets)

    def get_value_change(self, date: datetime = None) -> float:
        return sum([a.get_value_change(date) for a in self._assets])
    
    def get_percent_change(self, date: datetime = None) -> float:
        return (self.get_value_change(date) / self.initial_value) if self.initial_value != 0 else 0
    
    def __str__(self) -> str:
        today = datetime.now()
        
        total_change = self.get_value_change()
        total_percent = self.get_percent_change() * 100
        
        output = [
            "==================================================",
            f"💼 Portfolio {self.name} Summary ({today.strftime('%Y-%m-%d')})",
            "==================================================",
            f"Initial value:  {self.initial_value:,.2f}",
            f"Current value:  {self.value:,.2f}",
            f"Value change:   {total_change:+,.2f} ({total_percent:+.2f}%)",
            "==================================================",
            f"Assets:",
            "--------------------------------------------------"
        ]
        
        for asset in self._assets:
            asset_change = asset.get_value_change()
            asset_percent = (asset_change / asset.initial_value * 100) if asset.initial_value != 0 else 0.0
            
            asset_info = (
                f"• {asset.symbol:<8} | "
                f"Value: {asset.value:,.2f} | "
                f"Change: {asset_change:+,.2f} ({asset_percent:+.2f}%)"
            )
            output.append(asset_info)
            
        output.append("==================================================")
        
        return "\n".join(output)