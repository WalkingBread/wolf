from dataclasses import dataclass
from datetime import datetime

from core.portfolio.asset import Portfolio

@dataclass
class EngineState:
    date: datetime
    portfolio: Portfolio
    cash: float

    @property
    def total_value(self):
        return self.portfolio.get_value_at_date(self.date) + self.cash

@dataclass
class StrategyResult:
    type: str
    amount: float