from dataclasses import dataclass
from datetime import datetime

from core.portfolio.asset import Portfolio

@dataclass
class EngineState:
    date: datetime
    portfolio: Portfolio
    cash: float

@dataclass
class StrategyResult:
    type: str
    amount: float