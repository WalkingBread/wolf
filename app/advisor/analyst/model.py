from pydantic import BaseModel, Field
from typing import Union, Literal, Annotated
from enum import Enum

class MarketAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class AnalystDecision(BaseModel):
    decision: MarketAction = Field(..., description="Should the instrument be acquired or disposed based on provided data?")
    reasoning: list[str] = Field(default_factory=list, description="Reasoning behind the decision delivered as multiple points.")