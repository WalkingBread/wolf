from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from pydantic import BaseModel, Field
from typing import Union, Literal, Annotated
from enum import Enum

from app.advisor.analyst.base import InstrumentAnalyst

class MarketAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class AnalystDecision(BaseModel):
    decision: MarketAction = Field(..., description="Should the instrument be acquired or disposed based on provided data?")
    reasoning: list[str] = Field(default_factory=list, description="Reasoning behind the decision delivered as multiple points.")
    

class FinancialHealthAnalyst(InstrumentAnalyst):
    
    def _compile_chain(self) -> Runnable:
        system_template = (
            "You are an expert Financial Analyst specializing in fundamental analysis.\n"
            "Your task is to analyze the financial health of a company based on the provided corporate metrics.\n"
            "Assess metrics such as liquidity (quick ratio), profitability (profit margin, ROE), debt levels, "
            "and growth (revenue growth) to make a final decision."
        )
        
        human_template = (
            "Please analyze the following financial health metrics for a given company:\n\n"
            "{financial_health_data}\n\n"
            "Provide a definitive BUY or SELL recommendation with a detailed, point-by-point financial reasoning."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        return prompt | self.llm.with_structured_output(AnalystDecision)