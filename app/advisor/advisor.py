from app.advisor.genai import ModelProvider
from app.advisor.analyst.autonomous import AnalystDecision, FinancialHealthAnalyst

from app.data.instrument.instrument import Instrument

from enum import Enum, auto

class AnalystType(Enum):
    FINANCIAL_HEALTH = auto()

class InvestingAdvisor:
    def __init__(self, model_provider: ModelProvider):
        self._model_provider = model_provider
        
        self._analysts = {
            AnalystType.FINANCIAL_HEALTH: FinancialHealthAnalyst(self._model_provider.llm)
        }

    def analyze_instrument(self, instrument: Instrument)

    def analyze_financial_health(self, financial_health_data: dict) -> str:
        formatted_context = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" 
                                       for key, value in financial_health_data.items()])
    
        decision: AnalystDecision = self._analysts[AnalystType.FINANCIAL_HEALTH].invoke({
            "financial_health_data": formatted_context
        })

        return decision.model_dump(mode='json')