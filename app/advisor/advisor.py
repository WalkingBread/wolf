from app.advisor.genai import ModelProvider
from app.advisor.analyst.model import AnalystDecision
from app.advisor.analyst.autonomous import (
    FinancialHealthAnalyst, 
    FinancialMetricsAnalyst,
    StatementTrendAnalyst,
    NewsSentimentAnalyst,
    TechnicalAnalyst,
    GeneralAnalyst
)
from app.data.instrument.instrument import Instrument

from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

class AnalysisTypeNotSupportedError(Exception):
    pass

class AnalysisType(Enum):
    def _generate_next_value_(name, _start, _count, _last_values):
        return name.upper()
    
    FINANCIAL_HEALTH = auto()
    FINANCIAL_METRICS = auto()
    STATEMENT_TREND = auto()
    NEWS_SENTIMENT = auto()
    TECHNICAL = auto()

@dataclass
class AdvisorReport:
    final_decision: dict = field(default_factory=dict)
    sub_analysis_results: dict = field(default_factory=dict)
    analysis_cost_usd: float = field(default=0)

class InvestingAdvisor:
    def __init__(self, model_provider: ModelProvider):
        self._model_provider = model_provider
        
        self._analysts = {
            AnalysisType.FINANCIAL_HEALTH: FinancialHealthAnalyst(self._model_provider.llm),
            AnalysisType.FINANCIAL_METRICS: FinancialMetricsAnalyst(self._model_provider.llm),
            AnalysisType.STATEMENT_TREND: StatementTrendAnalyst(self._model_provider.llm),
            AnalysisType.TECHNICAL: TechnicalAnalyst(self._model_provider.llm),
            AnalysisType.NEWS_SENTIMENT: NewsSentimentAnalyst(self._model_provider.llm)
        }
        self._general_analyst = GeneralAnalyst(self._model_provider.llm)

    def analyze_instrument(self, instrument: Instrument) -> tuple[dict, dict]:
        analysis_result = {}
        total_cost = 0

        def _run_analyst(analyst):
            decision: AnalystDecision = analyst.analyze(instrument)
            usage = analyst.last_usage_metadata
            return analyst.__class__.__name__, decision.model_dump(mode='json'), usage

        with ThreadPoolExecutor(max_workers=len(self._analysts)) as executor:
            futures = [executor.submit(_run_analyst, analyst) for analyst in self._analysts.values()]
            for future in as_completed(futures):
                analyst_name, result, usage = future.result()
                analysis_result[analyst_name] = result

                total_cost += usage["total_cost_usd"]

        final_decision = self._general_analyst.analyze(analysis_result)

        return AdvisorReport(
            final_decision=final_decision.model_dump(mode='json'),
            sub_analysis_results=analysis_result,
            analysis_cost_usd=total_cost
        )

    def analyze_instrument_component(self, analysis_type: AnalysisType, instrument: Instrument) -> dict:
        analyst = self._analysts.get(analysis_type)
        if not analyst:
            raise AnalysisTypeNotSupportedError(
                f"Unsupported analysis type: '{analysis_type}'. "
                f"No corresponding analyst instance registered."
            )
        analysis_result = analyst.analyze(instrument)    
        return analysis_result.model_dump(mode='json')