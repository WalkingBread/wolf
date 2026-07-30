from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel

from app.data.instrument.instrument import Instrument
from app.advisor.analyst.model import AnalystDecision
from app.advisor.analyst.chain.base import BaseChainWrapper

import app.advisor.analyst.chain.analyst as chain_module

class AnalystChainNotFoundError(Exception):
    pass

class Analyst(ABC):
    def __init__(self, llm: BaseChatModel):
        current_class_name = self.__class__.__name__
        target_chain_name = f"{current_class_name}Chain"
        chain_class = getattr(chain_module, target_chain_name, None)
        
        if chain_class is None:
            raise AnalystChainNotFoundError(
                f"❌ Couldn't find chain named '{target_chain_name}' "
                f"in module '{chain_module.__name__}' for '{current_class_name}'."
            )
        self._chain: BaseChainWrapper = chain_class(llm)

    @property
    def last_usage_metadata(self):
        return self._chain.last_usage_metadata


class ComponentAnalyst(Analyst):

    @abstractmethod
    def _prepare_context(self, instrument: Instrument) -> dict:
        pass

    def analyze(self, instrument: Instrument) -> AnalystDecision:
        ctx = self._prepare_context(instrument)
        return self._chain.invoke(ctx)


class FinancialHealthAnalyst(ComponentAnalyst):
    
    def _prepare_context(self, instrument: Instrument) -> dict:
        financial_health = instrument.get_financial_health()

        formatted_context = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" 
                                       for key, value in financial_health.items()])
    
        return {
            "financial_health": formatted_context
        }
    
class FinancialMetricsAnalyst(ComponentAnalyst):

    def _prepare_context(self, instrument: Instrument):
        financial_metrics = instrument.get_financial_metrics()

        formatted_context = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" 
                                       for key, value in financial_metrics.items()])
        
        return {
            "financial_metrics": formatted_context
        }
    
class TechnicalAnalyst(ComponentAnalyst):
    def _prepare_context(self, instrument: Instrument) -> dict:
        market_data = instrument.get_current_market_data()
        
        curr = market_data.get("current_price") or 0.0
        low_52 = market_data.get("fifty_two_week_low") or 0.0
        high_52 = market_data.get("fifty_two_week_high") or 0.0
        volume = market_data.get("volume") or 0
        avg_volume = market_data.get("average_volume") or 1
        
        range_position = (
            ((curr - low_52) / (high_52 - low_52)) * 100 
            if (high_52 - low_52) > 0 else 50.0
        )
        volume_ratio = volume / avg_volume if avg_volume else 1.0

        return {
            "technical_context": (
                f"- Current Price: {curr}\n"
                f"- 52-Week Range: {low_52} - {high_52}\n"
                f"- 52-Week Position: {range_position:.1f}% (0% = at 52-week low, 100% = at 52-week high)\n"
                f"- Volume Relative to Avg: {volume_ratio:.2f}x average daily volume"
            )
        }
    
class StatementTrendAnalyst(ComponentAnalyst):
    def _prepare_context(self, instrument: Instrument) -> dict:
        statements = instrument.get_financial_statements()
        income_stmt = statements.get("yearly_income_statement", {})
        cashflow = statements.get("yearly_cashflow", {})
        
        summary_lines = []
        for line_item in ["Total Revenue", "Operating Income", "Net Income"]:
            if line_item in income_stmt:
                years_data = income_stmt[line_item]
                formatted_years = ", ".join([f"{k}: {v:,.0f}" for k, v in years_data.items() if v])
                summary_lines.append(f"- {line_item}: {formatted_years}")
                
        if "Free Cash Flow" in cashflow:
            fcf_data = cashflow["Free Cash Flow"]
            formatted_fcf = ", ".join([f"{k}: {v:,.0f}" for k, v in fcf_data.items() if v])
            summary_lines.append(f"- Free Cash Flow: {formatted_fcf}")

        return {
            "statement_trends": "\n".join(summary_lines) if summary_lines else "No multi-year statements found."
        }
    
class NewsSentimentAnalyst(ComponentAnalyst):
    def _prepare_context(self, instrument: Instrument) -> dict:
        news_items = instrument.get_news(max_workers=3)
        
        if not news_items:
            return {"news_context": "No recent news coverage available for this instrument."}
            
        formatted_articles = []
        for article in news_items[:5]:
            title = article.get("title", "No Title")
            source = article.get("source", "Unknown Source")
            content = article.get("content", "")[:300]
            formatted_articles.append(f"- [{source}] {title}\n  Summary: {content}...")
            
        return {
            "news_context": "\n\n".join(formatted_articles)
        }
    
class GeneralAnalyst(Analyst):
    
    def analyze(self, joint_analyst_report: dict) -> AnalystDecision:
        report_blocks = []
        for analyst, report in joint_analyst_report.items():
            report_block = (
                f"=== ANALYST: {analyst} ===\n"
                f"Recommendation: {report['decision']}\n"
                f"Key Arguments:\n" + "\n".join([f" - {point}" for point in report['reasoning']])
            )
            report_blocks.append(report_block)
        
        aggregated_reports = "\n\n".join(report_blocks)

        return self._chain.invoke({
            "analyst_report": aggregated_reports
        })