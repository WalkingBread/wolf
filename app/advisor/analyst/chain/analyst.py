from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.advisor.analyst.chain.base import BaseChainWrapper
from app.advisor.analyst.model import AnalystDecision


class FinancialHealthAnalystChain(BaseChainWrapper):
    
    def _compile_chain(self) -> Runnable:
        system_template = (
            "You are an expert Financial Analyst specializing in fundamental analysis.\n"
            "Your task is to analyze the financial health of a company based on the provided corporate metrics.\n"
            "Assess metrics such as liquidity (quick ratio), profitability (profit margin, ROE), debt levels, "
            "and growth (revenue growth) to make a final decision."
        )
        
        human_template = (
            "Please analyze the following financial health metrics for a given company:\n\n"
            "{financial_health}\n\n"
            "Provide a definitive BUY or SELL recommendation with a detailed, point-by-point financial reasoning."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        return prompt | self.llm.with_structured_output(AnalystDecision)
    
class FinancialMetricsAnalystChain(BaseChainWrapper):

    def _compile_chain(self) -> Runnable:
        system_template = (
            "You are an expert Equity Research Analyst specializing in market valuation metrics.\n"
            "Your task is to determine whether a stock is overvalued, undervalued, or fairly priced.\n"
            "Analyze metrics such as Trailing P/E vs Forward P/E (to see earnings momentum), "
            "PEG Ratio (valuation relative to growth, where < 1.0 is often undervalued), "
            "Price-to-Book (P/B), Dividend Yield, and Beta (systematic risk).\n"
            "Contextualize your decision in terms of margin of safety and risk-return profile."
        )
        
        human_template = (
            "Please evaluate the following market valuation metrics for a given company:\n\n"
            "{financial_metrics}\n\n"
            "Provide a definitive BUY or SELL recommendation with a detailed, point-by-point financial reasoning."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        return prompt | self.llm.with_structured_output(AnalystDecision)
    
class GeneralAnalystChain(BaseChainWrapper):

    def _compile_chain(self) -> Runnable:
        system_template = (
            "You are the Chief Investment Officer (CIO) and Chairman of the Investment Committee.\n"
            "Your job is to review reports submitted by your specialist sub-analysts.\n"
            "Your role is to resolve conflicts. For example, if a stock has amazing financial health "
            "but terrible technical momentum, you must decide whether to sell (SELL) or buy the dip (BUY).\n"
            "Be conservative, manage risk."
        )
        
        human_template = (
            "Review the analyst submissions:\n\n"
            "{analyst_report}"
            "Deliver the final executive verdict based on the provided information."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        return prompt | self.llm.with_structured_output(AnalystDecision)