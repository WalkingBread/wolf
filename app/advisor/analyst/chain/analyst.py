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
            "{financial_health_data}\n\n"
            "Provide a definitive BUY or SELL recommendation with a detailed, point-by-point financial reasoning."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        return prompt | self.llm.with_structured_output(AnalystDecision)