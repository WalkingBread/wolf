from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel

from app.data.instrument.instrument import Instrument
from app.advisor.analyst.model import AnalystDecision
from app.advisor.analyst.chain.base import BaseChainWrapper

import app.advisor.analyst.chain.analyst as chain_module

class AnalystChainNotFoundError(Exception):
    pass

class ComponentAnalyst(ABC):
    def __init__(self, llm: BaseChatModel):
        current_class_name = self.__class__.__name__
        target_chain_name = f"{current_class_name}Chain"
        chain_class = getattr(chain_module, target_chain_name, None)
        
        if chain_class is None:
            raise AnalystChainNotFoundError(
                f"❌ Couldn't fint chain named '{target_chain_name}' "
                f"in module '{chain_module.__name__}' for '{current_class_name}'."
            )
        self._chain: BaseChainWrapper = chain_class(llm)

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
            "financial_health_data": formatted_context
        }