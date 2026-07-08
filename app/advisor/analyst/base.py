from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from app.logger.config import get_logger

logger = get_logger(stdout=False)

class InstrumentAnalyst(ABC):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain

    @abstractmethod
    def _compile_chain(self) -> Runnable:
        pass

    def _log_prompt_interceptor(self, prompt_value) -> any:
        compiled_text = prompt_value.to_string()
        
        chain_name = self.__class__.__name__
        logger.info(f'{chain_name} prompt content:\n{compiled_text}')
        
        return prompt_value

    def _build_chain(self) -> Runnable:
        core_chain = self._compile_chain()
        
        prompt_step = core_chain.steps[0]
        remaining_steps = core_chain.steps[1:]
        
        pipeline = prompt_step | RunnableLambda(self._log_prompt_interceptor)
        for step in remaining_steps:
            pipeline = pipeline | step
            
        return pipeline

    def invoke(self, context: dict) -> any:
        return self.chain.invoke(context)