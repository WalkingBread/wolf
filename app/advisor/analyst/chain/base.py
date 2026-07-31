from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_community.callbacks import get_openai_callback

from core.tools.logger import get_logger

logger = get_logger(logger_name='LangChain', stdout=False)

class BaseChainWrapper(ABC):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._chain = None
        self.last_usage_metadata = {}

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
        with get_openai_callback() as cb:
            result = self.chain.invoke(context)

            self.last_usage_metadata = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost_usd": cb.total_cost
            }

            chain_name = self.__class__.__name__
            logger.info(
                f"[{chain_name}] Tokens: {cb.total_tokens} "
                f"(Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens}) | "
                f"Cost: ${cb.total_cost:.6f}"
            )

            return result