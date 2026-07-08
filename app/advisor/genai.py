from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.settings import (
    OPEN_AI_API_KEY, 
    OPEN_AI_API_URL,
    AZURE_API_VERSION,
    VECTOR_DIMENSIONS
)

class ModelProvider(ABC):
    def __init__(self):
        self._llm: BaseChatModel = self._init_llm()
        self._embeddings: Embeddings = self._init_embeddings()

    @abstractmethod
    def _init_llm(self) -> BaseChatModel:
        pass

    @abstractmethod
    def _init_embeddings(self) -> Embeddings:
        pass

    @property
    def llm_name(self):
        return getattr(self._llm, "model_name", getattr(self._llm, "model", "Unknown LLM"))
    
    @property
    def embedding_model_name(self):
        return getattr(self._embeddings, "model", "Unknown Embeddings")

    @property
    def llm(self):
        return self._llm
    
    @property
    def embeddings(self):
        return self._embeddings
    
    
LANGUAGE_MODEL = 'gpt-5'
EMBEDDING_MODEL = 'text-embedding-3-small'

class AzureModelProvider(ModelProvider):
    def __init__(self):
        super().__init__()

    def _init_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            model=LANGUAGE_MODEL,
            openai_api_key=OPEN_AI_API_KEY, 
            azure_endpoint=f'{OPEN_AI_API_URL}/{LANGUAGE_MODEL}',
            api_version=AZURE_API_VERSION
        )
    
    def _init_embeddings(self) -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPEN_AI_API_KEY,
            azure_endpoint=f'{OPEN_AI_API_URL}/{EMBEDDING_MODEL}',
            dimensions=VECTOR_DIMENSIONS,
            api_version=AZURE_API_VERSION
        )