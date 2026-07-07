from app.advisor.genai import ModelProvider

class InvestingAdvisor:
    def __init__(self, model_provider: ModelProvider):
        self._model_provider = model_provider

    