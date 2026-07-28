class CurrencyConverter:
    def __init__(self, target_currency: str):
        self.target_currency = target_currency

        self._currency_rates = {
            'PLN': 1,
            'USD': 3.60,
            'EUR': 4.20
        }

    def convert(self, value: float, currency: str) -> float:
        return value * self._currency_rates[currency] / self._currency_rates[self.target_currency]