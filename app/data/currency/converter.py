import requests
import json

CURRENCY_RATES_URL = 'https://api.nbp.pl/api/exchangerates/tables/A/?format=json'

RATES_FIELD = 'rates'
CODE_FIELD = 'code'
RATE_FIELD = 'mid'

class InvalidCurrencyError(Exception):
    pass

class CurrencyConverter:
    def __init__(self, target_currency: str):
        self._currency_rates = self._fetch_currency_data()

        if target_currency not in self._currency_rates:
            raise InvalidCurrencyError(f'Provided target currency {target_currency} is not valid.')

        self.target_currency = target_currency

    def convert(self, value: float, currency: str) -> float:
        if currency not in self._currency_rates:
            raise InvalidCurrencyError(f'Provided currency {currency} is not valid.')

        return value * self._currency_rates[currency] / self._currency_rates[self.target_currency]
    
    def _fetch_currency_data(self) -> dict[str, float]:
        response = requests.get(CURRENCY_RATES_URL)

        if response.status_code == 200:
            raw_response = response.text
            response_json = json.loads(raw_response)[0]

            rates = response_json.get(RATES_FIELD)

            return self._parse_currencies(rates)

        return None
    
    def _parse_currencies(self, rates_data: list[dict]) -> dict[str, float]:
        currencies = {}
        for currency_data in rates_data:
            code = currency_data[CODE_FIELD]
            rate = currency_data[RATE_FIELD]

            currencies[code] = rate

        currencies['PLN'] = 1

        return currencies