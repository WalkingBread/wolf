from app.data.instrument.instrument import Instrument
from app.data.instrument.symbols import INSTRUMENT_SYMBOLS

class InstrumentProvider:
    def __init__(self, instrument_symbols = INSTRUMENT_SYMBOLS):
        self._instruments = self._init_instruments(instrument_symbols)

    @property
    def instrument_symbols(self):
        return list(self._instruments.keys())
        
    def _init_instruments(self, instrument_symbols: list[str]) -> dict[Instrument]:
        instruments = {}
        for symbol in instrument_symbols:
            instruments[symbol] = Instrument(symbol)
        return instruments

    def get_instrument(self, symbol) -> Instrument:
        return self._instruments.get(symbol)
    