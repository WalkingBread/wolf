from app.data.instrument.provider import InstrumentProvider
from app.data.instrument.instrument import Instrument
from app.portfolio.asset import Asset, Portfolio

provider = InstrumentProvider()
instrument: Instrument = provider.get_instrument('CDP')

p = Portfolio()
