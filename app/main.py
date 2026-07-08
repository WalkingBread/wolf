from app.data.instrument.provider import InstrumentProvider
from app.data.instrument.instrument import Instrument
from app.portfolio.asset import Asset, Portfolio
from app.advisor.advisor import InvestingAdvisor
from app.advisor.genai import AzureModelProvider
from datetime import datetime

provider = InstrumentProvider()
instrument: Instrument = provider.get_instrument('MSFT')

ia = InvestingAdvisor(AzureModelProvider())

print(ia.analyze_instrument(instrument))

#p = Portfolio('p1')

#asset = Asset(instrument, 10, 406.74, datetime(2025, 10, 11))
#p.add(asset)

#print(p)