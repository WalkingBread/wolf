from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.data.instrument.instrument import Instrument
from core.portfolio.asset import Portfolio, Asset  


from app.persistence.models import Base, PortfolioModel, AssetModel

class PortfolioRepository:
    def __init__(self, db_path: str = "sqlite:///./portfolios.db"):
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(bind=self.engine)

        self.session_local = sessionmaker(bind=self.engine)

    def save(self, portfolio: Portfolio) -> None:
        with self.session_local() as session:
            db_portfolio = session.query(PortfolioModel).filter_by(name=portfolio.name).first()
            
            if not db_portfolio:
                db_portfolio = PortfolioModel(
                    name=portfolio.name, 
                    currency=portfolio.currency
                )
                session.add(db_portfolio)
                session.flush()

            db_portfolio.assets.clear()
            for asset in portfolio._assets:
                db_asset = AssetModel(
                    symbol=asset.symbol,
                    volume=asset.volume,
                    buy_price=asset.buy_price,
                    currency=asset.currency,
                    purchase_date=asset.purchase_date
                )
                db_portfolio.assets.append(db_asset)

            session.commit()

    def get_by_name(self, name: str, instrument_provider) -> Portfolio | None:
        with self.session_local() as session:
            db_portfolio = session.query(PortfolioModel).filter_by(name=name).first()
            if not db_portfolio:
                return None

            portfolio = Portfolio(
                name=db_portfolio.name, 
                currency=db_portfolio.currency
            )
            
            for db_asset in db_portfolio.assets:
                instrument: Instrument = instrument_provider.get_instrument(db_asset.symbol)
                
                asset = Asset(
                    instrument=instrument,
                    volume=db_asset.volume,
                    buy_price=db_asset.buy_price,
                    purchase_date=db_asset.purchase_date
                )
                portfolio.add(asset)

            return portfolio
        
    def get_all_portfolio_names(self) -> list[str]:
        with self.session_local() as session:
            return session.scalars(session.query(PortfolioModel.name)).all()