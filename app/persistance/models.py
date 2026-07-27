from datetime import datetime

from sqlalchemy import ForeignKey, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class PortfolioModel(Base):
    __tablename__ = 'portfolios'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    assets: Mapped[list["AssetModel"]] = relationship(
        back_populates="portfolio", 
        cascade="all, delete-orphan"
    )


class AssetModel(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    volume: Mapped[float] = mapped_column(Float)
    buy_price: Mapped[float] = mapped_column(Float)
    purchase_date: Mapped[datetime] = mapped_column(DateTime)

    portfolio: Mapped["PortfolioModel"] = relationship(back_populates="assets")