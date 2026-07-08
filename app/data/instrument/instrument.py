from yfinance import Ticker
from newspaper import Article

from app.logger.config import get_logger

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = get_logger(to_file=False)

class InstrumentDataFetchError(Exception):
    pass

class Instrument:
    def __init__(self, symbol):
        self._ticker = Ticker(symbol)
        self._info = None
    
    @property
    def info(self):
        if not self._info:
            try: 
                self._info = self._ticker.info
            except Exception as e:
                logger.warning(f'Failed to fetch instrument ({self.symbol}) data: {str(e)}')
                raise InstrumentDataFetchError(f'Error while fetching instrument data: {str(e)}')

        return self._info
    
    @property
    def symbol(self):
        return self.info.get('symbol')
    
    @property
    def full_name(self):
        return self.info.get('longName')
    
    @property
    def current_price(self):
        return self.info.get('currentPrice')

    def get_news(self, max_workers: int = 3) -> list[dict]:
        try:
            news = self._ticker.news
        except Exception as e:
            logger.warning(f'Failed to fetch news for instrument ({self.symbol}): {str(e)}')
            return []
        
        if not news:
            return []
        
        articles = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_article = {
                executor.submit(self._parse_article, item): item 
                for item in news
            }
            for future in as_completed(future_to_article):
                try:
                    parsed_article = future.result()
                    articles.append(parsed_article)
                except Exception as e:
                    logger.warning(f"Worker failed processing an article for {self.symbol}: {e}")

        articles.sort(key=lambda x: x.get('date') or '', reverse=True)
        
        return articles
    
    def _parse_article(self, article_data) -> dict:
        metadata = article_data.get('content', {})

        url = metadata.get('clickThroughUrl', {}).get('url')

        if url:
            article = Article(url)
            article.download()
            article.parse()

        return {
            "title": metadata.get('title'),
            "date": metadata.get('pubDate'),
            "source": metadata.get('provider', {}).get('displayName'),
            "content": article.text
        }
    
    def get_basic_info(self) -> dict:
        info = self.info
        return {
            "short_name": info.get("shortName"),
            "long_name": info.get("longName"),
            "currency": info.get("currency"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "summary": info.get("longBusinessSummary")
        }
    
    def get_current_market_data(self) -> dict:
        info = self.info
        return {
            "current_price": info.get("currentPrice"),
            "previous_close": info.get("previousClose"),
            "open": info.get("open"),
            "day_low": info.get("dayLow"),
            "day_high": info.get("dayHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "volume": info.get("volume"),
            "average_volume": info.get("averageVolume"),
            "market_cap": info.get("marketCap")
        }
    
    def get_historical_market_data(self, start: datetime, end: datetime, interval = '1d'): 
        try:
            return self._ticker.history(
                start=start.strftime("%Y-%m-%d"), 
                end=end.strftime("%Y-%m-%d"), 
                interval=interval
            )
        except Exception as e:
            raise InstrumentDataFetchError(f'Error while fetching instrument data: {str(e)}')
        
    def get_market_data_at_closest_trading_day(self, date: datetime, time_window_days: int = 5) -> dict:
        window_end_date = date + timedelta(days=time_window_days)
        market_data_df = self.get_historical_market_data(date, window_end_date)

        if market_data_df.empty:
            return {}
        
        closest_day = market_data_df.iloc[0]
        return {
            "trading_date": closest_day.index[0].strftime("%Y-%m-%d"),
            "open": round(float(closest_day["Open"]), 2),
            "high": round(float(closest_day["High"]), 2),
            "low": round(float(closest_day["Low"]), 2),
            "close": round(float(closest_day["Close"]), 2),
            "volume": int(closest_day["Volume"])
        }
        
    
    def get_financial_metrics(self) -> dict:
        info = self.info
        return {
            "trailing_pe": info.get("trailingPE"),       # Price-to-Earnings (Past year)
            "forward_pe": info.get("forwardPE"),         # Price-to-Earnings (Next year forecast)
            "peg_ratio": info.get("pegRatio"),           # Price/Earnings-to-Growth
            "price_to_book": info.get("priceToBook"),     # Price-to-Book
            "dividend_yield": info.get("dividendYield"),  # e.g., 0.015 = 1.5%
            "beta": info.get("beta")                      # Volatility relative to the market (>1 is more volatile)
        }
    
    def get_financial_health(self) -> dict:
        info = self.info
        return {
            "total_revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"), # YoY Revenue Growth
            "ebitda": info.get("ebitda"),
            "profit_margin": info.get("profitMargins"),
            "total_debt": info.get("totalDebt"),
            "quick_ratio": info.get("quickRatio"),       # Short-term liquidity
            "return_on_equity": info.get("returnOnEquity")
        }

    def get_financial_statements(self) -> dict:        
        statements_map = {
            "yearly_income_statement": "income_stmt",
            "yearly_balance_sheet": "balance_sheet",
            "yearly_cashflow": "cashflow"
        }
        
        result = {}
        for key, attribute_name in statements_map.items():
            try:
                statement_data = getattr(self._ticker, attribute_name)
                result[key] = statement_data.to_dict() if not statement_data.empty else {}
            except Exception as e:
                logger.warning(f'Failed to fetch {key} for instrument {self.symbol}: {str(e)}')
                result[key] = {}
                
        return result