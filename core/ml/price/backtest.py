from core.portfolio.asset import Asset, Portfolio
from core.data.instrument.instrument import Instrument
from core.ml.price.dto import StrategyResult, EngineState
from core.ml.price.strategy import ModelStrategy

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

@dataclass
class Transaction:
    symbol: str
    date: datetime
    type: str
    volume: float
    price: float
    total_amount: float
    fee: float

class BacktestEngine:
    def __init__(self, portfolio: Portfolio, start_date: datetime, 
                 strategy: ModelStrategy, initial_cash: float = 1000.0):
        self._portfolio = portfolio
        self._start_date = start_date
        self._strategy = strategy
        self._initial_cash = initial_cash
        self._cash = initial_cash

        self.date = start_date
        self.transaction_log: list[Transaction] = []

        self.daily_history: list[dict] = []

    @property
    def cash(self):
        return self._cash
    
    @property
    def assets_values(self):
        return self._portfolio.get_value_at_date(self.date)
    
    @property
    def total_value(self):
        return self.assets_values + self.cash

    @property
    def daily_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.daily_history)
    
    def next_day(self):
        engine_state = EngineState(self.date, self._portfolio, self._cash)

        if self._strategy is not None:
            result: Optional[StrategyResult] = self._strategy.apply(engine_state)
            
            if result is not None:
                if result.type == 'BUY':
                    self.buy(self._strategy.instrument, result.amount)
                elif result.type == 'SELL':
                    self.sell(self._strategy.instrument.symbol, result.amount)

        self.daily_history.append({
            'Date': self.date,
            'Cash': self.cash,
            'Assets Value': self.assets_values,
            'Total Value': self.total_value
        })

        if self.date < datetime.now():
            self.date += timedelta(days=1)
            return False

        return True


    def buy(self, instrument: Instrument, amount: float, fee: float = 0.0):
        if self._cash < amount:
            raise ValueError(f'Insufficient cash to acquire the instrument for amount {amount}.')

        market_data = instrument.get_market_data_at_closest_trading_day(self.date)
        price = self._portfolio.convert_to_native_currency(market_data['close'], instrument.currency)

        volume = (amount - fee) / price if price != 0 else 0

        asset = Asset(instrument, volume, price, self.date)

        self._portfolio.add(asset)

        self._cash -= amount

        self.transaction_log.append(
            Transaction(instrument.symbol, self.date, 'BUY', volume, price, amount, fee)
        )

    def sell(self, symbol: str, amount: float, fee: float = 0.0):
        asset: Asset = self._portfolio.get_asset(symbol)

        if not asset:
            raise ValueError(f"Cannot sell {symbol}: position does not exist in portfolio.")


        market_data = asset.instrument.get_market_data_at_closest_trading_day(self.date)
        price = self._portfolio.convert_to_native_currency(market_data['close'], asset.currency)

        volume_to_sell = amount / price if price != 0 else 0
        if volume_to_sell > asset.volume:
            volume_to_sell = asset.volume

        self._portfolio.reduce_asset_exposure(symbol, volume_to_sell)

        if asset.volume <= 1e-6:
            self._portfolio.remove(asset.symbol)

        amount = volume_to_sell * price
        self._cash += amount - fee

        self.transaction_log.append(
            Transaction(symbol, self.date, 'SELL', volume_to_sell, price, amount, fee)
        )

    def show_trasaction_log(self):
        if not self.transaction_log:
            return "No transactions recorded."

        return '\n'.join(
            f"Transaction: {t.type} {t.volume:.4f} x {t.symbol} @ ${t.price:.2f} "
            f"| Total: ${t.total_amount:.2f} (Fee: ${t.fee:.2f})"
            for t in self.transaction_log
        )
    
    def show_transaction_log_df(self) -> pd.DataFrame:
        if not self.transaction_log:
            print("No transactions recorded.")
            return pd.DataFrame()
        
        data = [
            {
                "Type": t.type,
                "Symbol": t.symbol,
                "Volume": t.volume,
                "Price ($)": f"{t.price:,.2f}",
                "Total Amount ($)": f"{t.total_amount:,.2f}",
                "Fee ($)": f"{t.fee:,.2f}"
            }
            for t in self.transaction_log
        ]
        
        df = pd.DataFrame(data)
        return df.style.set_properties(**{'text-align': 'center'}).hide(axis="index")


def plot_strategy_vs_benchmark(engine, instrument, figsize=(12, 8)):
    """
    Plots daily strategy total portfolio value against a $1,000 Buy & Hold benchmark,
    complete with trade execution markers and drawdown analysis.
    """
    # 1. Extract Historical Market Data for Benchmark
    price_df = instrument.market_data_history.copy()
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)

    # Filter market data from backtest start date to engine end date
    price_df = price_df.loc[engine._start_date : engine.date]
    if price_df.empty:
        raise ValueError("No price history found within the backtest date range.")

    close_prices = price_df['Close']

    # Normalize Benchmark to initial cash ($1,000)
    initial_price = close_prices.iloc[0]
    benchmark_value = (close_prices / initial_price) * engine._initial_cash

    # 2. Extract Strategy Portfolio Value Curve
    # Ensure daily history was tracked during the backtest run
    if hasattr(engine, 'daily_history_df') and not engine.daily_history_df.empty:
        strategy_df = engine.daily_history_df.set_index('Date')
    else:
        # Fallback: create a single-line df if daily tracking wasn't stored
        strategy_df = pd.DataFrame(
            {'Total Value': [engine.total_value]}, index=[engine.date]
        )

    # Reindex strategy values to align with market trading days
    combined_df = pd.DataFrame({
        'Benchmark ($1,000)': benchmark_value,
        'Strategy Value': strategy_df['Total Value']
    }).ffill().dropna()

    # Calculate Drawdown
    peak = combined_df['Strategy Value'].cummax()
    drawdown = (combined_df['Strategy Value'] - peak) / peak * 100.0

    # 3. Create Plots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )
    plt.subplots_adjust(hspace=0.08)

    # --- Top Subplot: Value Comparison ---
    ax1.plot(
        combined_df.index, combined_df['Benchmark ($1,000)'], 
        label=f'Buy & Hold ({instrument.symbol})', color='#7f8c8d', linestyle='--', linewidth=1.5
    )
    ax1.plot(
        combined_df.index, combined_df['Strategy Value'], 
        label='Calibrated GBDT Strategy', color='#2980b9', linewidth=2.0
    )

    # Add Transaction Markers
    if engine.transaction_log:
        tx_df = pd.DataFrame([
            {'date': t.date, 'type': t.type, 'price': t.price} 
            for t in engine.transaction_log
        ])
        
        buys = tx_df[tx_df['type'] == 'BUY']
        sells = tx_df[tx_df['type'] == 'SELL']

        # Get portfolio value at exact trade dates
        buy_vals = combined_df.loc[combined_df.index.isin(buys['date']), 'Strategy Value']
        sell_vals = combined_df.loc[combined_df.index.isin(sells['date']), 'Strategy Value']

        ax1.scatter(
            buy_vals.index, buy_vals.values, 
            marker='^', color='#2ecc71', s=80, label='BUY Execution', zorder=5
        )
        ax1.scatter(
            sell_vals.index, sell_vals.values, 
            marker='v', color='#e74c3c', s=80, label='SELL Execution', zorder=5
        )

    ax1.set_title(
        f"Backtest Performance: {instrument.symbol} Strategy vs. $1,000 Buy & Hold", 
        fontsize=14, fontweight='bold', pad=12
    )
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    # --- Bottom Subplot: Strategy Drawdown ---
    ax2.plot(combined_df.index, drawdown, color='#c0392b', linewidth=1.2)
    ax2.fill_between(combined_df.index, drawdown, 0, color='#e74c3c', alpha=0.25)
    
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.yaxis.set_major_formatter('{x:.1f}%')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Format Date Axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()